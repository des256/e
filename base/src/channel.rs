use std::{
    collections::VecDeque,
    fmt,
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
};

struct Inner<T> {
    buffer: VecDeque<T>,
    recv_waker: Option<Waker>,
    sender_count: usize,
    receiver_alive: bool,
    condvar: Arc<Condvar>,
}

/// Error returned by [`Sender::send`] when the [`Receiver`] has been dropped.
///
/// Contains the value that could not be sent, recoverable via the `.0` field.
///
/// ```
/// use base::*;
///
/// let (tx, rx) = channel::<&str>();
/// drop(rx);
/// let err = tx.send("hello").unwrap_err();
/// assert_eq!(err.0, "hello");
/// ```
pub struct SendError<T>(pub T);

impl<T: fmt::Debug> fmt::Debug for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SendError").field(&self.0).finish()
    }
}

/// Error returned by [`Receiver::recv`].
#[derive(Debug, PartialEq, Eq)]
pub enum RecvError {
    /// All [`Sender`]s have been dropped and the buffer has been drained.
    /// No further values will ever be received.
    Disconnected,
}

/// Error returned by [`Receiver::try_recv`].
#[derive(Debug, PartialEq, Eq)]
pub enum TryRecvError {
    /// The channel buffer is currently empty, but at least one [`Sender`] is
    /// still alive so values may arrive in the future.
    Empty,
    /// All [`Sender`]s have been dropped and the buffer has been drained.
    /// No further values will ever be received.
    Disconnected,
}

/// The sending half of an unbounded MPSC channel, created by [`channel`].
///
/// `Sender` is [`Clone`]: each clone is an independent producer that can send
/// values from any thread. When the last `Sender` is dropped the channel
/// closes and [`Receiver::recv`] returns `None`.
///
/// # Examples
///
/// Sending from a worker thread:
///
/// ```
/// use base::*;
///
/// let (tx, rx) = channel();
/// std::thread::spawn(move || {
///     tx.send(42).unwrap();
/// });
/// Executor::block_on(async move {
///     assert_eq!(rx.recv().await, Some(42));
/// });
/// ```
pub struct Sender<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.inner.lock().unwrap().sender_count += 1;
        Sender {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        inner.sender_count -= 1;
        if inner.sender_count == 0 {
            if let Some(waker) = inner.recv_waker.take() {
                waker.wake();
            }
            inner.condvar.notify_one();
        }
    }
}

impl<T> Sender<T> {
    /// Sends a value on the channel, returning [`Err(SendError)`](SendError)
    /// if the [`Receiver`] has been dropped.
    ///
    /// This call blocks only for the duration of an internal mutex acquisition
    /// and is safe to use from synchronous (worker-thread) contexts.
    ///
    /// # Examples
    ///
    /// ```
    /// use base::*;
    ///
    /// let (tx, rx) = channel();
    /// tx.send(1).unwrap();
    /// tx.send(2).unwrap();
    /// assert_eq!(rx.try_recv(), Ok(1));
    /// assert_eq!(rx.try_recv(), Ok(2));
    /// ```
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.receiver_alive {
            return Err(SendError(value));
        }
        inner.buffer.push_back(value);
        if let Some(waker) = inner.recv_waker.take() {
            waker.wake();
        }
        inner.condvar.notify_one();
        Ok(())
    }
}

/// The receiving half of an unbounded MPSC channel, created by [`channel`].
///
/// Not cloneable — there is exactly one consumer. Values arrive in FIFO
/// order. When all [`Sender`]s are dropped, [`recv`](Receiver::recv) returns
/// `None` and [`try_recv`](Receiver::try_recv) returns
/// [`Err(TryRecvError::Disconnected)`](TryRecvError::Disconnected) once the
/// buffer is drained.
///
/// # Examples
///
/// Async receive loop:
///
/// ```
/// use base::*;
///
/// Executor::block_on(async {
///     let (tx, rx) = channel();
///     tx.send(1).unwrap();
///     tx.send(2).unwrap();
///     drop(tx);
///
///     let mut values = Vec::new();
///     while let Some(v) = rx.recv().await {
///         values.push(v);
///     }
///     assert_eq!(values, vec![1, 2]);
/// });
/// ```
pub struct Receiver<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.inner.lock().unwrap().receiver_alive = false;
    }
}

impl<T> Receiver<T> {
    /// Returns a future that resolves to the next value in the channel.
    ///
    /// The future yields `Some(value)` when a value is available. It yields
    /// `None` when all [`Sender`]s have been dropped and the buffer is
    /// drained, signalling that no further values will arrive.
    ///
    /// If the buffer is empty and senders are still alive, the future parks
    /// the current task (via its [`Waker`](std::task::Waker)) and is woken
    /// when a value is sent or the last sender is dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use base::*;
    ///
    /// let value = Executor::block_on(async {
    ///     let (tx, rx) = channel();
    ///     tx.send("hello").unwrap();
    ///     drop(tx);
    ///     let first = rx.recv().await;  // Some("hello")
    ///     let end   = rx.recv().await;  // None
    ///     (first, end)
    /// });
    /// assert_eq!(value, (Some("hello"), None));
    /// ```
    pub fn recv(&self) -> impl Future<Output = Option<T>> + '_ {
        Recv { receiver: self }
    }

    /// Blocks the calling thread until a value is available or all
    /// [`Sender`]s have been dropped.
    ///
    /// Returns `Some(value)` when a value is received, or `None` when all
    /// senders have been dropped and the buffer is drained.
    ///
    /// This method is intended for synchronous (non-async) contexts. Do not
    /// call it from within an async task — use [`recv`](Receiver::recv)
    /// instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use base::*;
    ///
    /// let (tx, rx) = channel();
    /// std::thread::spawn(move || {
    ///     tx.send(42).unwrap();
    /// });
    /// assert_eq!(rx.blocking_recv(), Some(42));
    /// assert_eq!(rx.blocking_recv(), None);
    /// ```
    pub fn blocking_recv(&self) -> Option<T> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(value) = inner.buffer.pop_front() {
                return Some(value);
            }
            if inner.sender_count == 0 {
                return None;
            }
            inner = inner.condvar.clone().wait(inner).unwrap();
        }
    }

    /// Attempts to receive a value without parking the current task.
    ///
    /// Returns:
    /// - `Ok(value)` if a value was available.
    /// - [`Err(TryRecvError::Empty)`](TryRecvError::Empty) if the buffer is
    ///   empty but senders are still alive.
    /// - [`Err(TryRecvError::Disconnected)`](TryRecvError::Disconnected) if
    ///   all senders have been dropped and the buffer is drained.
    ///
    /// This method does not register a waker and is safe to call from any
    /// context.
    ///
    /// # Examples
    ///
    /// ```
    /// use base::*;
    ///
    /// let (tx, rx) = channel::<i32>();
    /// assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    ///
    /// tx.send(1).unwrap();
    /// assert_eq!(rx.try_recv(), Ok(1));
    ///
    /// drop(tx);
    /// assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    /// ```
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(value) = inner.buffer.pop_front() {
            Ok(value)
        } else if inner.sender_count == 0 {
            Err(TryRecvError::Disconnected)
        } else {
            Err(TryRecvError::Empty)
        }
    }
}

struct Recv<'a, T> {
    receiver: &'a Receiver<T>,
}

impl<T> Future for Recv<'_, T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        let mut inner = self.receiver.inner.lock().unwrap();
        if let Some(value) = inner.buffer.pop_front() {
            Poll::Ready(Some(value))
        } else if inner.sender_count == 0 {
            Poll::Ready(None)
        } else {
            inner.recv_waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Creates an unbounded MPSC (multi-producer, single-consumer) channel.
///
/// Returns a ([`Sender`], [`Receiver`]) pair. The sender half is cloneable,
/// allowing multiple producers. The channel is unbounded: [`Sender::send`]
/// never blocks on capacity.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use base::*;
///
/// let (tx, rx) = channel();
/// tx.send("hello").unwrap();
/// assert_eq!(rx.try_recv(), Ok("hello"));
/// ```
///
/// Multiple producers with async receiver:
///
/// ```
/// use base::*;
///
/// Executor::block_on(async {
///     let (tx, rx) = channel();
///     let tx2 = tx.clone();
///     tx.send(1).unwrap();
///     tx2.send(2).unwrap();
///     drop(tx);
///     drop(tx2);
///
///     assert_eq!(rx.recv().await, Some(1));
///     assert_eq!(rx.recv().await, Some(2));
///     assert_eq!(rx.recv().await, None);
/// });
/// ```
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let condvar = Arc::new(Condvar::new());
    let inner = Arc::new(Mutex::new(Inner {
        buffer: VecDeque::new(),
        recv_waker: None,
        sender_count: 1,
        receiver_alive: true,
        condvar,
    }));
    (
        Sender {
            inner: inner.clone(),
        },
        Receiver { inner },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Executor;

    #[test]
    fn send_and_try_recv() {
        let (tx, rx) = channel();
        tx.send(42).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 42);
    }

    #[test]
    fn try_recv_empty() {
        let (_tx, rx) = channel::<i32>();
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn try_recv_disconnected() {
        let (tx, rx) = channel::<i32>();
        drop(tx);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn try_recv_drain_then_disconnect() {
        let (tx, rx) = channel();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn send_after_receiver_dropped_returns_value() {
        let (tx, rx) = channel();
        drop(rx);
        let err = tx.send(42).unwrap_err();
        assert_eq!(err.0, 42);
    }

    #[test]
    fn multiple_senders() {
        let (tx1, rx) = channel();
        let tx2 = tx1.clone();
        let tx3 = tx1.clone();
        tx1.send(1).unwrap();
        tx2.send(2).unwrap();
        tx3.send(3).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv().unwrap(), 3);
    }

    #[test]
    fn disconnect_requires_all_senders_dropped() {
        let (tx1, rx) = channel::<i32>();
        let tx2 = tx1.clone();
        drop(tx1);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
        drop(tx2);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn fifo_ordering() {
        let (tx, rx) = channel();
        for i in 0..10 {
            tx.send(i).unwrap();
        }
        for i in 0..10 {
            assert_eq!(rx.try_recv().unwrap(), i);
        }
    }

    #[test]
    fn async_recv_immediate() {
        let (tx, rx) = channel();
        tx.send(99).unwrap();
        let value = Executor::block_on(async move { rx.recv().await });
        assert_eq!(value, Some(99));
    }

    #[test]
    fn async_recv_disconnected() {
        let (tx, rx) = channel::<i32>();
        drop(tx);
        let value = Executor::block_on(async move { rx.recv().await });
        assert_eq!(value, None);
    }

    #[test]
    fn async_recv_values_then_disconnect() {
        let (tx, rx) = channel();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        let values = Executor::block_on(async move {
            let a = rx.recv().await;
            let b = rx.recv().await;
            let c = rx.recv().await;
            (a, b, c)
        });
        assert_eq!(values, (Some(1), Some(2), None));
    }

    #[test]
    fn async_recv_from_thread() {
        let result = Executor::block_on(async {
            let (tx, rx) = channel();
            std::thread::spawn(move || {
                tx.send(42).unwrap();
            });
            rx.recv().await
        });
        assert_eq!(result, Some(42));
    }

    #[test]
    fn async_recv_multiple_from_thread() {
        let result = Executor::block_on(async {
            let (tx, rx) = channel();
            std::thread::spawn(move || {
                for i in 0..5 {
                    tx.send(i).unwrap();
                }
            });
            let mut values = Vec::new();
            for _ in 0..5 {
                values.push(rx.recv().await.unwrap());
            }
            values
        });
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn async_recv_multiple_sender_threads() {
        let result = Executor::block_on(async {
            let (tx, rx) = channel();
            for i in 0..3 {
                let sender = tx.clone();
                std::thread::spawn(move || {
                    sender.send(i).unwrap();
                });
            }
            drop(tx);
            let mut values = Vec::new();
            while let Some(v) = rx.recv().await {
                values.push(v);
            }
            values.sort();
            values
        });
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn blocking_recv_immediate() {
        let (tx, rx) = channel();
        tx.send(42).unwrap();
        assert_eq!(rx.blocking_recv(), Some(42));
    }

    #[test]
    fn blocking_recv_disconnected() {
        let (tx, rx) = channel::<i32>();
        drop(tx);
        assert_eq!(rx.blocking_recv(), None);
    }

    #[test]
    fn blocking_recv_drain_then_disconnect() {
        let (tx, rx) = channel();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        assert_eq!(rx.blocking_recv(), Some(1));
        assert_eq!(rx.blocking_recv(), Some(2));
        assert_eq!(rx.blocking_recv(), None);
    }

    #[test]
    fn blocking_recv_from_thread() {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            tx.send(99).unwrap();
        });
        assert_eq!(rx.blocking_recv(), Some(99));
        assert_eq!(rx.blocking_recv(), None);
    }

    #[test]
    fn blocking_recv_multiple_from_thread() {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for i in 0..5 {
                tx.send(i).unwrap();
            }
        });
        let mut values = Vec::new();
        while let Some(v) = rx.blocking_recv() {
            values.push(v);
        }
        assert_eq!(values, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn blocking_recv_multiple_sender_threads() {
        let (tx, rx) = channel();
        for i in 0..3 {
            let sender = tx.clone();
            std::thread::spawn(move || {
                sender.send(i).unwrap();
            });
        }
        drop(tx);
        let mut values = Vec::new();
        while let Some(v) = rx.blocking_recv() {
            values.push(v);
        }
        values.sort();
        assert_eq!(values, vec![0, 1, 2]);
    }
}
