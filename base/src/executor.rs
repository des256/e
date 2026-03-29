use std::{
    cell::RefCell,
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

type Task = Pin<Box<dyn Future<Output = ()> + Send>>;

struct JoinState<T> {
    result: Option<T>,
    waker: Option<Waker>,
}

/// A handle to a spawned task that resolves to the task's return value.
///
/// `JoinHandle<T>` implements [`Future`], so it can be `.await`ed from within
/// another task to wait for the spawned task to complete and retrieve its result.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let sum = Executor::block_on(async {
///     let a = spawn(async { 2 + 3 }).await;
///     let b = spawn(async move { a * 10 }).await;
///     b
/// });
/// assert_eq!(sum, 50);
/// ```
pub struct JoinHandle<T> {
    state: Arc<Mutex<JoinState<T>>>,
}

impl<T> Future for JoinHandle<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut state = self.state.lock().unwrap();
        if let Some(result) = state.result.take() {
            Poll::Ready(result)
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

fn wrap_task<T: Send + 'static>(
    future: impl Future<Output = T> + Send + 'static,
) -> (Task, JoinHandle<T>) {
    let join_state = Arc::new(Mutex::new(JoinState {
        result: None,
        waker: None,
    }));
    let task_state = join_state.clone();
    let task: Task = Box::pin(async move {
        let result = future.await;
        let mut s = task_state.lock().unwrap();
        s.result = Some(result);
        if let Some(waker) = s.waker.take() {
            waker.wake();
        }
    });
    (task, JoinHandle { state: join_state })
}

// --- Shared executor state ---

struct SharedState {
    ready_queue: VecDeque<usize>,
    pending_spawns: Vec<Task>,
}

struct Shared {
    state: Mutex<SharedState>,
    condvar: Condvar,
}

// --- Waker ---

struct WakeData {
    shared: Arc<Shared>,
    task_id: usize,
}

const VTABLE: RawWakerVTable = RawWakerVTable::new(
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const WakeData) };
        let cloned = arc.clone();
        std::mem::forget(arc);
        RawWaker::new(Arc::into_raw(cloned) as *const (), &VTABLE)
    },
    |ptr| {
        let arc = unsafe { Arc::from_raw(ptr as *const WakeData) };
        let mut state = arc.shared.state.lock().unwrap();
        state.ready_queue.push_back(arc.task_id);
        drop(state);
        arc.shared.condvar.notify_one();
    },
    |ptr| {
        let data = unsafe { &*(ptr as *const WakeData) };
        let mut state = data.shared.state.lock().unwrap();
        state.ready_queue.push_back(data.task_id);
        drop(state);
        data.shared.condvar.notify_one();
    },
    |ptr| {
        unsafe { Arc::from_raw(ptr as *const WakeData) };
    },
);

thread_local! {
    static CURRENT: RefCell<Option<Spawner>> = const { RefCell::new(None) };
}

/// Spawns a new task on the current executor and returns a [`JoinHandle`] for
/// its result.
///
/// This is the primary way to create concurrent work from within async code.
/// The spawned task runs on the same single-threaded executor as the caller.
///
/// # Panics
///
/// Panics if called outside of [`Executor::block_on`].
///
/// # Examples
///
/// ```
/// use base::*;
///
/// Executor::block_on(async {
///     let handle = spawn(async { 42 });
///     assert_eq!(handle.await, 42);
/// });
/// ```
pub fn spawn<T: Send + 'static>(
    future: impl Future<Output = T> + Send + 'static,
) -> JoinHandle<T> {
    CURRENT.with(|c| {
        c.borrow()
            .as_ref()
            .expect("spawn() called outside executor")
            .spawn(future)
    })
}

// --- Spawner ---

#[derive(Clone)]
struct Spawner {
    shared: Arc<Shared>,
}

impl Spawner {
    fn spawn<T: Send + 'static>(
        &self,
        future: impl Future<Output = T> + Send + 'static,
    ) -> JoinHandle<T> {
        let (task, handle) = wrap_task(future);
        let mut state = self.shared.state.lock().unwrap();
        state.pending_spawns.push(task);
        drop(state);
        self.shared.condvar.notify_one();
        handle
    }
}

/// A single-threaded async executor.
///
/// The public API consists of [`Executor::block_on`] and the free function
/// [`spawn`]. Together they provide a minimal but complete async runtime:
///
/// ```
/// use base::*;
///
/// let result = Executor::block_on(async {
///     let a = spawn(async { 1 }).await;
///     let b = spawn(async { 2 }).await;
///     a + b
/// });
/// assert_eq!(result, 3);
/// ```
pub struct Executor {
    shared: Arc<Shared>,
    tasks: Vec<Option<Task>>,
}

impl Executor {
    fn new() -> Self {
        Executor {
            shared: Arc::new(Shared {
                state: Mutex::new(SharedState {
                    ready_queue: VecDeque::new(),
                    pending_spawns: Vec::new(),
                }),
                condvar: Condvar::new(),
            }),
            tasks: Vec::new(),
        }
    }

    fn spawner(&self) -> Spawner {
        Spawner {
            shared: Arc::clone(&self.shared),
        }
    }

    fn spawn<T: Send + 'static>(
        &mut self,
        future: impl Future<Output = T> + Send + 'static,
    ) -> JoinHandle<T> {
        let (task, handle) = wrap_task(future);
        let id = self.alloc(task);
        self.shared.state.lock().unwrap().ready_queue.push_back(id);
        handle
    }

    /// Runs a future to completion on a new single-threaded executor, returning
    /// its result.
    ///
    /// This is the main entry point for the async runtime. It creates an
    /// executor, drives the given future and any tasks it [`spawn`]s, and
    /// blocks the calling thread until all tasks complete.
    ///
    /// # Examples
    ///
    /// Returning a value:
    ///
    /// ```
    /// use base::*;
    ///
    /// let value = Executor::block_on(async { 1 + 1 });
    /// assert_eq!(value, 2);
    /// ```
    ///
    /// Spawning concurrent work:
    ///
    /// ```
    /// use base::*;
    ///
    /// Executor::block_on(async {
    ///     let handle = spawn(async { expensive_computation() });
    ///     // ... do other work ...
    ///     let result = handle.await;
    /// });
    ///
    /// # fn expensive_computation() -> i32 { 42 }
    /// ```
    pub fn block_on<T: Send + 'static>(
        future: impl Future<Output = T> + Send + 'static,
    ) -> T {
        let mut exec = Executor::new();
        let handle = exec.spawn(future);
        exec.run();
        let result = handle.state.lock().unwrap().result.take().expect("task did not complete");
        result
    }

    fn run(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = Some(self.spawner()));

        loop {
            self.drain_spawns();

            let ready: Vec<usize> =
                self.shared.state.lock().unwrap().ready_queue.drain(..).collect();

            for id in ready {
                if let Some(mut task) = self.tasks[id].take() {
                    let waker = self.make_waker(id);
                    let mut cx = Context::from_waker(&waker);
                    if task.as_mut().poll(&mut cx).is_pending() {
                        self.tasks[id] = Some(task);
                    }
                }
            }

            let state = self.shared.state.lock().unwrap();
            if state.ready_queue.is_empty() && state.pending_spawns.is_empty() {
                if self.tasks.iter().all(|t| t.is_none()) {
                    break;
                }
                let _state = self.shared.condvar.wait(state).unwrap();
            }
        }

        CURRENT.with(|c| *c.borrow_mut() = None);
    }

    fn alloc(&mut self, task: Task) -> usize {
        if let Some(id) = self.tasks.iter().position(|t| t.is_none()) {
            self.tasks[id] = Some(task);
            id
        } else {
            let id = self.tasks.len();
            self.tasks.push(Some(task));
            id
        }
    }

    fn drain_spawns(&mut self) {
        let spawns: Vec<Task> =
            self.shared.state.lock().unwrap().pending_spawns.drain(..).collect();
        for task in spawns {
            let id = self.alloc(task);
            self.shared.state.lock().unwrap().ready_queue.push_back(id);
        }
    }

    fn make_waker(&self, id: usize) -> Waker {
        let data = Arc::new(WakeData {
            shared: Arc::clone(&self.shared),
            task_id: id,
        });
        let raw = RawWaker::new(Arc::into_raw(data) as *const (), &VTABLE);
        unsafe { Waker::from_raw(raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    #[test]
    fn basic_task_completes() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        Executor::block_on(async move { c.fetch_add(1, Ordering::SeqCst); });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_tasks() {
        let counter = Arc::new(AtomicUsize::new(0));
        let inner = counter.clone();
        Executor::block_on(async move {
            for _ in 0..5 {
                let c = inner.clone();
                spawn(async move { c.fetch_add(1, Ordering::SeqCst); });
            }
        });
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn task_yields_and_resumes() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        Executor::block_on(async move {
            YieldOnce(false).await;
            c.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn wake_from_another_thread() {
        struct WakeFromThread(bool);
        impl Future for WakeFromThread {
            type Output = ();
            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                if self.0 {
                    Poll::Ready(())
                } else {
                    self.0 = true;
                    let waker = cx.waker().clone();
                    std::thread::spawn(move || { waker.wake(); });
                    Poll::Pending
                }
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        Executor::block_on(async move {
            WakeFromThread(false).await;
            c.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn free_spawn_from_task() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();
        Executor::block_on(async move {
            c1.fetch_add(1, Ordering::SeqCst);
            spawn(async move {
                c2.fetch_add(1, Ordering::SeqCst);
            });
        });
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn nested_free_spawn() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();
        let c3 = counter.clone();
        Executor::block_on(async move {
            c1.fetch_add(1, Ordering::SeqCst);
            spawn(async move {
                c2.fetch_add(1, Ordering::SeqCst);
                spawn(async move {
                    c3.fetch_add(1, Ordering::SeqCst);
                });
            });
        });
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn join_handle_chain() {
        let value = Executor::block_on(async {
            let a = spawn(async { 10usize }).await;
            let b = spawn(async move { a + 5 }).await;
            b
        });
        assert_eq!(value, 15);
    }

    #[test]
    fn block_on_returns_value() {
        let value = Executor::block_on(async { 99usize });
        assert_eq!(value, 99);
    }

    #[test]
    fn block_on_with_spawns() {
        let value = Executor::block_on(async {
            let a = spawn(async { 10usize }).await;
            let b = spawn(async move { a * 3 }).await;
            b
        });
        assert_eq!(value, 30);
    }

    #[test]
    fn slot_recycling() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let mut exec = Executor::new();
        exec.spawn(async move { c.fetch_add(1, Ordering::SeqCst); });
        exec.run();
        assert_eq!(exec.tasks.len(), 1);
        assert!(exec.tasks[0].is_none());

        let c = counter.clone();
        exec.spawn(async move { c.fetch_add(1, Ordering::SeqCst); });
        assert_eq!(exec.tasks.len(), 1);
        exec.run();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
