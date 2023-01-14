// taken from rust-lang.github.io/async-book to test the executor
// aspirations:
// - should not be part of E, atleast not in this way

use {
    std::{
        future::Future,
        pin::Pin,
        sync::{
            Arc,
            Mutex,
        },
        task::{
            Context,
            Waker,
            Poll,
        },
        thread,
        time::Duration,
    },
};

pub struct Timer {
    shared_state: Arc<Mutex<SharedState>>,
}

struct SharedState {
    completed: bool,
    waker: Option<Waker>,
}

impl Timer {
    pub fn new(duration: Duration) -> Self {
        let shared_state = Arc::new(Mutex::new(SharedState { completed: false, waker: None, }));
        let thread_shared_state = shared_state.clone();
        thread::spawn(move || {
            thread::sleep(duration);
            let mut shared_state = thread_shared_state.lock().unwrap();
            shared_state.completed = true;
            if let Some(waker) = shared_state.waker.take() {
                waker.wake()
            }
        });
        Timer { shared_state }
    }
}

impl Future for Timer {
    type Output = ();
    fn poll(self: Pin<&mut Self>,cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut shared_state = self.shared_state.lock().unwrap();
        if shared_state.completed {
            Poll::Ready(())
        }
        else {
            shared_state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
