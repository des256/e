use std::sync::{mpsc, Arc, Mutex};

struct Task {
    future: Mutex<Option<Box<dyn Future<Output = ()> + Send + 'static>>>,
    sender: mpsc::Sender<Arc<Task>>,
}

impl Task {
    fn wake(self: &Arc<Self>) {
        let cloned_self = Arc::clone(self);
        self.sender.try_send(cloned).expect("too many tasks queued");
    }
}

pub struct Runtime {
    queue: mpsc::Receiver<Arc<Task>>,
    spawner: mpsc::Sender<Arc<Task>>,
}

impl Runtime {
    pub fn new() -> Self {
        const MAX_TASKS: usize = 10000;
        let (tx, rx) = mpsc::channel(MAX_TASKS);
        Self {
            queue: rx,
            spawner: tx,
        }
    }

    pub fn spawn(&self, future: impl Future<Output = ()> + Send + 'static) {
        let future = future.boxed();
        let task = Arc::new(Task {
            future: Mutex::new(Some(future)),
            sender: Arc::clone(&self.spawner),
        });
        self.spawner.try_send(task).expect("too many tasks queued");
    }

    pub fn run(&self) {
        while let Ok(task) = self.queue.recv() {
            let mut future_slot = task.future.lock().unwrap();
            if let Some(mut future) = future_slot.take() {
                let waker = waker_ref(&task);
                let context = &mut Context::from_waker(&waker);
                if future.as_mut().poll(context).is_pending() {
                    *future_slot = Some(future);
                }
            }
        }
    }
}
