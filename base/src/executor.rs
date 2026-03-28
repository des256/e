use std::{
    cell::RefCell,
    collections::VecDeque,
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{Arc, RwLock},
    task::{
        Context,
        Poll,
        RawWaker,
        RawWakerVTable,
        Waker,
    },
};

type Task = Pin<Box<dyn Future<Output = ()>>>;

struct WakeData {
    ready_queue: Arc<Mutex<VecDeque<usize>>>,
    thread: std::thread::Thread,
    task_id: usize,
}

const VTABLE: RawWakerVTable = RawWakerVTable::new(
    |ptr| {
        // clone: increment Arc refcount
        let arc = unsafe { Arc::from_raw(ptr as *const WakeData) };
        let cloned = arc.clone();
        std::mem::forget(arc); // don't decrement original
        RawWaker::new(Arc::into_raw(cloned) as *const (), &VTABLE)
    },
    |ptr| {
        // wake: enqueue + unpark, consumes the Arc
        let arc = unsafe { Arc::from_raw(ptr as *const WakeData) };
        arc.ready_queue.lock().unwrap().push_back(arc.task_id);
        arc.thread.unpark();
        // arc drops here — correct
    },
    |ptr| {
        // wake_by_ref: enqueue + unpark, does NOT consume
        let arc = unsafe { &*(ptr as *const WakeData) };
        arc.ready_queue.lock().unwrap().push_back(arc.task_id);
        arc.thread.unpark();
    },
    |ptr| {
        // drop
        unsafe { Arc::from_raw(ptr as *const WakeData) };
    },
);

struct Executor {
    ready_queue: Arc<RwLock<VecDeque<usize>>>,
    tasks: Vec<Option<Task>>,
    main_thread: std::thread::Thread,
}

impl Executor {
    pub fn new() -> Self {

    }

    pub fn run(&mut self) {
        loop {
            // drain ready tasks
            let ready: Vec<usize> = self.ready_queue.write().drain(..).collect();

            // if all tasks complete, break loop
            if ready.is_empty() && self.tasks.iter().all(|task| task.is_none()) {
                break;
            }

            // poll
            for id in ready {
                let waker = self.make_waker(id);
                let mut context = Context::from_waker(&waker);
                if let Some(task) = &mut self.tasks[id] {
                    if task.as_mut().poll(&mut context).is_ready() {
                        self.tasks[id] = None;
                    }
                }
            }

            // if nothing is ready, park until waker fires
            if self.ready_queue.read().is_empty() {
                std::thread::park();
            }
        }
    }

    fn make_waker(data: Arc<WakeData>) -> Waker {
        let raw = RawWaker::new(Arc::into_raw(data) as *const (), &VTABLE);
        unsafe { Waker::from_raw(raw) }
    }
}
