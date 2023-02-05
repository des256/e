// dump of rust-lang.github.io/async-book, as a starting point

use {
    std::{
        sync::{
            mpsc::{
                sync_channel,
                Receiver,
                SyncSender,
            },
            Arc,
            Mutex,
        },
        boxed,
        future::Future,
        task::{
            Context,
            Waker,
            RawWaker,
            RawWakerVTable,
        },
        mem::ManuallyDrop,
        pin::Pin,
        marker::PhantomData,
        ops::Deref,
    },
};

// some pinned boxed object that implements Future outputting T, Send and outlives 'a
//type BoxFuture<'a,T> = Pin<Box<dyn Future<Output=T> + Send + 'a>>;

struct Task {
    future: Mutex<Option<Future<Output=()>>>,
    task_sender: SyncSender<Arc<Task>>,
}

// a waker reference that outlives 'a
pub struct WakerRef<'a> {
    waker: ManuallyDrop<Waker>,
    _marker: PhantomData<&'a ()>,
}

// implementation of the waker reference
impl<'a> WakerRef<'a> {

    // create new owned reference for a waker
    pub fn new(waker: &'a Waker) -> Self {
        let waker = ManuallyDrop::new(unsafe { std::ptr::read(waker) });
        Self { waker, _marker: PhantomData }
    }

    // create new unowned reference for a waker
    pub fn new_unowned(waker: ManuallyDrop<Waker>) -> Self {
        Self { waker, _marker: PhantomData }
    }
}

// clone raw waker
unsafe fn clone_arc_raw<T: ArcWake>(data: *const ()) -> RawWaker {
    let arc = ManuallyDrop::new(Arc::<T>::from_raw(data.cast::<T>()));
    let _arc_clone: ManuallyDrop<_> = arc.clone();
    RawWaker::new(data, waker_vtable::<T>())
}

// wake raw waker
unsafe fn wake_arc_raw<T: ArcWake>(data: *const ()) {
    let arc: Arc<T> = Arc::from_raw(data.cast::<T>());
    ArcWake::wake(arc);
}

// wake raw waker by reference
unsafe fn wake_by_ref_arc_raw<T: ArcWake>(data: *const ()) {
    let arc = ManuallyDrop::new(Arc::<T>::from_raw(data.cast::<T>()));
    ArcWake::wake_by_ref(&arc);
}

// drop raw waker
unsafe fn drop_arc_raw<T: ArcWake>(data: *const ()) {
    drop(Arc::<T>::from_raw(data.cast::<T>()))
}

// collect all waker functions into vtable
fn waker_vtable<W: ArcWake>() -> &'static RawWakerVTable {
    &RawWakerVTable::new(
        clone_arc_raw::<W>,
        wake_arc_raw::<W>,
        wake_by_ref_arc_raw::<W>,
        drop_arc_raw::<W>,
    )
}

// dereference waker reference into waker
impl Deref for WakerRef<'_> {
    type Target = Waker;
    fn deref(&self) -> &Waker {
        &self.waker
    }
}

// the executor with queue of tasks
pub struct Executor {
    ready_queue: Receiver<Arc<Task>>,
}

impl Executor {
    pub fn run(&self) {

        // get next task from the queue
        while let Ok(task) = self.ready_queue.recv() {

            // lock the future slot
            let mut future_slot = task.future.lock().unwrap();
            if let Some(mut future) = future_slot.take() {

                // execute the task until the next await, or until it's done
                let ptr = Arc::as_ptr(&task).cast::<()>();

                // create new waker for this future
                let waker = WakerRef::new_unowned(
                    ManuallyDrop::new(unsafe {
                        Waker::from_raw(RawWaker::new(ptr,waker_vtable::<Task>()))
                    })
                );
                let context = &mut Context::from_waker(&*waker);

                // if the future is still not complete, put it back in the slot
                if future.as_mut().poll(context).is_pending() {
                    *future_slot = Some(future);
                }
            }
        }
    }
}

pub struct Spawner {
    task_sender: SyncSender<Arc<Task>>,
}

impl Spawner {
    pub fn spawn(&self, future: impl Future<Output = ()> + 'static + Send) {
        let future = boxed::Box::pin(future);
        let task = Arc::new(Task {
            future: Mutex::new(Some(future)),
            task_sender: self.task_sender.clone(),
        });
        self.task_sender.send(task).expect("too many tasks queued");
    }
}

pub fn new_executor_and_spawner() -> (Executor, Spawner) {
    const MAX_QUEUED_TASKS: usize = 10000;
    let (task_sender,ready_queue) = sync_channel(MAX_QUEUED_TASKS);
    (Executor { ready_queue }, Spawner { task_sender })
}

// trait to wake queued objects
pub trait ArcWake: Send + Sync {
    fn wake_by_ref(arc_self: &Arc<Self>);
    fn wake(arc_self: Arc<Self>);
}

// implement ArcWake for the queued tasks
impl ArcWake for Task {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        let cloned = arc_self.clone();
        arc_self.task_sender.send(cloned).expect("too many tasks queued");
    }

    fn wake(arc_self: Arc<Self>) {
        let cloned = arc_self.clone();
        arc_self.task_sender.send(cloned).expect("too many tasks queued");
    }
}
