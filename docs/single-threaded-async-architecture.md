# Single-Threaded Async Runtime with Worker Threads in Rust

## High-Level Architecture

```
┌─────────────────────────────────────────────────┐
│                 Main Thread                       │
│  ┌───────────────────────────────────────────┐   │
│  │  Single-threaded Executor                  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐     │   │
│  │  │ Task A  │ │ Task B  │ │ Task C  │ ... │   │
│  │  └─────────┘ └─────────┘ └─────────┘     │   │
│  │                                            │   │
│  │  Rc<RefCell<T>> shared state (no Arc!)     │   │
│  │  Waker registry                            │   │
│  └───────────────────────────────────────────┘   │
│       ▲                          │                │
│       │ async recv               │ send command   │
│  ┌────┴──────┐             ┌─────▼──────┐        │
│  │ rx (async)│             │ tx (sync)  │        │
│  └────┬──────┘             └─────┬──────┘        │
└───────┼──────────────────────────┼────────────────┘
        │                          │
   ─────┼──────────────────────────┼───── thread boundary
        │                          │
┌───────┼──────────────────────────┼────────────────┐
│  ┌────┴──────┐             ┌─────▼──────┐        │
│  │ tx (sync) │             │ rx (sync)  │        │
│  └───────────┘             └────────────┘        │
│                                                   │
│  Worker Thread(s) — blocking I/O, inference, etc. │
│  Arc<Mutex<T>> or channel-only communication      │
└───────────────────────────────────────────────────┘
```

---

## 1. The Executor

The executor is the event loop that polls `Future`s on the main thread. Since it's single-threaded, it never needs `Send` or `Sync` on its tasks.

### Minimal Executor Design

```rust
use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

type Task = Pin<Box<dyn Future<Output = ()>>>;  // no Send bound!

struct Executor {
    ready_queue: Rc<RefCell<VecDeque<usize>>>,
    tasks: Vec<Option<Task>>,
}
```

### The Poll Loop

The executor has two main design choices:

**Option A: Spin loop (simplest, burns CPU)**
```rust
loop {
    while let Some(id) = self.ready_queue.borrow_mut().pop_front() {
        let waker = self.make_waker(id);
        let mut cx = Context::from_waker(&waker);
        if let Some(task) = &mut self.tasks[id] {
            if task.as_mut().poll(&mut cx).is_ready() {
                self.tasks[id] = None;
            }
        }
    }
    // Spin or yield — wasteful
    std::thread::yield_now();
}
```

**Option B: Park/unpark (recommended)**
```rust
use std::sync::{Arc, Mutex};

// The ready_queue must be Arc<Mutex<>> because worker threads
// need to wake the executor from across the thread boundary
struct Executor {
    ready_queue: Arc<Mutex<VecDeque<usize>>>,
    tasks: Vec<Option<Task>>,  // only accessed on main thread
    main_thread: std::thread::Thread,
}

fn run(&mut self) {
    loop {
        // Drain all ready tasks
        let ready: Vec<usize> = self.ready_queue.lock().unwrap().drain(..).collect();

        if ready.is_empty() && self.tasks.iter().all(|t| t.is_none()) {
            break; // All tasks complete
        }

        for id in ready {
            let waker = self.make_waker(id);
            let mut cx = Context::from_waker(&waker);
            if let Some(task) = &mut self.tasks[id] {
                if task.as_mut().poll(&mut cx).is_ready() {
                    self.tasks[id] = None;
                }
            }
        }

        // If nothing is ready, park until a waker fires
        if self.ready_queue.lock().unwrap().is_empty() {
            std::thread::park(); // <-- blocks, zero CPU
        }
    }
}
```

**Option C: epoll/kqueue integration (for timer/fd readiness)**

If you ever need timers or socket readiness on the main thread itself (not delegated to workers), you'd integrate `epoll_wait`/`kevent` as the blocking primitive instead of `thread::park`. This is what mio does. For this design where all I/O lives on workers, park/unpark is sufficient.

### The Waker

The waker is the bridge between worker threads and the executor. When a worker finishes, it wakes the corresponding task:

```rust
fn make_waker(&self, task_id: usize) -> Waker {
    let ready_queue = self.ready_queue.clone();
    let thread = self.main_thread.clone();

    // This is the simplest approach using Arc.
    // For zero-alloc wakers, you'd use RawWaker — see caveat below.
    let wake_state = Arc::new((ready_queue, thread, task_id));

    fn wake(state: Arc<(Arc<Mutex<VecDeque<usize>>>, std::thread::Thread, usize)>) {
        state.0.lock().unwrap().push_back(state.2);
        state.1.unpark();
    }

    // Use the nightly Wake trait or build a RawWaker.
    // Shown conceptually:
    unsafe { waker_from_arc(wake_state, wake) }
}
```

**Caveat: `RawWaker` is error-prone.** The `RawWakerVTable` requires you to manually implement clone/wake/drop with raw pointers. A common bug is double-free or forgetting to clone the Arc when `clone` is called on the RawWaker. The stable-recommended pattern:

```rust
use std::task::{RawWaker, RawWakerVTable, Waker};

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

fn make_waker(data: Arc<WakeData>) -> Waker {
    let raw = RawWaker::new(Arc::into_raw(data) as *const (), &VTABLE);
    unsafe { Waker::from_raw(raw) }
}
```

On nightly, `impl Wake for WakeData` avoids all of this.

---

## 2. Async-Aware MPSC Channel

This is the critical piece. You need a channel where:
- **Receiver** is async (polls on the main thread, registers a waker)
- **Sender** is sync (called from worker threads via `Arc`)

### Implementation

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::task::Waker;

struct ChannelInner<T> {
    queue: VecDeque<T>,
    waker: Option<Waker>,
    closed: bool,
}

// Sender: Clone + Send + Sync — lives on worker threads
pub struct Sender<T> {
    inner: Arc<Mutex<ChannelInner<T>>>,
}

// Receiver: !Send, !Sync — lives on main thread only
pub struct Receiver<T> {
    inner: Arc<Mutex<ChannelInner<T>>>,
    _not_send: PhantomData<Rc<()>>, // prevent sending to another thread
}
```

**Send side (blocking, called from workers):**
```rust
impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return Err(SendError(value));
        }
        inner.queue.push_back(value);
        if let Some(waker) = inner.waker.take() {
            waker.wake(); // wakes the executor's thread::park
        }
        Ok(())
    }
}
```

**Receive side (async, polled on main thread):**
```rust
impl<T> Receiver<T> {
    pub fn recv(&self) -> RecvFuture<'_, T> {
        RecvFuture { receiver: self }
    }
}

struct RecvFuture<'a, T> {
    receiver: &'a Receiver<T>,
}

impl<'a, T> Future for RecvFuture<'a, T> {
    type Output = Option<T>; // None = channel closed

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = self.receiver.inner.lock().unwrap();
        if let Some(value) = inner.queue.pop_front() {
            Poll::Ready(Some(value))
        } else if inner.closed {
            Poll::Ready(None)
        } else {
            inner.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
```

### Design Decisions and Tradeoffs

**Unbounded vs. bounded channel:**

| | Unbounded | Bounded |
|---|---|---|
| Worker side | Never blocks on send | Must block (or return error) when full |
| Backpressure | None — workers can overwhelm main thread | Natural backpressure |
| Memory | Unbounded growth risk | Capped |
| Complexity | Simple | Need a Condvar for blocking sender |

For bounded channels, the sender blocks via a `Condvar` when the queue is full:

```rust
pub fn send(&self, value: T) -> Result<(), SendError<T>> {
    let mut inner = self.inner.lock().unwrap();
    while inner.queue.len() >= inner.capacity {
        inner = self.not_full.wait(inner).unwrap(); // block worker
    }
    inner.queue.push_back(value);
    if let Some(waker) = inner.waker.take() {
        waker.wake();
    }
    Ok(())
}
```

And the receiver signals `not_full` after consuming:
```rust
// In poll(), after pop_front() succeeds:
self.receiver.not_full.notify_one();
```

**Recommendation:** Start unbounded. Add bounded later only if you observe memory pressure. The complexity of bounded channels with async receivers and blocking senders is non-trivial.

**Single vs. multiple receivers:**

This design naturally has a single receiver (main thread) and multiple senders (workers). This is the standard mpsc pattern. Don't build multi-consumer unless you need it — it adds significant complexity.

---

## 3. Rc<RefCell<>> on the Main Thread

Since the executor is single-threaded, tasks can share state without atomics:

```rust
use std::cell::RefCell;
use std::rc::Rc;

struct AppState {
    connections: Vec<ConnectionInfo>,
    model_status: ModelStatus,
    pending_requests: HashMap<RequestId, oneshot::Sender<Response>>,
}

// Shared across all tasks on the main thread
let state = Rc::new(RefCell::new(AppState::default()));

// In task A:
let state = state.clone(); // Rc clone, cheap
executor.spawn(async move {
    let mut s = state.borrow_mut();
    s.connections.push(new_conn);
});

// In task B:
let state = state.clone();
executor.spawn(async move {
    let s = state.borrow();
    println!("connections: {}", s.connections.len());
});
```

### Gotchas

**1. RefCell panics at runtime if you borrow_mut while already borrowed:**

```rust
// PANIC: this will blow up
let s1 = state.borrow();
let s2 = state.borrow_mut(); // panics!
```

This seems obvious in sequential code, but with async it's subtle:

```rust
async fn bad(state: Rc<RefCell<AppState>>) {
    let mut s = state.borrow_mut();
    // If this .await yields (returns Pending), the borrow_mut guard
    // is held across the yield point. Any other task that tries to
    // borrow will panic.
    let response = some_channel.recv().await;  // <-- yield point!
    s.process(response);
}
```

**Rule: Never hold a RefCell borrow across an `.await` point.** Instead:

```rust
async fn good(state: Rc<RefCell<AppState>>) {
    let response = some_channel.recv().await;
    // Borrow only for the synchronous section
    let mut s = state.borrow_mut();
    s.process(response);
    // s drops here before any await
}
```

**2. Rc<RefCell<T>> is !Send — tasks cannot be sent to other threads.** This is actually a feature in this design — the compiler enforces that these tasks only run on the main thread. But it means the `Task` type cannot have a `Send` bound, which is why tokio's `spawn` won't work with these types (it requires `Send`).

**3. Consider splitting state into multiple RefCells** to reduce contention:

```rust
struct AppState {
    connections: Rc<RefCell<Vec<ConnectionInfo>>>,
    model: Rc<RefCell<ModelStatus>>,
    requests: Rc<RefCell<HashMap<RequestId, ...>>>,
}
```

This way, one task can mutably borrow `connections` while another reads `model`.

---

## 4. Worker Thread Pool

### Option A: Dedicated threads per subsystem

```rust
// Each subsystem gets its own long-lived thread
let (http_tx, http_rx) = channel::<HttpCommand>();
std::thread::spawn(move || {
    // This thread owns an HTTP client (reqwest::blocking, ureq, etc.)
    let client = ureq::agent();
    for cmd in http_rx {
        match cmd {
            HttpCommand::Get { url, reply_tx } => {
                let result = client.get(&url).call();
                reply_tx.send(result).ok();
            }
        }
    }
});

let (ws_tx, ws_rx) = channel::<WsCommand>();
std::thread::spawn(move || {
    // WebSocket thread with tungstenite
    // ...
});

let (inference_tx, inference_rx) = channel::<InferenceCommand>();
std::thread::spawn(move || {
    // ML inference thread — holds the model in memory
    let model = load_model("weights.bin");
    for cmd in inference_rx {
        let result = model.infer(cmd.input);
        cmd.reply_tx.send(result).ok();
    }
});
```

**Pros:** Simple, each thread can hold expensive resources (model weights, connection pools). No thread scheduling overhead. Subsystems are naturally isolated.

**Cons:** Fixed thread count regardless of load. Idle threads waste OS resources (though they sleep on channel recv, so minimal CPU).

### Option B: Generic thread pool

```rust
use std::sync::{mpsc, Arc, Mutex};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    sender: mpsc::Sender<Job>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));

        for _ in 0..size {
            let rx = rx.clone();
            std::thread::spawn(move || {
                while let Ok(job) = rx.lock().unwrap().recv() {
                    job();
                }
            });
        }

        ThreadPool { sender: tx }
    }

    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.sender.send(Box::new(f)).unwrap();
    }
}
```

**Pros:** Efficient use of threads, automatic load balancing, handles bursty workloads.

**Cons:** No thread affinity — can't hold per-thread resources like GPU contexts or loaded models. The `Arc<Mutex<Receiver>>` pattern means workers contend on the lock to receive jobs (use crossbeam's work-stealing deque for better performance at scale).

### Option C: Hybrid (recommended)

- **Dedicated threads** for stateful subsystems (inference with loaded model, persistent WebSocket connections)
- **Thread pool** for stateless work (HTTP requests, file I/O, CPU-bound computation)

```
Main Thread (executor)
├── Channel → Inference Thread (holds model, dedicated)
├── Channel → WebSocket Thread (holds connections, dedicated)
└── Channel → Thread Pool (HTTP, file I/O, misc)
                ├── Worker 0
                ├── Worker 1
                └── Worker 2
```

---

## 5. Communication Patterns

### Request-Reply with Oneshot Channels

The main thread sends a command and gets a future that resolves when the worker replies:

```rust
// Oneshot: single-value, single-use channel
struct OneshotSender<T> {
    inner: Arc<Mutex<OneshotInner<T>>>,
}

struct OneshotReceiver<T> {
    inner: Arc<Mutex<OneshotInner<T>>>,
}

struct OneshotInner<T> {
    value: Option<T>,
    waker: Option<Waker>,
    closed: bool,
}

impl<T> Future for OneshotReceiver<T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(val) = inner.value.take() {
            Poll::Ready(Some(val))
        } else if inner.closed {
            Poll::Ready(None)
        } else {
            inner.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
```

Usage from the main thread:

```rust
async fn fetch_url(pool_tx: &Sender<PoolJob>, url: String) -> Result<String, Error> {
    let (reply_tx, reply_rx) = oneshot::channel();
    pool_tx.send(PoolJob::Http { url, reply_tx })?;
    reply_rx.await.ok_or(Error::WorkerDied)
}
```

### Event Streams from Workers

For things like WebSocket messages that arrive continuously:

```rust
// Worker thread pushes events
ws_event_tx.send(WsEvent::Message(msg))?;
ws_event_tx.send(WsEvent::Disconnected)?;

// Main thread has a task that continuously receives
async fn ws_event_loop(rx: Receiver<WsEvent>, state: Rc<RefCell<AppState>>) {
    while let Some(event) = rx.recv().await {
        match event {
            WsEvent::Message(msg) => {
                state.borrow_mut().handle_ws_message(msg);
            }
            WsEvent::Disconnected => {
                state.borrow_mut().mark_disconnected();
            }
        }
    }
}
```

---

## 6. Complete Skeleton

```rust
fn main() {
    let mut executor = Executor::new();
    let state = Rc::new(RefCell::new(AppState::new()));

    // Spawn worker infrastructure
    let (cmd_tx, cmd_rx) = async_channel::channel();   // main → workers
    let (event_tx, event_rx) = async_channel::channel(); // workers → main

    // Inference thread (dedicated)
    let event_tx2 = event_tx.clone_sync(); // sync sender clone
    let cmd_rx2 = cmd_rx.clone_sync();
    std::thread::spawn(move || {
        let model = load_model();
        while let Ok(cmd) = cmd_rx2.recv_blocking() {
            if let Command::Infer(input, reply) = cmd {
                let result = model.run(input);
                reply.send(result).ok();
            }
        }
    });

    // Thread pool for HTTP/file I/O
    let pool = ThreadPool::new(4);

    // Main async task: orchestrator
    let state2 = state.clone();
    executor.spawn(async move {
        // Send an HTTP request via the pool
        let body = fetch_url(&pool, "https://api.example.com/data").await?;
        state2.borrow_mut().latest_data = body;

        // Run inference
        let prediction = infer(&cmd_tx, state2.borrow().latest_data.clone()).await?;
        state2.borrow_mut().prediction = prediction;

        Ok::<(), Error>(())
    });

    // Event listener task
    let state3 = state.clone();
    executor.spawn(async move {
        while let Some(event) = event_rx.recv().await {
            state3.borrow_mut().handle_event(event);
        }
    });

    executor.run(); // blocks until all tasks complete
}
```

---

## 7. Comprehensive Pros, Cons, and Gotchas

### Pros of this architecture

| Benefit | Why |
|---|---|
| No `Send`/`Sync` tax on main thread | `Rc<RefCell<>>` is cheaper than `Arc<Mutex<>>`. No atomic operations for shared state. |
| No hidden complexity | You control the poll loop, waker mechanics, and scheduling. No runtime "magic". |
| Minimal dependencies | No tokio, no async-std. Just `std`. |
| Deterministic task scheduling | Single-threaded = no data races, no scheduling surprises. Tasks run to completion between yield points. |
| Worker isolation | Workers are plain threads. Use any blocking library (ureq, tungstenite, onnxruntime). No `spawn_blocking` gymnastics. |

### Cons and risks

| Risk | Mitigation |
|---|---|
| **Blocking the executor** — if any task does CPU-heavy or blocking work, all tasks stall | Strict discipline: main thread is coordination only. ALL blocking work goes to workers. Lint for this in code review. |
| **RefCell across await** — runtime panics | Clippy lint `await_holding_refcell_ref` (nightly). Manual code review. Structure code so borrows are always in synchronous blocks. |
| **Waker correctness** — bugs in RawWaker cause UB or hangs | Test thoroughly. Consider using `futures::task::ArcWake` as a dependency (tiny, no runtime) instead of hand-rolling. |
| **No ecosystem compatibility** — can't use `tokio::spawn`, `reqwest` (async), or most async libraries | Stick to blocking libraries on workers. For the main thread, you can only use futures that are runtime-agnostic (most in `futures` crate work). |
| **Deadlock potential** — bounded channel + synchronous call pattern | Main thread sends to worker, worker sends reply. If the command channel is bounded and full, and the main thread is blocked waiting to send while the worker is blocked waiting to reply... deadlock. Use unbounded for command channels, or ensure send and recv never happen synchronously on the same execution path. |
| **No timer/sleep** — `std` has no async timer | Build your own: maintain a BinaryHeap of `(Instant, Waker)` in the executor. Use `thread::park_timeout` instead of `thread::park`, with timeout = time to next timer expiry. |
| **No `select!`** — can't race multiple futures without a macro | Implement a `Select` combinator or use the `futures` crate's `select!`. It's just a future that polls multiple sub-futures. |

### Subtle gotchas

**1. Spurious wakes are legal.** The executor must handle a task being woken and then returning `Pending` again. Never assume a wake means the future is ready.

**2. Waker deduplication.** If a future is polled, registers a waker, gets woken, but isn't polled before being woken *again*, the second wake must not be lost. The ready_queue should handle duplicate task IDs gracefully (either deduplicate on insert, or just poll twice — the second poll returns Pending, which is harmless).

**3. Drop ordering.** When the executor shuts down, tasks are dropped. If a task holds the `Sender` half of a channel, dropping it closes the channel. Workers doing `rx.recv()` will get an error and should exit. Design workers to exit cleanly on channel close.

**4. Panic propagation.** If a worker panics, the `Sender` it holds is dropped, closing the channel. The main thread's `recv().await` returns `None`. You must handle this — otherwise the main thread hangs waiting for a reply that never comes. Use `std::thread::Builder::spawn` and keep the `JoinHandle` to detect panics.

**5. Stack size for inference threads.** Deep learning inference can use significant stack space. Use `std::thread::Builder::new().stack_size(8 * 1024 * 1024)` for inference workers.

**6. The `Mutex` in the async channel is held very briefly** (just push/pop + waker clone), so contention is low. But if you see lock contention at scale, replace with a lock-free queue (`crossbeam-queue::SegQueue`) and an `AtomicWaker` pattern.

---

## 8. Alternatives Avoided and Why That's (Mostly) Fine

| What you skip | What you lose | Does it matter for this design? |
|---|---|---|
| tokio | Multi-threaded executor, `spawn_blocking`, timers, I/O driver | No — workers handle I/O. Build your own timer. |
| async-std | Similar to tokio | Same |
| smol | Lightweight runtime | Closest to what you're building. If you ever reconsider, smol is the least invasive option. |
| `futures` crate | `select!`, `join!`, `FuturesUnordered`, `Stream` | Consider pulling in `futures-core` + `futures-util` (no runtime) for these combinators. They're pure logic, no executor dependency. |

The `futures-lite` crate is another minimal option — just combinators, no runtime. Using it doesn't compromise the "roll your own" goal.

---

## Summary Recommendation

1. **Executor**: Park/unpark based, single-threaded, `!Send` tasks. Use `futures::task::ArcWake` to avoid hand-rolling RawWaker.
2. **Channels**: Hand-roll async mpsc (~80 lines). Start unbounded. Oneshot for request-reply.
3. **State**: `Rc<RefCell<>>` split by domain. Enforce no-borrow-across-await via code convention and clippy.
4. **Workers**: Hybrid — dedicated threads for stateful subsystems, small thread pool for stateless work.
5. **Consider pulling in**: `futures-lite` or `futures-util` for combinators. `crossbeam-channel` if `std::sync::mpsc` performance isn't enough. These are libraries, not runtimes.
