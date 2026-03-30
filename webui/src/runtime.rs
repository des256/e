use std::{
    any::Any,
    cell::RefCell,
    collections::HashSet,
    marker::PhantomData,
    rc::Rc,
};

// -- internals --

struct Slot {
    value: Box<dyn Any>,
    subscribers: HashSet<usize>,
}

struct Effect {
    f: Rc<dyn Fn(&Context)>,
    deps: HashSet<usize>,
}

struct Runtime {
    slots: Vec<Slot>,
    effects: Vec<Effect>,
    tracking: Option<usize>,
    dirty: Vec<usize>,
    flushing: bool,
}

impl Runtime {
    fn new() -> Self {
        Runtime {
            slots: Vec::new(),
            effects: Vec::new(),
            tracking: None,
            dirty: Vec::new(),
            flushing: false,
        }
    }
}

thread_local! {
    static RUNTIME: RefCell<Runtime> = RefCell::new(Runtime::new());
}

// -- context --

/// Context handle for the reactive runtime.
///
/// All signal reads/writes and effect creation go through `Context`.
/// Obtained via [`with_context`] or from effect/event callbacks.
pub struct Context {
    _private: (),
}

/// Run a closure with access to the reactive context.
///
/// This is the main entry point for non-event code. Event callbacks
/// and effects receive a [`Context`] as a parameter.
///
/// # Examples
///
/// ```
/// use webui::runtime::*;
///
/// with_context(|context| {
///     let count = context.signal(0u32);
///     assert_eq!(count.get(context), 0);
/// });
/// ```
pub fn with_context(f: impl FnOnce(&Context)) {
    f(&Context { _private: () });
}

impl Context {
    /// Create a context handle (crate-internal).
    pub(crate) fn new() -> Self {
        Context { _private: () }
    }

    /// Create a new signal with an initial value.
    ///
    /// # Examples
    ///
    /// ```
    /// use webui::runtime::*;
    ///
    /// with_context(|context| {
    ///     let name = context.signal("Alice".to_string());
    ///     assert_eq!(name.get(context), "Alice");
    /// });
    /// ```
    pub fn signal<T: Clone + 'static>(&self, value: T) -> Signal<T> {
        RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            let id = rt.slots.len();
            rt.slots.push(Slot {
                value: Box::new(value),
                subscribers: HashSet::new(),
            });
            Signal { id, _marker: PhantomData }
        })
    }

    /// Register a reactive effect.
    ///
    /// The closure runs immediately to discover its signal dependencies,
    /// then re-runs whenever any of those signals change.
    ///
    /// # Examples
    ///
    /// ```
    /// use webui::runtime::*;
    /// use std::{cell::Cell, rc::Rc};
    ///
    /// with_context(|context| {
    ///     let a = context.signal(1);
    ///     let b = context.signal(2);
    ///     let sum = context.signal(0);
    ///     context.effect(move |context| {
    ///         sum.set(context, a.get(context) + b.get(context));
    ///     });
    ///     assert_eq!(sum.get(context), 3);
    ///     a.set(context, 10);
    ///     assert_eq!(sum.get(context), 12);
    /// });
    /// ```
    pub fn effect(&self, f: impl Fn(&Context) + 'static) {
        let effect_id = RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            let id = rt.effects.len();
            rt.effects.push(Effect {
                f: Rc::new(f),
                deps: HashSet::new(),
            });
            id
        });
        run_effect(effect_id);
    }
}

// -- signals --

/// A copyable handle to a reactive value in the signal arena.
///
/// `Signal<T>` is `Copy` — it is just an index. The runtime owns the
/// actual data. Read with [`get`](Signal::get), write with
/// [`set`](Signal::set).
///
/// # Examples
///
/// ```
/// use webui::runtime::*;
///
/// with_context(|context| {
///     let s = context.signal(42);
///     assert_eq!(s.get(context), 42);
///     s.set(context, 99);
///     assert_eq!(s.get(context), 99);
/// });
/// ```
pub struct Signal<T> {
    id: usize,
    _marker: PhantomData<T>,
}

impl<T> Clone for Signal<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> Copy for Signal<T> {}

impl<T: Clone + 'static> Signal<T> {
    /// Read the current value (cloned).
    ///
    /// If called inside an effect, registers this signal as a dependency
    /// so the effect re-runs when the signal changes.
    pub fn get(&self, _context: &Context) -> T {
        RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            if let Some(eid) = rt.tracking {
                rt.slots[self.id].subscribers.insert(eid);
                rt.effects[eid].deps.insert(self.id);
            }
            rt.slots[self.id]
                .value
                .downcast_ref::<T>()
                .expect("signal type mismatch")
                .clone()
        })
    }

    /// Write a new value, scheduling dependent effects for re-execution.
    pub fn set(&self, _context: &Context, value: T) {
        RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            rt.slots[self.id].value = Box::new(value);
            let dirty: Vec<usize> = rt.slots[self.id].subscribers.iter().copied().collect();
            for eid in dirty {
                if !rt.dirty.contains(&eid) {
                    rt.dirty.push(eid);
                }
            }
        });
        flush();
    }

    /// Update the value via a closure.
    pub fn update(&self, context: &Context, f: impl FnOnce(&T) -> T) {
        let new_val = f(&self.get(context));
        self.set(context, new_val);
    }
}

// -- flush --

/// Flush all dirty effects. Re-entrant calls are ignored; effects
/// queued during a flush are processed by the outer loop.
fn flush() {
    let already = RUNTIME.with(|rt| rt.borrow().flushing);
    if already {
        return;
    }
    RUNTIME.with(|rt| rt.borrow_mut().flushing = true);
    loop {
        let eid = RUNTIME.with(|rt| rt.borrow_mut().dirty.pop());
        match eid {
            Some(eid) => run_effect(eid),
            None => break,
        }
    }
    RUNTIME.with(|rt| rt.borrow_mut().flushing = false);
}

/// Run a single effect: clear old deps, track new deps during execution.
fn run_effect(effect_id: usize) {
    let (f, prev_tracking) = RUNTIME.with(|rt| {
        let mut rt = rt.borrow_mut();
        let prev = rt.tracking;
        let old_deps: Vec<usize> = rt.effects[effect_id].deps.drain().collect();
        for sig_id in old_deps {
            rt.slots[sig_id].subscribers.remove(&effect_id);
        }
        rt.tracking = Some(effect_id);
        (rt.effects[effect_id].f.clone(), prev)
    });
    // Runtime NOT borrowed here — closure may call get/set.
    f(&Context::new());
    RUNTIME.with(|rt| {
        rt.borrow_mut().tracking = prev_tracking;
    });
}

// -- tests --

#[cfg(test)]
fn reset_runtime() {
    RUNTIME.with(|rt| {
        *rt.borrow_mut() = Runtime::new();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn signal_get_set() {
        reset_runtime();
        with_context(|context| {
            let s = context.signal(42);
            assert_eq!(s.get(context), 42);
            s.set(context, 99);
            assert_eq!(s.get(context), 99);
        });
    }

    #[test]
    fn signal_update() {
        reset_runtime();
        with_context(|context| {
            let s = context.signal(10);
            s.update(context, |v| v + 5);
            assert_eq!(s.get(context), 15);
        });
    }

    #[test]
    fn effect_runs_immediately() {
        reset_runtime();
        let ran = Rc::new(Cell::new(false));
        let ran2 = ran.clone();
        with_context(|context| {
            context.effect(move |_context| {
                ran2.set(true);
            });
        });
        assert!(ran.get());
    }

    #[test]
    fn effect_tracks_dependency() {
        reset_runtime();
        let count = Rc::new(Cell::new(0u32));
        let count2 = count.clone();
        with_context(|context| {
            let s = context.signal(0);
            context.effect(move |context| {
                let _ = s.get(context);
                count2.set(count2.get() + 1);
            });
            assert_eq!(count.get(), 1);
            s.set(context, 1);
            assert_eq!(count.get(), 2);
            s.set(context, 2);
            assert_eq!(count.get(), 3);
        });
    }

    #[test]
    fn effect_only_tracks_read_signals() {
        reset_runtime();
        let count = Rc::new(Cell::new(0u32));
        let count2 = count.clone();
        with_context(|context| {
            let a = context.signal(0);
            let b = context.signal(0);
            context.effect(move |context| {
                let _ = a.get(context);
                count2.set(count2.get() + 1);
            });
            assert_eq!(count.get(), 1);
            a.set(context, 1);
            assert_eq!(count.get(), 2);
            // Changing `b` does NOT trigger the effect.
            b.set(context, 1);
            assert_eq!(count.get(), 2);
        });
    }

    #[test]
    fn derived_signal_pattern() {
        reset_runtime();
        with_context(|context| {
            let first = context.signal("Alice".to_string());
            let last = context.signal("Smith".to_string());
            let full = context.signal(String::new());
            context.effect(move |context| {
                let f = first.get(context);
                let l = last.get(context);
                full.set(context, format!("{f} {l}"));
            });
            assert_eq!(full.get(context), "Alice Smith");
            first.set(context, "Bob".to_string());
            assert_eq!(full.get(context), "Bob Smith");
        });
    }

    #[test]
    fn effect_re_tracks_on_branch_change() {
        reset_runtime();
        let count = Rc::new(Cell::new(0u32));
        let count2 = count.clone();
        with_context(|context| {
            let flag = context.signal(true);
            let a = context.signal(1);
            let b = context.signal(2);
            let out = context.signal(0);
            context.effect(move |context| {
                count2.set(count2.get() + 1);
                if flag.get(context) {
                    out.set(context, a.get(context));
                } else {
                    out.set(context, b.get(context));
                }
            });
            assert_eq!(count.get(), 1);
            assert_eq!(out.get(context), 1);
            a.set(context, 10);
            assert_eq!(out.get(context), 10);
            // Switch branch — now `b` is the dep.
            flag.set(context, false);
            assert_eq!(out.get(context), 2);
            // `a` should no longer trigger the effect.
            let c = count.get();
            a.set(context, 99);
            assert_eq!(count.get(), c);
        });
    }
}
