//! Reactive core: signals, effects and the owner tree.
//!
//! Signals live in an arena; effects track the signals they read and
//! re-run when one changes. Effects also form an owner tree: an effect
//! created while another effect runs belongs to it, and so do the
//! signals and cleanups registered during that run. Re-running or
//! disposing an effect first disposes everything it created last time,
//! so nothing built inside a reactive scope can outlive it.

use std::{
    any::Any,
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    marker::PhantomData,
    rc::Rc,
};

// -- internals --

/// One signal's storage.
struct Slot {
    /// The value; `None` once the owning effect freed the slot.
    value: Option<Box<dyn Any>>,
    /// Effects that read this signal during their last run.
    subscribers: HashSet<usize>,
    /// Bumped when the slot is freed so stale [`Signal`] handles are rejected.
    gen: u32,
}

/// One effect: its body, what it read, and what it owns.
struct Effect {
    /// The body; `None` once disposed.
    f: Option<Rc<dyn Fn(&Context)>>,
    /// Signals read during the last run.
    deps: HashSet<usize>,
    /// Effects created during the last run.
    children: Vec<usize>,
    /// Cleanups registered during the last run.
    cleanups: Vec<Box<dyn FnOnce()>>,
    /// Signals created during the last run.
    owned_signals: Vec<usize>,
    /// Distance from the root scope; owners run before their descendants.
    depth: u32,
    /// Bumped on dispose so stale dirty entries are skipped.
    gen: u32,
    /// Whether a dirty entry for the current generation is queued.
    queued: bool,
}

struct Runtime {
    slots: Vec<Slot>,
    free_slots: Vec<usize>,
    effects: Vec<Effect>,
    free_effects: Vec<usize>,
    /// The running effect: it tracks reads and owns what gets created.
    tracking: Option<usize>,
    /// Dirty effects as `(depth, id, gen)`, lowest depth first.
    dirty: BinaryHeap<Reverse<(u32, usize, u32)>>,
    flushing: bool,
}

impl Runtime {
    fn new() -> Self {
        Runtime {
            slots: Vec::new(),
            free_slots: Vec::new(),
            effects: Vec::new(),
            free_effects: Vec::new(),
            tracking: None,
            dirty: BinaryHeap::new(),
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
/// and effects receive a [`Context`] as a parameter. Code run here is
/// in the root scope: effects and signals it creates live for the rest
/// of the program.
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
    /// A signal created inside an effect is owned by it and freed when
    /// the effect re-runs or is disposed; using the handle afterwards
    /// panics.
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
            let value: Box<dyn Any> = Box::new(value);
            let id = match rt.free_slots.pop() {
                Some(id) => {
                    rt.slots[id].value = Some(value);
                    id
                }
                None => {
                    rt.slots.push(Slot { value: Some(value), subscribers: HashSet::new(), gen: 0 });
                    rt.slots.len() - 1
                }
            };
            let gen = rt.slots[id].gen;
            if let Some(eid) = rt.tracking {
                rt.effects[eid].owned_signals.push(id);
            }
            Signal { id, gen, _marker: PhantomData }
        })
    }

    /// Register a reactive effect.
    ///
    /// The closure runs immediately to discover its signal dependencies,
    /// then re-runs whenever any of those signals change. An effect
    /// created while another effect runs is owned by it: before the
    /// owner re-runs, and when it is disposed, every owned effect,
    /// signal and cleanup from the previous run is disposed first.
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
        let id = RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            let owner = rt.tracking;
            let depth = owner.map_or(0, |o| rt.effects[o].depth + 1);
            let f: Rc<dyn Fn(&Context)> = Rc::new(f);
            let id = match rt.free_effects.pop() {
                Some(id) => {
                    let e = &mut rt.effects[id];
                    e.f = Some(f);
                    e.depth = depth;
                    id
                }
                None => {
                    rt.effects.push(Effect {
                        f: Some(f),
                        deps: HashSet::new(),
                        children: Vec::new(),
                        cleanups: Vec::new(),
                        owned_signals: Vec::new(),
                        depth,
                        gen: 0,
                        queued: false,
                    });
                    rt.effects.len() - 1
                }
            };
            if let Some(o) = owner {
                rt.effects[o].children.push(id);
            }
            id
        });
        run_effect(id);
    }

    /// Register a cleanup with the running effect.
    ///
    /// Cleanups run, latest first, before the effect re-runs and when it
    /// is disposed. Outside any effect (the root scope) the closure is
    /// dropped: root resources live for the rest of the program.
    ///
    /// # Examples
    ///
    /// ```
    /// use webui::runtime::*;
    /// use std::{cell::Cell, rc::Rc};
    ///
    /// let cleaned = Rc::new(Cell::new(0));
    /// let counter = cleaned.clone();
    /// with_context(|context| {
    ///     let trigger = context.signal(0);
    ///     context.effect(move |context| {
    ///         let _ = trigger.get(context);
    ///         let counter = counter.clone();
    ///         context.on_cleanup(move || counter.set(counter.get() + 1));
    ///     });
    ///     trigger.set(context, 1);
    /// });
    /// assert_eq!(cleaned.get(), 1);
    /// ```
    pub fn on_cleanup(&self, f: impl FnOnce() + 'static) {
        RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            if let Some(eid) = rt.tracking {
                rt.effects[eid].cleanups.push(Box::new(f));
            }
        });
    }
}

// -- signals --

/// A copyable handle to a reactive value in the signal arena.
///
/// `Signal<T>` is `Copy` — it is an index plus a generation. The runtime
/// owns the actual data. Read with [`get`](Signal::get), write with
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
    gen: u32,
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
    ///
    /// # Panics
    ///
    /// If the signal was freed because the effect that created it
    /// re-ran or was disposed.
    pub fn get(&self, _context: &Context) -> T {
        RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            let value = {
                let slot = &rt.slots[self.id];
                assert!(slot.gen == self.gen && slot.value.is_some(), "signal disposed");
                slot.value
                    .as_ref()
                    .unwrap()
                    .downcast_ref::<T>()
                    .expect("signal type mismatch")
                    .clone()
            };
            if let Some(eid) = rt.tracking {
                rt.slots[self.id].subscribers.insert(eid);
                rt.effects[eid].deps.insert(self.id);
            }
            value
        })
    }

    /// Write a new value, scheduling dependent effects for re-execution.
    ///
    /// # Panics
    ///
    /// If the signal was freed; see [`get`](Signal::get).
    pub fn set(&self, _context: &Context, value: T) {
        RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            let Runtime { slots, effects, dirty, .. } = &mut *rt;
            let slot = &mut slots[self.id];
            assert!(slot.gen == self.gen && slot.value.is_some(), "signal disposed");
            slot.value = Some(Box::new(value));
            for &eid in &slot.subscribers {
                let e = &mut effects[eid];
                if e.f.is_some() && !e.queued {
                    e.queued = true;
                    dirty.push(Reverse((e.depth, eid, e.gen)));
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

// -- stats --

/// Live object counts, for tests and diagnostics.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Stats {
    /// Effects that have not been disposed.
    pub live_effects: usize,
    /// Signals that have not been freed.
    pub live_signals: usize,
}

/// Count the live effects and signals.
pub fn stats() -> Stats {
    RUNTIME.with(|rt| {
        let rt = rt.borrow();
        Stats {
            live_effects: rt.effects.iter().filter(|e| e.f.is_some()).count(),
            live_signals: rt.slots.iter().filter(|s| s.value.is_some()).count(),
        }
    })
}

// -- flush --

/// Run dirty effects, owners before descendants, until none are left.
/// Re-entrant calls return at once; effects queued during a flush are
/// processed by the outer loop. Entries for disposed or re-created
/// effects are skipped.
fn flush() {
    let already = RUNTIME.with(|rt| rt.borrow().flushing);
    if already {
        return;
    }
    RUNTIME.with(|rt| rt.borrow_mut().flushing = true);
    loop {
        let next = RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            while let Some(Reverse((_, eid, gen))) = rt.dirty.pop() {
                let e = &mut rt.effects[eid];
                if e.gen != gen || e.f.is_none() {
                    continue;
                }
                e.queued = false;
                return Some(eid);
            }
            None
        });
        match next {
            Some(eid) => run_effect(eid),
            None => break,
        }
    }
    RUNTIME.with(|rt| rt.borrow_mut().flushing = false);
}

// -- effects --

/// Run one effect: tear down its previous run, then execute the body
/// with tracking and ownership pointed at it.
fn run_effect(effect_id: usize) {
    teardown(effect_id);
    let (f, prev_tracking) = RUNTIME.with(|rt| {
        let mut rt = rt.borrow_mut();
        let prev = rt.tracking;
        rt.tracking = Some(effect_id);
        (rt.effects[effect_id].f.clone(), prev)
    });
    // Runtime NOT borrowed here — the body may call get/set/effect.
    if let Some(f) = f {
        f(&Context::new());
    }
    RUNTIME.with(|rt| {
        rt.borrow_mut().tracking = prev_tracking;
    });
}

/// Undo everything an effect's last run produced: dispose its child
/// effects, run its cleanups latest first, free its signals and drop its
/// subscriptions. The effect itself stays alive.
fn teardown(effect_id: usize) {
    let (children, cleanups, owned, deps) = RUNTIME.with(|rt| {
        let mut rt = rt.borrow_mut();
        let e = &mut rt.effects[effect_id];
        (
            std::mem::take(&mut e.children),
            std::mem::take(&mut e.cleanups),
            std::mem::take(&mut e.owned_signals),
            e.deps.drain().collect::<Vec<_>>(),
        )
    });
    for child in children {
        dispose(child);
    }
    // Cleanups run with the runtime unborrowed and nothing tracking, so
    // they may call into the FFI or write signals without subscribing.
    let prev_tracking = RUNTIME.with(|rt| {
        let mut rt = rt.borrow_mut();
        let prev = rt.tracking;
        rt.tracking = None;
        prev
    });
    for cleanup in cleanups.into_iter().rev() {
        cleanup();
    }
    RUNTIME.with(|rt| {
        let mut rt = rt.borrow_mut();
        rt.tracking = prev_tracking;
        for sid in owned {
            free_slot(&mut rt, sid);
        }
        for sid in deps {
            rt.slots[sid].subscribers.remove(&effect_id);
        }
    });
}

/// Tear an effect down and retire it; its id goes on the free list.
fn dispose(effect_id: usize) {
    teardown(effect_id);
    RUNTIME.with(|rt| {
        let mut rt = rt.borrow_mut();
        let e = &mut rt.effects[effect_id];
        e.f = None;
        e.queued = false;
        e.gen = e.gen.wrapping_add(1);
        rt.free_effects.push(effect_id);
    });
}

/// Free a signal slot; stale handles see the generation bump.
fn free_slot(rt: &mut Runtime, sid: usize) {
    let slot = &mut rt.slots[sid];
    slot.value = None;
    slot.subscribers.clear();
    slot.gen = slot.gen.wrapping_add(1);
    rt.free_slots.push(sid);
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

    // -- ownership --

    #[test]
    fn child_of_rerun_effect_stops_firing() {
        reset_runtime();
        let child_runs = Rc::new(Cell::new(0u32));
        let child_runs2 = child_runs.clone();
        with_context(|context| {
            let outer = context.signal(0);
            let inner = context.signal(0);
            context.effect(move |context| {
                let _ = outer.get(context);
                let child_runs3 = child_runs2.clone();
                context.effect(move |context| {
                    let _ = inner.get(context);
                    child_runs3.set(child_runs3.get() + 1);
                });
            });
            assert_eq!(child_runs.get(), 1);
            // Re-run the owner: the old child must be disposed, the new one runs once.
            outer.set(context, 1);
            assert_eq!(child_runs.get(), 2);
            // Only the live child may react to `inner`.
            inner.set(context, 1);
            assert_eq!(child_runs.get(), 3, "disposed child effect still fires");
        });
    }

    #[test]
    fn rerun_disposes_child_effects_and_reuses_ids() {
        reset_runtime();
        with_context(|context| {
            let outer = context.signal(0);
            context.effect(move |context| {
                let _ = outer.get(context);
                context.effect(|_| {});
                context.effect(|_| {});
            });
            for i in 1..=100 {
                outer.set(context, i);
            }
            assert_eq!(stats().live_effects, 3);
            let arena = RUNTIME.with(|rt| rt.borrow().effects.len());
            assert_eq!(arena, 3, "disposed effect ids are not reused");
        });
    }

    #[test]
    fn cleanups_run_latest_first_exactly_once() {
        reset_runtime();
        let log = Rc::new(RefCell::new(Vec::new()));
        let log2 = log.clone();
        with_context(|context| {
            let trigger = context.signal(0);
            context.effect(move |context| {
                let _ = trigger.get(context);
                let (a, b) = (log2.clone(), log2.clone());
                context.on_cleanup(move || a.borrow_mut().push("first"));
                context.on_cleanup(move || b.borrow_mut().push("second"));
            });
            assert!(log.borrow().is_empty());
            trigger.set(context, 1);
            assert_eq!(*log.borrow(), vec!["second", "first"]);
            trigger.set(context, 2);
            assert_eq!(*log.borrow(), vec!["second", "first", "second", "first"]);
        });
    }

    #[test]
    fn nested_dispose_reaches_grandchild() {
        reset_runtime();
        let grandchild_runs = Rc::new(Cell::new(0u32));
        let cleaned = Rc::new(Cell::new(0u32));
        let (gr, cl) = (grandchild_runs.clone(), cleaned.clone());
        with_context(|context| {
            let top = context.signal(0);
            let leaf = context.signal(0);
            context.effect(move |context| {
                let _ = top.get(context);
                let (gr, cl) = (gr.clone(), cl.clone());
                context.effect(move |context| {
                    let (gr, cl) = (gr.clone(), cl.clone());
                    context.effect(move |context| {
                        let _ = leaf.get(context);
                        gr.set(gr.get() + 1);
                        let cl = cl.clone();
                        context.on_cleanup(move || cl.set(cl.get() + 1));
                    });
                });
            });
            assert_eq!(stats().live_effects, 3);
            top.set(context, 1);
            assert_eq!(cleaned.get(), 1, "grandchild cleanup did not run");
            assert_eq!(stats().live_effects, 3);
            leaf.set(context, 1);
            assert_eq!(grandchild_runs.get(), 3, "old grandchild still subscribed");
        });
    }

    #[test]
    fn owner_runs_before_dirty_child_and_stale_entry_is_skipped() {
        reset_runtime();
        let child_runs = Rc::new(Cell::new(0u32));
        let owner_runs = Rc::new(Cell::new(0u32));
        let (cr, or) = (child_runs.clone(), owner_runs.clone());
        with_context(|context| {
            let s = context.signal(0);
            context.effect(move |context| {
                let _ = s.get(context);
                or.set(or.get() + 1);
                let cr = cr.clone();
                context.effect(move |context| {
                    let _ = s.get(context);
                    cr.set(cr.get() + 1);
                });
            });
            assert_eq!((owner_runs.get(), child_runs.get()), (1, 1));
            // Both are dirty. The owner runs first and disposes the child,
            // whose id the new child reuses; the child's stale dirty entry
            // must not run the new child a second time.
            s.set(context, 1);
            assert_eq!(owner_runs.get(), 2);
            assert_eq!(child_runs.get(), 2, "child ran on stale state or twice");
        });
    }

    #[test]
    fn owned_signal_is_freed_on_rerun() {
        reset_runtime();
        with_context(|context| {
            let trigger = context.signal(0);
            context.effect(move |context| {
                let _ = trigger.get(context);
                let _local = context.signal(String::from("scratch"));
            });
            assert_eq!(stats().live_signals, 2);
            for i in 1..=10 {
                trigger.set(context, i);
            }
            assert_eq!(stats().live_signals, 2);
            let arena = RUNTIME.with(|rt| rt.borrow().slots.len());
            assert_eq!(arena, 2, "freed signal slots are not reused");
        });
    }

    #[test]
    #[should_panic(expected = "signal disposed")]
    fn stale_signal_handle_panics() {
        reset_runtime();
        with_context(|context| {
            let trigger = context.signal(0);
            let leaked = Rc::new(Cell::new(None));
            let leaked2 = leaked.clone();
            context.effect(move |context| {
                let _ = trigger.get(context);
                leaked2.set(Some(context.signal(1u8)));
            });
            let stale = leaked.get().unwrap();
            trigger.set(context, 1);
            let _ = stale.get(context);
        });
    }

    #[test]
    fn set_inside_cleanup_during_flush_is_processed() {
        reset_runtime();
        let seen = Rc::new(Cell::new(0u32));
        let seen2 = seen.clone();
        with_context(|context| {
            let trigger = context.signal(0);
            let note = context.signal(0u32);
            context.effect(move |context| {
                seen2.set(note.get(context));
            });
            context.effect(move |context| {
                let _ = trigger.get(context);
                context.on_cleanup(move || with_context(|cx| note.set(cx, 7)));
            });
            trigger.set(context, 1);
            assert_eq!(seen.get(), 7);
        });
    }

    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn set_cost_with_many_subscribers() {
        reset_runtime();
        with_context(|context| {
            let s = context.signal(0usize);
            for _ in 0..40_000 {
                context.effect(move |context| {
                    let _ = s.get(context);
                });
            }
            let t = std::time::Instant::now();
            s.set(context, 1);
            println!("40000 subscribers: set() took {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
        });
    }
}
