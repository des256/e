use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonically increasing generation counter for invalidation tracking.
///
/// Epochs start at 1 and advance atomically. Useful for cache invalidation
/// and dirty-checking: store the epoch when a value was last computed and
/// compare against [`current`](Epoch::current) to detect staleness.
pub struct Epoch(AtomicU64);

impl Epoch {
    /// Create a new epoch counter starting at 1.
    pub fn new() -> Self {
        Epoch(AtomicU64::new(1))
    }

    /// Read the current epoch value.
    pub fn current(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Increment the epoch, returning the previous value.
    pub fn advance(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }

    /// Check whether `epoch` matches the current value.
    pub fn is_current(&self, epoch: u64) -> bool {
        epoch == self.0.load(Ordering::Relaxed)
    }
}
