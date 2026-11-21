//! Where each information set's action slots sit in a flat table.
//!
//! A flat store keeps one slot per (info set, action), info sets in index
//! order.  Finding an info set's slots needs its first slot and its action
//! count — which the stores used to keep in two arrays with one entry per info
//! set: 5 bytes each, 3 GB at the 595 M info sets of the 200 bb blueprint, and
//! a random read into them on every node visit.
//!
//! Info sets are not laid out arbitrarily, though.  An [`IndexedGame`] numbers
//! them in blocks — the blueprint gives every betting-tree node one contiguous
//! block of card buckets, all with that node's action count — so the same
//! information is a short list of **runs** of equal-width info sets: a few
//! hundred thousand entries instead of hundreds of millions.  [`Layout`] is
//! that list.
//!
//! [`IndexedGame`]: super::IndexedGame

use serde::{Deserialize, Serialize};

/// One info set's slots in a flat table: `base..base + n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub base: usize,
    pub n: usize,
}

impl Span {
    #[inline]
    pub fn range(self) -> std::ops::Range<usize> {
        self.base..self.base + self.n
    }
}

/// The slot layout of a flat table, run-length encoded: consecutive info sets
/// with the same action count form one run.
///
/// Always in canonical form (no empty runs, neighbouring runs differ in
/// width), so two layouts are equal exactly when they place every info set in
/// the same slots — which is how a checkpoint or a policy file is checked
/// against the game it is loaded for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    /// First info-set index of each run (ascending, `starts[0] == 0`).
    starts: Vec<u64>,
    /// First slot of each run.
    slots: Vec<u64>,
    /// Action count of every info set in the run.
    widths: Vec<u8>,
    /// Number of info sets.
    capacity: u64,
    /// Number of slots: `Σ` action counts.
    total_slots: u64,
}

impl Layout {
    /// Build from `(info sets, action count)` runs in index order.  Empty runs
    /// are dropped and equal-width neighbours merged, so any way of cutting
    /// the same sequence into runs yields the same layout.
    pub fn from_runs(runs: impl IntoIterator<Item = (u64, usize)>) -> Self {
        let mut layout =
            Layout { starts: Vec::new(), slots: Vec::new(), widths: Vec::new(), capacity: 0, total_slots: 0 };
        for (count, width) in runs {
            if count == 0 {
                continue;
            }
            let width8 = u8::try_from(width).expect("action count fits u8");
            if layout.widths.last() != Some(&width8) {
                layout.starts.push(layout.capacity);
                layout.slots.push(layout.total_slots);
                layout.widths.push(width8);
            }
            layout.capacity += count;
            layout.total_slots = count
                .checked_mul(width as u64)
                .and_then(|s| layout.total_slots.checked_add(s))
                .expect("slot count fits u64");
        }
        layout
    }

    /// Build by asking for every info set's action count in turn.
    pub fn build(capacity: usize, actions: impl Fn(usize) -> usize) -> Self {
        Self::from_runs((0..capacity).map(|i| (1, actions(i))))
    }

    /// Number of info sets.
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Number of (info set, action) slots — the length of a flat array.
    pub fn total_slots(&self) -> usize {
        self.total_slots as usize
    }

    /// The widest info set's action count (`0` for an empty layout).
    pub fn max_width(&self) -> usize {
        self.widths.iter().copied().max().unwrap_or(0) as usize
    }

    /// Number of runs — what the layout itself costs to store.
    pub fn runs(&self) -> usize {
        self.widths.len()
    }

    /// The slots of info set `index`: a binary search over the runs.  The
    /// traversals avoid it where the game can say where its cursor's info set
    /// lives ([`IndexedGame::slot_span`](super::IndexedGame::slot_span)); this
    /// is for the paths that start from an index.
    ///
    /// Panics if `index` is out of range.
    #[inline]
    pub fn span(&self, index: usize) -> Span {
        let i = index as u64;
        assert!(i < self.capacity, "info set {index} outside a layout of {}", self.capacity);
        let run = self.starts.partition_point(|&s| s <= i) - 1;
        let n = self.widths[run] as usize;
        Span { base: (self.slots[run] + (i - self.starts[run]) * n as u64) as usize, n }
    }

    /// Every info set's span, in index order.
    pub fn spans(&self) -> impl Iterator<Item = Span> + '_ {
        (0..self.widths.len()).flat_map(move |run| {
            let end = self.starts.get(run + 1).copied().unwrap_or(self.capacity);
            let n = self.widths[run] as usize;
            let first = self.slots[run] as usize;
            (0..(end - self.starts[run]) as usize).map(move |k| Span { base: first + k * n, n })
        })
    }

    /// Check the invariants [`from_runs`](Self::from_runs) establishes — for a
    /// layout read from a file, where a damaged one would otherwise send
    /// lookups to the wrong slots.
    pub fn validate(&self) -> Result<(), String> {
        let runs = self.widths.len();
        if self.starts.len() != runs || self.slots.len() != runs {
            return Err("layout arrays differ in length".into());
        }
        let (mut index, mut slot) = (0u64, 0u64);
        for run in 0..runs {
            if self.starts[run] != index || self.slots[run] != slot {
                return Err(format!("layout run {run} does not continue the one before it"));
            }
            if run > 0 && self.widths[run] == self.widths[run - 1] {
                return Err(format!("layout run {run} repeats its neighbour's width"));
            }
            let end = self.starts.get(run + 1).copied().unwrap_or(self.capacity);
            if end <= index {
                return Err(format!("layout run {run} is empty"));
            }
            slot = (end - index)
                .checked_mul(self.widths[run] as u64)
                .and_then(|s| slot.checked_add(s))
                .ok_or("layout slot count overflows")?;
            index = end;
        }
        if index != self.capacity || slot != self.total_slots {
            return Err("layout totals disagree with its runs".into());
        }
        Ok(())
    }

    /// One action count per info set — the expanded form the stores used to
    /// keep, for the tests that write checkpoints in the old layouts.
    #[cfg(test)]
    pub(crate) fn expanded_widths(&self) -> Vec<u8> {
        self.spans().map(|s| s.n as u8).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_match_a_per_info_set_table() {
        // Widths that change, repeat, and include a long run: every lookup
        // must agree with the cumulative sum the old per-info-set arrays held.
        let widths = [2usize, 2, 2, 3, 3, 2, 5, 5, 5, 5, 1, 2, 2];
        let layout = Layout::build(widths.len(), |i| widths[i]);
        assert_eq!(layout.capacity(), widths.len());
        assert_eq!(layout.total_slots(), widths.iter().sum::<usize>());
        assert_eq!(layout.runs(), 6);
        assert_eq!(layout.max_width(), 5);
        let mut base = 0;
        for (i, &n) in widths.iter().enumerate() {
            assert_eq!(layout.span(i), Span { base, n }, "info set {i}");
            base += n;
        }
        assert_eq!(layout.spans().collect::<Vec<_>>(), (0..widths.len()).map(|i| layout.span(i)).collect::<Vec<_>>());
        assert_eq!(layout.expanded_widths(), widths.iter().map(|&w| w as u8).collect::<Vec<_>>());
        layout.validate().unwrap();
    }

    #[test]
    fn the_same_sequence_cut_differently_is_the_same_layout() {
        // A game may describe its blocks node by node; the table may build
        // from one info set at a time.  Equality must not depend on which.
        let by_block = Layout::from_runs([(3, 2), (0, 7), (2, 2), (4, 3), (1, 2)]);
        let one_by_one = Layout::build(10, |i| if (5..9).contains(&i) { 3 } else { 2 });
        assert_eq!(by_block, one_by_one);
        assert_eq!(by_block.runs(), 3);
        assert_ne!(by_block, Layout::build(10, |_| 2));
    }

    #[test]
    fn slot_counts_past_u32_are_exact() {
        // The old layout counted slots in a `u32`, which wraps silently in a
        // release build past 4.29 B slots — a 34 GB table at 8 bytes a slot.
        let layout = Layout::from_runs([(1_500_000_000, 3), (10, 2)]);
        assert_eq!(layout.total_slots() as u64, 4_500_000_020);
        assert_eq!(layout.span(1_500_000_000), Span { base: 4_500_000_000, n: 2 });
        assert_eq!(layout.span(1_499_999_999), Span { base: 4_499_999_997, n: 3 });
        layout.validate().unwrap();
    }

    #[test]
    fn an_empty_layout_is_valid_and_a_damaged_one_is_not() {
        let empty = Layout::from_runs([]);
        assert_eq!((empty.capacity(), empty.total_slots(), empty.max_width()), (0, 0, 0));
        empty.validate().unwrap();

        let good = Layout::from_runs([(4, 2), (4, 3)]);
        let mut shifted = good.clone();
        shifted.slots[1] += 1;
        assert!(shifted.validate().is_err(), "a run that starts at the wrong slot");
        let mut short = good.clone();
        short.total_slots -= 1;
        assert!(short.validate().is_err(), "totals that disagree with the runs");
        let mut ragged = good;
        ragged.widths.pop();
        assert!(ragged.validate().is_err(), "arrays of different length");
    }

    #[test]
    #[should_panic(expected = "outside a layout")]
    fn an_index_past_the_end_panics() {
        Layout::build(3, |_| 2).span(3);
    }
}
