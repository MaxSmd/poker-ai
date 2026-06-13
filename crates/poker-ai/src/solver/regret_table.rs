//! Structure-of-Arrays regret storage for the blueprint solver.
//!
//! The `HashMap<u64, Node>` store carries five heap `Vec`s of `f64`/`u32` per
//! information set — ~10× the plan's budget of 3 `f32` accumulators per
//! (info set, action).  When the info-set space is known up front (an
//! [`IndexedGame`](crate::games::IndexedGame): street × bucket × betting
//! sequence → a computed dense index), regrets can live in **flat `f32` arrays**
//! addressed by that index: no hashing, no per-node allocation, and a checkpoint
//! that is a handful of contiguous arrays rather than a serde-heavy map.
//!
//! Arithmetic is done in `f64` and stored as `f32` (standard for production CFR;
//! the strategy is a ratio of regrets, robust to the reduced mantissa).  The
//! optimistic (`prev_inst`) and pruning (`consec_below`) accumulators are
//! **optional** arrays — empty unless those features are enabled — so you only
//! pay their memory when you use them.
//!
//! A `bf16`-stored variant (half the RAM) was prototyped and rejected: stochastic
//! rounding's per-store variance random-walks through the read-modify-write
//! accumulators and, with DCFR's *bounded* regrets, swamps the signal — push/fold
//! convergence degraded ~4× vs `f32`.  See the implementation-progress note.

use std::io;
use std::ops::Range;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Flat `f32` regret / strategy-sum / baseline arrays indexed by a dense info-set
/// id, with per-info-set offsets and action counts.
#[derive(Serialize, Deserialize)]
pub struct RegretTable {
    /// Cumulative counterfactual regret per (info set, action).
    regret: Vec<f32>,
    /// Average-strategy numerator per (info set, action).
    strategy_sum: Vec<f32>,
    /// VR-MCCFR baseline value per (info set, action), player-0 perspective.
    baseline: Vec<f32>,
    /// Previous instantaneous regret (optimistic updates) — empty unless enabled.
    prev_inst: Vec<f32>,
    /// Consecutive-below-θ streak (RBP pruning) — empty unless enabled.
    consec_below: Vec<u32>,
    /// Actions per info set.
    num_actions: Vec<u8>,
    /// Start offset of each info set in the flat arrays.
    offsets: Vec<u32>,
}

impl RegretTable {
    /// Build the table for `capacity` info sets, `actions(i)` actions each.
    /// `optimistic` / `pruning` allocate their feature arrays only when set.
    pub fn with_layout(
        capacity: usize,
        actions: impl Fn(usize) -> usize,
        optimistic: bool,
        pruning: bool,
    ) -> Self {
        let mut num_actions = Vec::with_capacity(capacity);
        let mut offsets = Vec::with_capacity(capacity);
        let mut total = 0u32;
        for i in 0..capacity {
            let a = actions(i);
            debug_assert!(a <= u8::MAX as usize, "action count fits u8");
            offsets.push(total);
            num_actions.push(a as u8);
            total += a as u32;
        }
        let total = total as usize;
        Self {
            regret: vec![0.0; total],
            strategy_sum: vec![0.0; total],
            baseline: vec![0.0; total],
            prev_inst: if optimistic { vec![0.0; total] } else { Vec::new() },
            consec_below: if pruning { vec![0; total] } else { Vec::new() },
            num_actions,
            offsets,
        }
    }

    /// Number of info sets.
    pub fn capacity(&self) -> usize {
        self.num_actions.len()
    }

    /// Total (info set × action) slots — the flat array length.
    pub fn total_slots(&self) -> usize {
        self.regret.len()
    }

    /// Per-info-set memory footprint in bytes (the accumulators actually held).
    pub fn bytes_per_info_set(&self) -> usize {
        let arrays = 3 + usize::from(!self.prev_inst.is_empty()); // regret+strat+baseline (+prev_inst)
        let avg_actions = if self.capacity() == 0 { 0 } else { self.total_slots() / self.capacity() };
        arrays * 4 * avg_actions + if self.consec_below.is_empty() { 0 } else { 4 * avg_actions }
    }

    fn span(&self, info_set: usize) -> Range<usize> {
        let start = self.offsets[info_set] as usize;
        start..start + self.num_actions[info_set] as usize
    }

    pub fn num_actions(&self, info_set: usize) -> usize {
        self.num_actions[info_set] as usize
    }

    pub fn regret_mut(&mut self, info_set: usize) -> &mut [f32] {
        let span = self.span(info_set);
        &mut self.regret[span]
    }

    pub fn strategy_sum_mut(&mut self, info_set: usize) -> &mut [f32] {
        let span = self.span(info_set);
        &mut self.strategy_sum[span]
    }

    pub fn baseline(&self, info_set: usize) -> &[f32] {
        &self.baseline[self.span(info_set)]
    }

    pub fn baseline_mut(&mut self, info_set: usize) -> &mut [f32] {
        let span = self.span(info_set);
        &mut self.baseline[span]
    }

    /// Optimistic momentum accumulator (only present when enabled).
    pub fn prev_inst_mut(&mut self, info_set: usize) -> Option<&mut [f32]> {
        if self.prev_inst.is_empty() {
            return None;
        }
        let span = self.span(info_set);
        Some(&mut self.prev_inst[span])
    }

    /// RBP below-θ streak accumulator (only present when enabled).
    pub fn consec_below_mut(&mut self, info_set: usize) -> Option<&mut [u32]> {
        if self.consec_below.is_empty() {
            return None;
        }
        let span = self.span(info_set);
        Some(&mut self.consec_below[span])
    }

    /// Regret-matched current strategy for `info_set`, written into `out` (`f64`
    /// arithmetic over the `f32` regrets).
    pub fn strategy_into(&self, info_set: usize, out: &mut Vec<f64>) {
        let regret = &self.regret[self.span(info_set)];
        let n = regret.len();
        out.clear();
        let total: f64 = regret.iter().map(|&r| (r as f64).max(0.0)).sum();
        if total > 0.0 {
            out.extend(regret.iter().map(|&r| (r as f64).max(0.0) / total));
        } else {
            out.extend(std::iter::repeat(1.0 / n as f64).take(n));
        }
    }

    /// Average (deployable) strategy for `info_set`, written into `out`.
    pub fn average_into(&self, info_set: usize, out: &mut Vec<f64>) {
        let s = &self.strategy_sum[self.span(info_set)];
        let n = s.len();
        out.clear();
        let total: f64 = s.iter().map(|&x| x as f64).sum();
        if total > 0.0 {
            out.extend(s.iter().map(|&x| x as f64 / total));
        } else {
            out.extend(std::iter::repeat(1.0 / n as f64).take(n));
        }
    }

    /// Serialize to a bincode file (flat arrays — compact, fast to load).
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = bincode::serialize(self).map_err(io::Error::other)?;
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
    }

    /// Load from a bincode file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_offsets_and_widths() {
        let t = RegretTable::with_layout(4, |_| 2, false, false);
        assert_eq!(t.capacity(), 4);
        assert_eq!(t.total_slots(), 8);
        assert_eq!(t.num_actions(2), 2);
        assert!(t.prev_inst.is_empty() && t.consec_below.is_empty());
    }

    #[test]
    fn uniform_strategy_when_no_regret() {
        let t = RegretTable::with_layout(1, |_| 4, false, false);
        let mut s = Vec::new();
        t.strategy_into(0, &mut s);
        assert_eq!(s, vec![0.25; 4]);
    }

    #[test]
    fn regret_matching_normalizes_positive_regret() {
        let mut t = RegretTable::with_layout(1, |_| 3, false, false);
        t.regret_mut(0).copy_from_slice(&[3.0, 0.0, -2.0]);
        let mut s = Vec::new();
        t.strategy_into(0, &mut s);
        assert!((s[0] - 1.0).abs() < 1e-9 && s[1] == 0.0 && s[2] == 0.0);
    }

    #[test]
    fn optional_arrays_allocate_only_when_enabled() {
        let plain = RegretTable::with_layout(10, |_| 2, false, false);
        let full = RegretTable::with_layout(10, |_| 2, true, true);
        assert!(plain.prev_inst.is_empty());
        assert!(!full.prev_inst.is_empty() && !full.consec_below.is_empty());
        // 3 f32 accumulators × 2 actions = 24 bytes/info set, vs the HashMap
        // Node's five heap vecs (~350 B).
        assert_eq!(plain.bytes_per_info_set(), 24);
    }

    #[test]
    fn save_load_round_trips() {
        let mut t = RegretTable::with_layout(2, |_| 2, false, false);
        t.regret_mut(1).copy_from_slice(&[1.5, -0.5]);
        t.strategy_sum_mut(1).copy_from_slice(&[2.0, 1.0]);
        let path = std::env::temp_dir().join(format!("regret_table_test_{}.bin", std::process::id()));
        t.save(&path).unwrap();
        let loaded = RegretTable::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let mut a = Vec::new();
        let mut b = Vec::new();
        t.average_into(1, &mut a);
        loaded.average_into(1, &mut b);
        assert_eq!(a, b);
    }
}
