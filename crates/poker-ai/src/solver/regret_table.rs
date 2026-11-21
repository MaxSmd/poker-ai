//! Structure-of-Arrays regret storage for the blueprint solver.
//!
//! The `HashMap<u64, Node>` store carries five heap `Vec`s of `f64`/`u32` per
//! information set — ~10× the budget of 3 `f32` accumulators per
//! (info set, action).  When the info-set space is known up front (an
//! [`IndexedGame`](crate::games::IndexedGame): street × bucket × betting
//! sequence → a computed dense index), regrets can live in **flat `f32` arrays**
//! addressed by that index: no hashing, no per-node allocation, and a checkpoint
//! that is a handful of contiguous arrays rather than a serde-heavy map.
//!
//! Arithmetic is done in `f64`; the accumulators are stored `f32`: regret and
//! strategy sum always (8 bytes per (info set, action)), plus the VR-MCCFR
//! baseline when control variates are on (12).  The baseline is an
//! **optional** array — empty unless enabled — so a run without control
//! variates does not pay a third of the table for it.  Nothing is stored per
//! info set: where each one's slots are is a run-length
//! [`Layout`], a few megabytes at any scale.
//!
//! ## Why the strategy sum can be `f32` (and why it could not before)
//!
//! The strategy sum was `f64` — half the whole table — because `f32` *did*
//! break it: iteration `t` contributes weight `t^γ`, so after `n` visits to an
//! info set the increment is about `1/n` of the running sum, and once
//! `n > 2^24 ≈ 1.7e7` that increment falls below half an ulp and rounds to
//! nothing.  The deployed average then **silently freezes**: it stops tracking
//! the strategy while still looking like a converged artifact.  On the 2e9-
//! iteration blueprint the preflop info sets are visited every iteration, so
//! they would freeze after well under 1 % of training — the info sets that
//! matter most, failing first and invisibly.
//!
//! That is a property of round-to-nearest, not of `f32`.  Under **stochastic
//! rounding** a sub-ulp increment is carried up with probability equal to its
//! share of the gap, so it accumulates *in expectation* instead of vanishing.
//! The residual is a random walk, and the argument that it is harmless here is
//! the same one [`LeanTable`](crate::solver::lean_table::LeanTable) already
//! makes for its `u16` sums: this array is **write-only and monotone growing**,
//! never read back into the update, so rounding noise cannot compound the way
//! it does through a read-modify-write accumulator.  That is exactly the
//! distinction that sank the `bf16` experiment below — which quantized
//! *regret*, the array that does feed back.
//!
//! Concretely, at the end of a 2e9-iteration run a hot info set holds a sum of
//! ~3e27 with ulp ~2e20 and increments of ~4e18, so each add carries the ulp
//! with p ≈ 0.017; over the run the accumulated standard deviation is ~0.04 %
//! of the sum, against an RTN error of 100 %.  Gated by
//! `strategy_sum_does_not_freeze_past_the_f32_integer_limit` and
//! `late_strategy_shift_still_moves_the_average` below, plus the push/fold
//! exploitability comparison in `games::push_fold`.
//!
//! A `bf16`-stored variant (half the RAM again) was prototyped and rejected:
//! stochastic rounding's per-store variance random-walks through the
//! read-modify-write **regret** accumulator and, with DCFR's *bounded* regrets,
//! swamps the signal — push/fold convergence degraded ~4× vs `f32`.  See the
//! implementation-progress note.

use std::io;

use serde::{Deserialize, Serialize};

use super::variant::Variant;
use super::MAX_ACTIONS;
use crate::games::{Layout, Span};
use crate::util::rng::xorshift_next_unit;

/// EMA learning rate for the VR-MCCFR baseline (mirrors `mccfr::BASELINE_RATE`).
const BASELINE_RATE: f64 = 0.1;

/// `sum + inc`, rounded to `f32` **stochastically**: the exact result is carried
/// to whichever of its two neighbouring `f32` values with probability equal to
/// its share of the gap, using the uniform draw `u ∈ [0, 1)`.
///
/// Unbiased by construction — `E[result] = sum + inc` — which is what lets a
/// sub-ulp increment accumulate instead of rounding to nothing forever (see the
/// module header).  Split from [`sr_add`] so the atomic trainer can draw its
/// random number *outside* the compare-and-swap retry loop.
#[inline]
pub(crate) fn sr_add_with(sum: f32, inc: f64, u: f64) -> f32 {
    let exact = sum as f64 + inc;
    let near = exact as f32;
    let err = exact - near as f64;
    if err == 0.0 || !near.is_finite() {
        return near;
    }
    let other = if err > 0.0 { near.next_up() } else { near.next_down() };
    let step = other as f64 - near as f64;
    if !step.is_finite() || step == 0.0 {
        return near;
    }
    // `step` carries the sign of `err`, so this share is in (0, 1].
    if u < err / step {
        other
    } else {
        near
    }
}

/// [`sr_add_with`] drawing its own random number — the serial form.
#[inline]
pub(crate) fn sr_add(sum: &mut f32, inc: f64, rng: &mut u64) {
    *sum = sr_add_with(*sum, inc, xorshift_next_unit(rng));
}

/// The running-discount factor for iteration `t`'s strategy-sum update:
/// `(t/(t+1))^γ` (1 for vanilla).  Used by the quantized store, whose 16-bit
/// sums cannot hold absolute `t^γ` weights at all; the f32 store keeps the
/// absolute weights and relies on stochastic rounding for the dynamic range
/// (see the module header).
pub(crate) fn strategy_discount(t: u64, variant: Variant) -> f64 {
    match variant {
        Variant::Vanilla => 1.0,
        Variant::Dcfr(d) => (t as f64 / (t as f64 + 1.0)).powf(d.gamma),
    }
}

/// Storage backend for the serial SoA blueprint solver
/// ([`SoaMccfr`](crate::solver::mccfr::SoaMccfr)): the flat per-(info set,
/// action) accumulators *plus the update arithmetic*, which is
/// precision-specific (the f32 store adds directly; the quantized
/// [`LeanTable`](crate::solver::lean_table::LeanTable) rounds, saturates, and
/// rescales).  The parallel and atomic training paths are f32-only and keep
/// using [`RegretTable`]'s concrete accessors.
///
/// Every per-info-set method takes the info set's [`Span`] rather than its
/// index.  The traversal gets the span from the game where the game can say
/// ([`IndexedGame::slot_span`](crate::games::IndexedGame::slot_span)) and from
/// [`span`](RegretStore::span) otherwise — once per node, however many arrays
/// it then touches.
pub trait RegretStore: Serialize + Sized {
    /// Allocate zeroed accumulators for `layout`.
    fn build(layout: Layout) -> Self;
    /// Where each info set's slots are.
    fn layout(&self) -> &Layout;
    fn capacity(&self) -> usize {
        self.layout().capacity()
    }
    /// The slots of info set `info_set`, looked up in the layout.
    fn span(&self, info_set: usize) -> Span {
        self.layout().span(info_set)
    }
    /// Ever updated (any strategy mass) — gates export of unreached slots.
    fn is_visited(&self, at: Span) -> bool;
    /// Allocate the VR-MCCFR baseline accumulator.  Called by
    /// [`SoaMccfr::with_baseline`](crate::solver::mccfr::SoaMccfr::with_baseline)
    /// — the flag is a builder step, so the table cannot know at `build` time
    /// whether control variates will be used, and a run without them should
    /// not carry the array.
    fn enable_baseline(&mut self);
    fn bytes_per_info_set(&self) -> usize;
    /// Regret-matching current strategy, written into the first `n` slots of
    /// `out` and `n` returned.  The buffer is a fixed
    /// `[f64; MAX_ACTIONS]` rather than a `Vec` so the traversals can keep it on
    /// the stack: this is the per-decision-node call, and a `Vec` here is one
    /// heap allocation per visited node (~10¹² of them over a blueprint run).
    fn strategy_into(&self, at: Span, out: &mut [f64; MAX_ACTIONS]) -> usize;
    /// Average (deployable) strategy, written into `out`.
    fn average_into(&self, at: Span, out: &mut Vec<f64>);
    /// The traverser's regret update at iteration `t`: per-action lazy discount
    /// (per `variant`) then add of the instantaneous regret `util[a] − node_value`.
    fn add_regret(&mut self, at: Span, util: &[f64], node_value: f64, t: u64, variant: Variant);
    /// The opponent's average-strategy accumulation at iteration `t`.  `rng`
    /// drives the stochastic rounding both stores use for this array — draw it
    /// from a stream separate from action sampling, so changing the storage
    /// precision cannot perturb which trajectories a seed explores.
    fn add_strategy(&mut self, at: Span, strategy: &[f64], t: u64, variant: Variant, rng: &mut u64);
    /// `(Σ_a σ(a)·b(a), b(chosen))` — the opponent-node control-variate reads.
    fn baseline_pair(&self, at: Span, strategy: &[f64], chosen: usize) -> (f64, f64);
    /// One EMA step of baseline slot `a` toward `target` (player-0 perspective).
    fn baseline_ema(&mut self, at: Span, a: usize, target: f64);
}

/// Flat `f32` regret / strategy-sum / baseline arrays, one slot per
/// (info set, action), placed by a [`Layout`].
#[derive(Serialize, Deserialize)]
pub struct RegretTable {
    /// Cumulative counterfactual regret per (info set, action).
    regret: Vec<f32>,
    /// Average-strategy numerator per (info set, action).  Accumulated with
    /// **stochastic rounding** ([`sr_add`]) — round-to-nearest freezes this
    /// array after ~1.7e7 visits, which is why it used to be `f64` at double
    /// the bytes.  See the module header for the full argument.
    strategy_sum: Vec<f32>,
    /// VR-MCCFR baseline value per (info set, action), player-0 perspective —
    /// **empty unless [`enable_baseline`](RegretStore::enable_baseline) was
    /// called**, so a run without control variates does not carry a third
    /// array the size of the regrets.
    baseline: Vec<f32>,
    /// Where each info set's slots are.  Run-length encoded — a few megabytes
    /// where the per-info-set offset and action-count arrays it replaced were
    /// 5 bytes per info set (see [`Layout`]).
    layout: Layout,
}

impl RegretTable {
    /// A zeroed table for `layout`, without the baseline array (see
    /// [`enable_baseline`](RegretStore::enable_baseline)).
    pub fn new(layout: Layout) -> Self {
        let total = layout.total_slots();
        Self { regret: vec![0.0; total], strategy_sum: vec![0.0; total], baseline: Vec::new(), layout }
    }

    /// A zeroed table for `capacity` info sets, `actions(i)` actions each.
    pub fn with_layout(capacity: usize, actions: impl Fn(usize) -> usize) -> Self {
        Self::new(Layout::build(capacity, actions))
    }

    /// Reassemble a table from arrays read from a file — the checkpoint
    /// loaders, current and legacy.  Rejects arrays that do not fit the
    /// layout: every later access trusts the layout for its bounds.
    pub(crate) fn from_parts(
        regret: Vec<f32>,
        strategy_sum: Vec<f32>,
        baseline: Vec<f32>,
        layout: Layout,
    ) -> io::Result<Self> {
        let table = Self { regret, strategy_sum, baseline, layout };
        table.validate().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(table)
    }

    /// The layout is sound and every array is exactly as long as it says.
    pub(crate) fn validate(&self) -> Result<(), String> {
        self.layout.validate()?;
        let total = self.layout.total_slots();
        if self.regret.len() != total || self.strategy_sum.len() != total {
            return Err(format!(
                "table arrays hold {} / {} slots but the layout has {total}",
                self.regret.len(),
                self.strategy_sum.len()
            ));
        }
        if !self.baseline.is_empty() && self.baseline.len() != total {
            return Err(format!("baseline holds {} slots but the layout has {total}", self.baseline.len()));
        }
        if self.layout.max_width() > MAX_ACTIONS {
            return Err(format!("an info set is wider than {MAX_ACTIONS} actions"));
        }
        Ok(())
    }

    /// The raw arrays, for the checkpoint-migration tests that have to write
    /// files in the older layouts.
    #[cfg(test)]
    pub(crate) fn parts_for_test(&self) -> (&[f32], &[f32], &[f32], &Layout) {
        (&self.regret, &self.strategy_sum, &self.baseline, &self.layout)
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Number of info sets.
    pub fn capacity(&self) -> usize {
        self.layout.capacity()
    }

    /// Total (info set × action) slots — the flat array length.
    pub fn total_slots(&self) -> usize {
        self.regret.len()
    }

    /// Per-info-set memory footprint in bytes (the accumulators actually held).
    pub fn bytes_per_info_set(&self) -> usize {
        // regret + strategy-sum, both f32, plus the baseline when resident.
        let per_slot = 4 + 4 + if self.baseline.is_empty() { 0 } else { 4 };
        let avg_actions = if self.capacity() == 0 { 0 } else { self.total_slots() / self.capacity() };
        per_slot * avg_actions
    }

    /// The slots of info set `info_set`, looked up in the layout.
    pub fn span(&self, info_set: usize) -> Span {
        self.layout.span(info_set)
    }

    /// Whether the info set at `at` was ever updated (any non-zero
    /// strategy-sum mass) — used to skip never-reached slots when exporting a
    /// deployable strategy.
    pub fn is_visited(&self, at: Span) -> bool {
        self.strategy_sum[at.range()].iter().any(|&x| x != 0.0)
    }

    pub fn regret_mut(&mut self, at: Span) -> &mut [f32] {
        &mut self.regret[at.range()]
    }

    pub fn strategy_sum_mut(&mut self, at: Span) -> &mut [f32] {
        &mut self.strategy_sum[at.range()]
    }

    /// Whether the baseline array is resident (control variates enabled).
    pub fn has_baseline(&self) -> bool {
        !self.baseline.is_empty()
    }

    pub fn baseline(&self, at: Span) -> &[f32] {
        debug_assert!(self.has_baseline(), "baseline read before `enable_baseline`");
        &self.baseline[at.range()]
    }

    pub fn baseline_mut(&mut self, at: Span) -> &mut [f32] {
        &mut self.baseline[at.range()]
    }

    /// Raw pointers to the three accumulator arrays plus their length, for the
    /// lock-free atomic trainer (`SoaMccfr::train_atomic`), which must update
    /// the table concurrently through shared references.  The baseline
    /// pointer is dangling when [`has_baseline`](Self::has_baseline) is false
    /// and must not be dereferenced then.  Soundness
    /// contract: the caller keeps the exclusive borrow of `self` alive for the
    /// whole training run, performs **every** access to these arrays through
    /// atomics, so no non-atomic aliases exist while threads race, and stays
    /// below the returned slot count.
    pub(crate) fn atomic_parts(&mut self) -> (*mut f32, *mut f32, *mut f32, usize) {
        (self.regret.as_mut_ptr(), self.strategy_sum.as_mut_ptr(), self.baseline.as_mut_ptr(), self.regret.len())
    }

    /// Regret-matched current strategy for the info set at `at`, written into
    /// `out[..n]` and `n` returned (`f64` arithmetic over the `f32` regrets).
    pub fn strategy_into(&self, at: Span, out: &mut [f64; MAX_ACTIONS]) -> usize {
        let regret = &self.regret[at.range()];
        let n = regret.len();
        assert!(n <= MAX_ACTIONS, "info set wider than the stack buffer");
        let total: f64 = regret.iter().map(|&r| (r as f64).max(0.0)).sum();
        if total > 0.0 {
            for (o, &r) in out.iter_mut().zip(regret) {
                *o = (r as f64).max(0.0) / total;
            }
        } else {
            out[..n].fill(1.0 / n as f64);
        }
        n
    }

    /// Average (deployable) strategy for the info set at `at`, written into `out`.
    pub fn average_into(&self, at: Span, out: &mut Vec<f64>) {
        let s = &self.strategy_sum[at.range()];
        let n = s.len();
        out.clear();
        let total: f64 = s.iter().map(|&x| x as f64).sum();
        if total > 0.0 {
            out.extend(s.iter().map(|&x| x as f64 / total));
        } else {
            out.extend(std::iter::repeat_n(1.0 / n as f64, n));
        }
    }
}

// The f32 store's update arithmetic: f64 math, f32 store — moved here verbatim
// from the solver's former inline updates (the bit-identical solver gates cover
// the move).
impl RegretStore for RegretTable {
    fn build(layout: Layout) -> Self {
        Self::new(layout)
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn is_visited(&self, at: Span) -> bool {
        RegretTable::is_visited(self, at)
    }

    fn enable_baseline(&mut self) {
        if self.baseline.is_empty() {
            self.baseline = vec![0.0; self.regret.len()];
        }
    }

    fn bytes_per_info_set(&self) -> usize {
        RegretTable::bytes_per_info_set(self)
    }

    fn strategy_into(&self, at: Span, out: &mut [f64; MAX_ACTIONS]) -> usize {
        RegretTable::strategy_into(self, at, out)
    }

    fn average_into(&self, at: Span, out: &mut Vec<f64>) {
        RegretTable::average_into(self, at, out)
    }

    fn add_regret(&mut self, at: Span, util: &[f64], node_value: f64, t: u64, variant: Variant) {
        let (pos, neg) = match variant {
            Variant::Vanilla => (1.0, 1.0),
            Variant::Dcfr(d) => (d.positive_factor(t), d.negative_factor(t)),
        };
        let discount = matches!(variant, Variant::Dcfr(_));
        let regret = self.regret_mut(at);
        for (r32, &u) in regret.iter_mut().zip(util) {
            let mut r = *r32 as f64;
            if discount {
                r *= if r > 0.0 { pos } else { neg };
            }
            r += u - node_value;
            *r32 = r as f32;
        }
    }

    fn add_strategy(&mut self, at: Span, strategy: &[f64], t: u64, variant: Variant, rng: &mut u64) {
        let weight = match variant {
            Variant::Vanilla => 1.0,
            Variant::Dcfr(d) => d.strategy_weight(t),
        };
        let s = self.strategy_sum_mut(at);
        for (sum, &p) in s.iter_mut().zip(strategy) {
            sr_add(sum, weight * p, rng);
        }
    }

    fn baseline_pair(&self, at: Span, strategy: &[f64], chosen: usize) -> (f64, f64) {
        let b = self.baseline(at);
        let exp = strategy.iter().zip(b).map(|(&p, &v)| p * v as f64).sum();
        (exp, b[chosen] as f64)
    }

    fn baseline_ema(&mut self, at: Span, a: usize, target: f64) {
        let b = self.baseline_mut(at);
        b[a] = (b[a] as f64 + BASELINE_RATE * (target - b[a] as f64)) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-to-nearest into the same array, for the tests that have to show
    /// what stochastic rounding is buying.  This is the code the store used to
    /// run when the sums were `f64`, narrowed to `f32`.
    fn rtn_add(sum: &mut f32, inc: f64) {
        *sum = (*sum as f64 + inc) as f32;
    }

    #[test]
    fn stochastic_rounding_is_unbiased() {
        // The property everything else rests on: repeatedly adding an increment
        // far below one ulp must track the exact sum, not drift from it.
        let mut rng = 12_345u64;
        let base = 1.0e7f32; // ulp here is 1.0, so +0.1 is a pure sub-ulp add.
        let mut sr = base;
        let mut rtn = base;
        const N: usize = 200_000;
        for _ in 0..N {
            sr_add(&mut sr, 0.1, &mut rng);
            rtn_add(&mut rtn, 0.1);
        }
        let exact = base as f64 + 0.1 * N as f64;
        let err = (sr as f64 - exact).abs() / exact;
        assert!(err < 1e-4, "SR tracked the exact sum to {err:.2e} (sr={sr}, exact={exact})");
        // …and RTN does not: every single add rounds back to where it started.
        assert_eq!(rtn, base, "RTN discards sub-ulp increments entirely");
    }

    #[test]
    fn strategy_sum_does_not_freeze_past_the_f32_integer_limit() {
        // The failure this change exists to prevent.  Past 2^24 an f32 cannot
        // represent consecutive integers, so accumulating +1 per visit under
        // round-to-nearest stops dead — the deployed average freezes while
        // still looking like a converged artifact.
        const N: u64 = 30_000_000; // ~1.8× the 2^24 wall
        let mut t = RegretTable::with_layout(1, |_| 1);
        let mut rng = 99u64;
        for step in 1..=N {
            t.add_strategy(t.span(0), &[1.0], step, Variant::Vanilla, &mut rng);
        }
        let got = t.strategy_sum_mut(t.span(0))[0] as f64;
        let err = (got - N as f64).abs() / N as f64;
        assert!(err < 0.01, "SR sum {got} should track {N} (off by {:.2}%)", err * 100.0);

        let mut rtn = 0.0f32;
        for _ in 0..N {
            rtn_add(&mut rtn, 1.0);
        }
        assert_eq!(rtn, 16_777_216.0, "RTN pins at 2^24 — this is the freeze");
    }

    #[test]
    fn late_strategy_shift_still_moves_the_average() {
        // Freezing matters because it makes the average stop *tracking*.  Play
        // action 0 for the first half and action 1 for the second, well past
        // the 2^24 wall: the average must end up near 50/50.  Under RTN both
        // slots pin and the answer is whatever it was at 2^24.
        const HALF: u64 = 20_000_000;
        let mut t = RegretTable::with_layout(1, |_| 2);
        let mut rng = 7u64;
        for step in 1..=HALF {
            t.add_strategy(t.span(0), &[1.0, 0.0], step, Variant::Vanilla, &mut rng);
        }
        for step in HALF + 1..=2 * HALF {
            t.add_strategy(t.span(0), &[0.0, 1.0], step, Variant::Vanilla, &mut rng);
        }
        let mut avg = Vec::new();
        t.average_into(t.span(0), &mut avg);
        assert!(
            (avg[1] - 0.5).abs() < 0.02,
            "the second half must register: got {avg:?}, RTN would give ~{:.2}",
            // RTN: slot 0 pins at 2^24, slot 1 never gets past it either.
            16_777_216.0 / (16_777_216.0 + 16_777_216.0f64)
        );
    }

    #[test]
    fn layout_places_each_info_set() {
        let t = RegretTable::with_layout(4, |i| 2 + i % 2);
        assert_eq!(t.capacity(), 4);
        assert_eq!(t.total_slots(), 10);
        assert_eq!(t.span(2), Span { base: 5, n: 2 });
        assert_eq!(t.span(3), Span { base: 7, n: 3 });
        t.validate().unwrap();
    }

    #[test]
    fn arrays_that_do_not_fit_the_layout_are_rejected() {
        // A table read from a file is trusted for its bounds afterwards, so a
        // truncated array or a foreign layout has to fail at the door.
        let layout = Layout::build(3, |_| 2);
        assert!(RegretTable::from_parts(vec![0.0; 6], vec![0.0; 6], Vec::new(), layout.clone()).is_ok());
        assert!(RegretTable::from_parts(vec![0.0; 6], vec![0.0; 6], vec![0.0; 6], layout.clone()).is_ok());
        assert!(RegretTable::from_parts(vec![0.0; 5], vec![0.0; 6], Vec::new(), layout.clone()).is_err());
        assert!(RegretTable::from_parts(vec![0.0; 6], vec![0.0; 6], vec![0.0; 4], layout).is_err());
    }

    #[test]
    fn uniform_strategy_when_no_regret() {
        let t = RegretTable::with_layout(1, |_| 4);
        let mut s = [0.0; MAX_ACTIONS];
        let n = t.strategy_into(t.span(0), &mut s);
        assert_eq!(&s[..n], &[0.25; 4]);
    }

    #[test]
    fn regret_matching_normalizes_positive_regret() {
        let mut t = RegretTable::with_layout(1, |_| 3);
        t.regret_mut(t.span(0)).copy_from_slice(&[3.0, 0.0, -2.0]);
        let mut s = [0.0; MAX_ACTIONS];
        assert_eq!(t.strategy_into(t.span(0), &mut s), 3);
        assert!((s[0] - 1.0).abs() < 1e-9 && s[1] == 0.0 && s[2] == 0.0);
    }

    #[test]
    fn the_baseline_allocates_only_when_enabled() {
        let plain = RegretTable::with_layout(10, |_| 2);
        assert!(!plain.has_baseline());
        // (f32 regret + f32 strategy-sum) × 2 actions = 16 bytes/info set, vs
        // the HashMap Node's five heap vecs (~350 B)…
        assert_eq!(plain.bytes_per_info_set(), 16);
        // …and 24 once control variates add the baseline: a third of the table.
        let mut with_baseline = RegretTable::with_layout(10, |_| 2);
        with_baseline.enable_baseline();
        assert!(with_baseline.has_baseline());
        assert_eq!(with_baseline.bytes_per_info_set(), 24);
        let at = with_baseline.span(3);
        with_baseline.baseline_ema(at, 1, 10.0);
        assert_eq!(with_baseline.baseline(at), &[0.0, 1.0], "one EMA step of 0.1 toward 10");
    }

    #[test]
    fn nothing_is_stored_per_info_set() {
        // The point of the run-length layout: a table's index costs the same
        // whether a block holds ten info sets or ten million.
        let small = RegretTable::with_layout(10, |_| 2);
        let large = RegretTable::with_layout(100_000, |_| 2);
        assert_eq!(small.layout().runs(), 1);
        assert_eq!(large.layout().runs(), 1);
    }
}
