//! Quantized ("lean") regret storage: `i16`/`u16` accumulators at half the RAM
//! of the `f32` [`RegretTable`](super::regret_table::RegretTable).
//!
//! ## Why this works where bf16 failed
//!
//! A bf16 store under DCFR was tried and rejected (see the note atop
//! `regret_table.rs`): DCFR keeps regrets *bounded*, so rounding noise
//! random-walks through the read-modify-write accumulators and swamps the
//! signal.  The lean store is designed for **Linear CFR**
//! ([`Discount::LINEAR`](crate::solver::dcfr::Discount::LINEAR)) instead, where
//! regret magnitudes **grow** with `t` — fixed-point quantization error becomes
//! relatively smaller as training proceeds (the Pluribus int-storage regime).
//! Per-array rounding follows the same experiment's findings:
//!
//! * **Regret (`i16`, fixed-point, round-to-nearest).**  RTN, not stochastic —
//!   SR's variance is what killed bf16.  Under LCFR the growing signal
//!   dominates the bounded RTN bias.
//! * **Strategy-sum (`u16`, stochastic rounding).**  This array is write-only
//!   and monotone growing, so SR's random walk is harmless relative to the sum
//!   — while RTN would *starve* small `σ` increments entirely (deterministic
//!   rounding of a sub-step increment is 0 forever, silently deleting
//!   low-probability actions from the average).
//! * **Baseline (`i16`, fixed-point RTN).**  An EMA is self-correcting toward
//!   its target, and a slightly-off baseline costs only variance, never bias
//!   (any fixed control variate is unbiased).
//!
//! Two structural consequences of 16-bit accumulators:
//!
//! * **Saturation → halve the info set.**  When a slot would overflow, every
//!   slot of that info set is halved.  Regret-matching and the average
//!   strategy are both *ratios within an info set*, so uniform scaling
//!   preserves the strategy exactly; for the accumulators it acts as one extra
//!   discount step, which CFR tolerates by construction.
//! * **Running-discount γ-averaging.**  The f32 store weights iteration `t`'s
//!   strategy contribution by the *absolute* `t^γ`, which no 16-bit sum can
//!   represent.  The lean store uses the equivalent running form instead —
//!   multiply the sum by `(t/(t+1))^γ` on visit, then add `σ` — applied
//!   lazily (on visit), matching how the whole solver family already applies
//!   regret discounts.

use serde::{Deserialize, Serialize};

use super::cfr::Variant;
use super::regret_table::RegretStore;
use crate::util::rng::xorshift_next_unit;

/// EMA learning rate for the VR-MCCFR baseline (mirrors `mccfr::BASELINE_RATE`).
const BASELINE_RATE: f64 = 0.1;
/// Fixed-point scale for regrets: 1/16 bb resolution, ±~2000 bb range.
const REGRET_SCALE: f64 = 16.0;
/// Fixed-point scale for baselines: 1/64 bb resolution, ±~500 bb range.
const BASELINE_SCALE: f64 = 64.0;
/// Scale for strategy-sum increments: `σ` resolution 1/64 per visit.
const STRAT_SCALE: f64 = 64.0;
/// Halve an info set when a slot magnitude crosses this (headroom below the
/// type max so one update cannot jump past it).
const REGRET_SAT: f64 = 30_000.0;
const STRAT_SAT: f64 = 60_000.0;

/// Flat quantized accumulators: 6 bytes per (info set, action) vs the f32
/// store's 12.  Same layout scheme (offsets + action counts) as `RegretTable`.
#[derive(Serialize, Deserialize)]
pub struct LeanTable {
    regret: Vec<i16>,
    strategy_sum: Vec<u16>,
    baseline: Vec<i16>,
    num_actions: Vec<u8>,
    offsets: Vec<u32>,
}

impl LeanTable {
    #[inline]
    fn span(&self, info_set: usize) -> std::ops::Range<usize> {
        let start = self.offsets[info_set] as usize;
        start..start + self.num_actions[info_set] as usize
    }
}

impl RegretStore for LeanTable {
    fn build(capacity: usize, actions: &dyn Fn(usize) -> usize) -> Self {
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
            regret: vec![0; total],
            strategy_sum: vec![0; total],
            baseline: vec![0; total],
            num_actions,
            offsets,
        }
    }

    fn capacity(&self) -> usize {
        self.num_actions.len()
    }

    fn num_actions(&self, info_set: usize) -> usize {
        self.num_actions[info_set] as usize
    }

    fn is_visited(&self, info_set: usize) -> bool {
        self.strategy_sum[self.span(info_set)].iter().any(|&x| x != 0)
    }

    fn bytes_per_info_set(&self) -> usize {
        let slots = self.regret.len();
        let cap = self.capacity().max(1);
        3 * 2 * (slots / cap)
    }

    fn strategy_into(&self, info_set: usize, out: &mut Vec<f64>) {
        let regret = &self.regret[self.span(info_set)];
        let n = regret.len();
        out.clear();
        let total: f64 = regret.iter().map(|&r| (r as f64).max(0.0)).sum();
        if total > 0.0 {
            out.extend(regret.iter().map(|&r| (r as f64).max(0.0) / total));
        } else {
            out.extend(std::iter::repeat_n(1.0 / n as f64, n));
        }
    }

    fn average_into(&self, info_set: usize, out: &mut Vec<f64>) {
        let s = &self.strategy_sum[self.span(info_set)];
        let n = s.len();
        out.clear();
        let total: f64 = s.iter().map(|&x| x as f64).sum();
        if total > 0.0 {
            out.extend(s.iter().map(|&x| x as f64 / total));
        } else {
            out.extend(std::iter::repeat_n(1.0 / n as f64, n));
        }
    }

    fn add_regret(&mut self, info_set: usize, util: &[f64], node_value: f64, t: u64, variant: Variant) {
        let (pos, neg) = match variant {
            Variant::Vanilla => (1.0, 1.0),
            Variant::Dcfr(d) => (d.positive_factor(t), d.negative_factor(t)),
        };
        let discount = matches!(variant, Variant::Dcfr(_));
        let span = self.span(info_set);
        let regret = &mut self.regret[span];

        // Compute the updated fixed-point values in f64 (RTN), then store —
        // halving the whole info set first if any slot would saturate.
        let mut new = [0.0f64; 8];
        let mut peak = 0.0f64;
        for (a, (&r16, &u)) in regret.iter().zip(util).enumerate() {
            let mut r = r16 as f64 / REGRET_SCALE;
            if discount {
                r *= if r > 0.0 { pos } else { neg };
            }
            r += u - node_value;
            let fp = (r * REGRET_SCALE).round();
            new[a] = fp;
            peak = peak.max(fp.abs());
        }
        let scale = if peak > REGRET_SAT { 0.5 } else { 1.0 };
        for (r16, &fp) in regret.iter_mut().zip(&new) {
            *r16 = (fp * scale).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
    }

    fn add_strategy(&mut self, info_set: usize, strategy: &[f64], t: u64, variant: Variant, rng: &mut u64) {
        // Running-discount γ-averaging (see module doc), applied lazily on visit.
        let factor = super::regret_table::strategy_discount(t, variant);
        let span = self.span(info_set);
        let sums = &mut self.strategy_sum[span];

        let mut new = [0.0f64; 8];
        let mut peak = 0.0f64;
        for (a, (&s16, &p)) in sums.iter().zip(strategy).enumerate() {
            // Discount (RTN — a multiplicative step, no starvation risk), then
            // add the increment with STOCHASTIC rounding so sub-step σ mass
            // accumulates in expectation instead of rounding to zero forever.
            let discounted = (s16 as f64 * factor).round();
            let inc = p * STRAT_SCALE;
            let sr = inc.floor() + f64::from(inc.fract() > xorshift_next_unit(rng));
            let fp = discounted + sr;
            new[a] = fp;
            peak = peak.max(fp);
        }
        let scale = if peak > STRAT_SAT { 0.5 } else { 1.0 };
        for (s16, &fp) in sums.iter_mut().zip(&new) {
            *s16 = (fp * scale).round().clamp(0.0, u16::MAX as f64) as u16;
        }
    }

    fn baseline_pair(&self, info_set: usize, strategy: &[f64], chosen: usize) -> (f64, f64) {
        let b = &self.baseline[self.span(info_set)];
        let exp = strategy.iter().zip(b).map(|(&p, &v)| p * v as f64 / BASELINE_SCALE).sum();
        (exp, b[chosen] as f64 / BASELINE_SCALE)
    }

    fn baseline_ema(&mut self, info_set: usize, a: usize, target: f64) {
        debug_assert!(a < self.num_actions[info_set] as usize);
        let b = &mut self.baseline[self.offsets[info_set] as usize + a];
        let cur = *b as f64 / BASELINE_SCALE;
        let next = cur + BASELINE_RATE * (target - cur);
        *b = (next * BASELINE_SCALE).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::dcfr::Discount;

    fn table2() -> LeanTable {
        LeanTable::build(2, &|_| 3)
    }

    #[test]
    fn strategy_from_positive_regrets() {
        let mut t = table2();
        // Regrets 1.0 bb / 3.0 bb / negative → RM proportions 0.25 / 0.75 / 0.
        t.add_regret(0, &[1.0, 3.0, -2.0], 0.0, 1, Variant::Vanilla);
        let mut s = Vec::new();
        t.strategy_into(0, &mut s);
        assert!((s[0] - 0.25).abs() < 1e-9 && (s[1] - 0.75).abs() < 1e-9 && s[2] == 0.0);
    }

    #[test]
    fn saturation_halves_preserve_the_strategy() {
        let mut t = table2();
        // Drive one slot to saturation; proportions must survive the halvings.
        for step in 1..=400u64 {
            t.add_regret(0, &[80.0, 40.0, 0.0], 0.0, step, Variant::Vanilla);
        }
        let mut s = Vec::new();
        t.strategy_into(0, &mut s);
        assert!((s[0] - 2.0 * s[1]).abs() < 0.01, "2:1 regret ratio preserved: {s:?}");
        assert!(t.regret[t.span(0)].iter().all(|&r| (r as f64) <= REGRET_SAT));
    }

    #[test]
    fn stochastic_rounding_accumulates_small_strategy_mass() {
        let mut t = table2();
        let mut rng = 42u64;
        // σ = 0.005 is far below the 1/64 RTN step; SR must still accumulate it.
        for step in 1..=20_000u64 {
            t.add_strategy(0, &[0.99, 0.005, 0.005], step, Variant::Vanilla, &mut rng);
        }
        let mut avg = Vec::new();
        t.average_into(0, &mut avg);
        assert!(avg[1] > 0.002 && avg[1] < 0.010, "small action retained: {avg:?}");
    }

    #[test]
    fn baseline_ema_tracks_target() {
        let mut t = table2();
        for _ in 0..200 {
            t.baseline_ema(1, 2, 7.5);
        }
        let (_, b) = t.baseline_pair(1, &[0.0, 0.0, 1.0], 2);
        // A quantized EMA rests within step/(2·rate) = (1/64)/0.2 ≈ 0.078 of its
        // target (RTN stalls once the update is below half a step).  A baseline
        // that close costs only a sliver of variance reduction, never bias.
        assert!((b - 7.5).abs() < 0.08, "EMA within the quantization resting gap: {b}");
    }

    #[test]
    fn lcfr_discounting_runs_and_visits_mark() {
        let mut t = table2();
        let mut rng = 7u64;
        assert!(!t.is_visited(0));
        for step in 1..=100u64 {
            t.add_regret(0, &[1.0, 0.0, 0.0], 0.2, step, Variant::Dcfr(Discount::LINEAR));
            t.add_strategy(0, &[0.5, 0.3, 0.2], step, Variant::Dcfr(Discount::LINEAR), &mut rng);
        }
        assert!(t.is_visited(0));
        assert_eq!(t.bytes_per_info_set(), 18); // 3 arrays × 2 B × 3 actions
    }
}
