//! Warm-starting subgame solvers from blueprint values (Phase 5 resolving).
//!
//! A resolving call has only a 2–5 s budget, so it should not start the subgame
//! solver from a uniform strategy when the blueprint already has a reasonable
//! strategy at those information sets.  Warm-starting seeds the predictive
//! solver's regrets so its *first iterate* reproduces the blueprint, then lets
//! CFR⁺ refine from there.  Empirically this reaches a target exploitability in
//! noticeably fewer iterations than a cold start.
//!
//! ## How the seed works
//!
//! CFR⁺ plays the regret-matching⁺ strategy `σ(a) ∝ [Z(a)]⁺`.  So if we want the
//! solver's opening strategy to equal a blueprint `π`, we seed
//!
//! ```text
//!   Z(a) = π(a) · scale          (all non-negative ⇒ σ(a) = π(a))
//! ```
//!
//! `scale` is the *confidence* in the blueprint: it is the amount of prior regret
//! mass the blueprint carries before the subgame solver starts accumulating its
//! own.  A large scale makes the blueprint sticky (many iterations to move off
//! it); a small scale lets the subgame overrule it within a few iterations.  A
//! value on the order of the per-iteration counterfactual reach (here ~1) is a
//! reasonable default — big enough to matter on iteration one, small enough that
//! a few hundred iterations dominate it.
//!
//! ## Key compatibility
//!
//! The seed is only meaningful if the blueprint is expressed over the *same*
//! information sets the subgame exposes (same `info_key`).  In the full system
//! the subgame is built to match the blueprint's abstraction, so this holds by
//! construction; [`crate::solver::predictive::PredictiveSolver::warm_start`]
//! defensively ignores any key whose action count disagrees.

use std::collections::HashMap;

/// A reasonable default blueprint confidence (see module docs): enough prior
/// regret to set the opening strategy, small enough to be overruled quickly.
pub const DEFAULT_SCALE: f64 = 1.0;

/// Convert a blueprint strategy (`info_key → action probabilities`) into
/// warm-start regrets for a predictive subgame solver: `Z(a) = π(a) · scale`.
///
/// Pass the result to
/// [`PredictiveSolver::warm_start`](crate::solver::predictive::PredictiveSolver::warm_start)
/// before training.
pub fn warm_start_regrets(
    blueprint: &HashMap<u64, Vec<f64>>,
    scale: f64,
) -> HashMap<u64, Vec<f64>> {
    blueprint
        .iter()
        .map(|(&key, probs)| {
            let seed = probs.iter().map(|&p| p.max(0.0) * scale).collect();
            (key, seed)
        })
        .collect()
}

/// Convenience: same as [`warm_start_regrets`] but reads an `f32` blueprint
/// table (the on-disk blueprint format) directly.
pub fn warm_start_regrets_f32(
    blueprint: &HashMap<u64, Vec<f32>>,
    scale: f64,
) -> HashMap<u64, Vec<f64>> {
    blueprint
        .iter()
        .map(|(&key, probs)| {
            let seed = probs.iter().map(|&p| (p.max(0.0) as f64) * scale).collect();
            (key, seed)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_reproduces_the_blueprint_as_the_opening_strategy() {
        // Seeding Z = π·scale (all ≥ 0) means σ ∝ [Z]⁺ = π exactly.
        let mut blueprint = HashMap::new();
        blueprint.insert(7u64, vec![0.25, 0.75]);
        let seed = warm_start_regrets(&blueprint, 4.0);
        let z = &seed[&7];
        assert_eq!(z, &vec![1.0, 3.0]);
        // Regret matching over [1, 3] gives [0.25, 0.75] — the blueprint.
        let total: f64 = z.iter().sum();
        let sigma: Vec<f64> = z.iter().map(|&r| r / total).collect();
        assert!((sigma[0] - 0.25).abs() < 1e-12 && (sigma[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn f32_table_seeds_identically() {
        let mut bp32 = HashMap::new();
        bp32.insert(1u64, vec![0.5f32, 0.5]);
        let seed = warm_start_regrets_f32(&bp32, DEFAULT_SCALE);
        assert_eq!(seed[&1], vec![0.5, 0.5]);
    }
}
