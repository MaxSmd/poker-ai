//! Regret-Based Pruning (RBP) — Brown & Sandholm, NIPS 2015.
//!
//! In CFR most of the tree is, late in training, provably suboptimal: an action
//! whose cumulative regret has been deeply negative for a long stretch will keep
//! its regret-matching probability pinned at zero, so traversing its subtree
//! contributes nothing but compute.  RBP **stops traversing** such branches and
//! spends the saved iterations on the parts of the tree that still matter.
//!
//! The policy is governed by two parameters the plan asks to keep configurable
//! and sweep on Leduc:
//!
//! * **θ (`theta`)** — the regret threshold.  An action counts as "below
//!   threshold" on an iteration when its cumulative regret is `< θ` (θ is
//!   negative).  A more negative θ prunes more conservatively.
//! * **K (`k`)** — an action is pruned only after it has been below θ for `k`
//!   *consecutive* iterations, so a single noisy dip does not freeze a branch.
//!
//! Pruning is enabled only after `start_fraction` of training (the plan's "first
//! 20%"), giving regrets time to settle before any branch is frozen.
//!
//! ## Interaction safeguards (the plan's three-way warning)
//!
//! Pruning interacts with both optimistic updates and the VR-MCCFR baseline: a
//! frozen branch stops receiving the optimistic correction *and* stops updating
//! its baseline, so when it is re-expanded its control variate is stale.  The
//! safeguard is a **periodic full (unpruned) traversal** every `refresh_interval`
//! iterations that re-touches every branch — refreshing baselines and letting a
//! branch whose regret has recovered above θ reset its counter and rejoin play.
//! The solver owns that schedule; this module owns the per-action predicate.

/// Tunable RBP parameters.  Defaults are a reasonable Leduc starting point; the
/// plan asks for a θ/K sensitivity sweep before committing them at scale.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PruningConfig {
    /// Regret threshold below which an action counts as a pruning candidate.
    pub theta: f64,
    /// Consecutive below-θ iterations required before an action is pruned.
    pub k: u32,
    /// Fraction of the iteration budget to train before pruning turns on.
    pub start_fraction: f64,
    /// Do a full unpruned traversal every this many iterations (baseline refresh
    /// + re-expansion check).  `0` disables the refresh.
    pub refresh_interval: u64,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self { theta: -300.0, k: 10, start_fraction: 0.2, refresh_interval: 1_000 }
    }
}

impl PruningConfig {
    /// Whether a cumulative regret value counts as below the pruning threshold
    /// this iteration.
    #[inline]
    pub fn below_threshold(&self, regret: f64) -> bool {
        regret < self.theta
    }

    /// Whether an action with this consecutive-below-θ streak should be pruned
    /// (skipped) on the current traversal.
    #[inline]
    pub fn should_prune(&self, consecutive_below: u32) -> bool {
        consecutive_below >= self.k
    }

    /// Whether `iteration` is a scheduled full-refresh iteration (no pruning, so
    /// every branch is re-touched).  Always true on iteration 0 of a cycle.
    #[inline]
    pub fn is_refresh_iteration(&self, iteration: u64) -> bool {
        self.refresh_interval != 0 && iteration % self.refresh_interval == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_and_streak_predicates() {
        let cfg = PruningConfig { theta: -100.0, k: 5, start_fraction: 0.2, refresh_interval: 0 };
        assert!(cfg.below_threshold(-150.0));
        assert!(!cfg.below_threshold(-50.0));
        assert!(!cfg.should_prune(4), "below the streak requirement");
        assert!(cfg.should_prune(5), "meets the streak requirement");
        assert!(cfg.should_prune(20));
    }

    #[test]
    fn refresh_schedule() {
        let cfg = PruningConfig { theta: -100.0, k: 5, start_fraction: 0.2, refresh_interval: 100 };
        assert!(cfg.is_refresh_iteration(200));
        assert!(!cfg.is_refresh_iteration(150));

        let never = PruningConfig { refresh_interval: 0, ..cfg };
        assert!(!never.is_refresh_iteration(100), "refresh disabled");
    }
}
