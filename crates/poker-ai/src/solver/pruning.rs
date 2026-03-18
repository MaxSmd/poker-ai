//! Regret-Based Pruning (RBP).
//!
//! Brown & Sandholm, NIPS 2015.
//! Configurable θ (regret threshold) and K (consecutive iteration threshold).

pub struct PruningConfig {
    /// Regret threshold below which a branch is pruned.
    pub theta: f32,
    /// Number of consecutive iterations a branch must be below theta to be pruned.
    pub k: u32,
    /// Fraction of training complete before pruning is enabled (e.g. 0.2 = 20%).
    pub start_fraction: f64,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            theta: -300.0,
            k: 10,
            start_fraction: 0.2,
        }
    }
}

/// Returns true if a branch should be pruned given its cumulative regret history.
pub fn should_prune(_regrets: &[f32], _config: &PruningConfig, _consecutive_below: u32) -> bool {
    todo!()
}
