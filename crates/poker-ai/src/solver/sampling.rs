//! Public chance sampling with external sampling for opponent hands.
//!
//! Sample the public board cards once per iteration, then traverse all player
//! paths. Opponent hands are externally sampled (one hand per opponent).

pub struct SamplingConfig {
    /// Number of opponent hand samples per traversal.
    pub opponent_samples: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { opponent_samples: 1 }
    }
}
