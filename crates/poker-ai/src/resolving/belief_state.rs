//! Per-opponent marginal hand distributions.
//!
//! Uses independent marginals (one distribution per opponent) as a tractable
//! approximation to the full joint distribution.

/// Belief distribution over one opponent's hole cards.
pub struct BeliefState {
    /// Probability for each possible hole card combination (1326 entries for NLHE).
    pub probs: Vec<f64>,
}

impl BeliefState {
    /// Initialise with a uniform prior over all 1326 combinations.
    pub fn uniform() -> Self {
        let n = 1326;
        Self {
            probs: vec![1.0 / n as f64; n],
        }
    }

    /// Update beliefs given an observed action using Bayes' rule.
    pub fn update(&mut self, _action: poker_core::action::Action, _strategy: &[f64]) {
        todo!()
    }
}
