//! Depth-limited subgame solver.
//!
//! Runs DCFR within the subgame using blueprint values at leaf nodes.
//! Depth limit: 1–2 streets. Time budget: 2–5 seconds.

pub struct SubgameSolver {
    pub depth_limit: u32,
    pub time_budget_ms: u64,
}

impl SubgameSolver {
    pub fn new(depth_limit: u32, time_budget_ms: u64) -> Self {
        Self { depth_limit, time_budget_ms }
    }

    /// Solve the subgame rooted at `state` and return the strategy.
    pub fn solve(
        &self,
        _state: &poker_core::state::GameState,
        _beliefs: &[crate::resolving::belief_state::BeliefState],
        _leaf_eval: &dyn crate::resolving::leaf_eval::LeafEvaluator,
    ) -> Vec<f32> {
        todo!()
    }
}
