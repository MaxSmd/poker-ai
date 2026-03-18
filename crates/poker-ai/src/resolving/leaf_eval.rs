//! Pluggable leaf evaluator trait for depth-limited resolving.

use poker_core::state::GameState;
use crate::resolving::belief_state::BeliefState;

/// Evaluate the expected value at a subgame leaf node for each player.
pub trait LeafEvaluator {
    fn evaluate(&self, state: &GameState, beliefs: &[BeliefState]) -> Vec<f64>;
}

/// Default leaf evaluator: looks up values from the blueprint regret table.
pub struct BlueprintLeafEval {
    // TODO: reference to the loaded blueprint
}

impl LeafEvaluator for BlueprintLeafEval {
    fn evaluate(&self, _state: &GameState, _beliefs: &[BeliefState]) -> Vec<f64> {
        todo!()
    }
}
