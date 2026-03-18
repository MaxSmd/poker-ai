//! Warm-start subgame solvers from blueprint values.

/// Initialise a subgame regret table from the blueprint strategy at the
/// subgame root, reducing the number of iterations needed to converge.
pub fn warm_start_from_blueprint(
    _state: &poker_core::state::GameState,
) -> crate::solver::regret_table::RegretTable {
    todo!()
}
