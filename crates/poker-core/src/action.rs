//! Action enum and legal action generation.

/// A player action in No-Limit Hold'em.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Fold,
    Check,
    Call,
    Raise(u32), // amount in chips
    AllIn,
}

/// Returns the legal actions available in `state`.
pub fn legal_actions(_state: &crate::state::GameState) -> Vec<Action> {
    todo!()
}
