//! Hand strength evaluation. Wraps a lookup-table evaluator.
//!
//! Provides 5-, 6-, and 7-card evaluation for use during equity calculation.

/// Evaluate the strength of a 7-card hand. Returns a rank where higher is better.
pub fn evaluate_7(_cards: &[u8; 7]) -> u16 {
    todo!()
}

/// Evaluate a 5-card hand.
pub fn evaluate_5(_cards: &[u8; 5]) -> u16 {
    todo!()
}
