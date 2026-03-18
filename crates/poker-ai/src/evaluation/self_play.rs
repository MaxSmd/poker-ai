//! Head-to-head match runner for evaluating bot versions.

pub struct MatchConfig {
    pub num_hands: u32,
    pub use_aivat: bool,
}

/// Run a head-to-head match and return the win rate in bb/hand.
pub fn run_match(_config: &MatchConfig) -> f64 {
    todo!()
}
