//! EHS, EHS², and draw potential feature computation for abstraction clustering.

/// Expected Hand Strength against a random opponent hand distribution.
pub fn ehs(_hole_cards: &[u8; 2], _board: &[u8]) -> f64 {
    todo!()
}

/// Second moment of hand strength distribution (variance proxy).
pub fn ehs2(_hole_cards: &[u8; 2], _board: &[u8]) -> f64 {
    todo!()
}

/// Probability of improving to a strong hand (draw potential).
pub fn draw_potential(_hole_cards: &[u8; 2], _board: &[u8]) -> f64 {
    todo!()
}

/// Build a discretized EHS histogram with `bins` buckets.
pub fn ehs_histogram(_hole_cards: &[u8; 2], _board: &[u8], bins: usize) -> Vec<f64> {
    vec![0.0; bins]
}
