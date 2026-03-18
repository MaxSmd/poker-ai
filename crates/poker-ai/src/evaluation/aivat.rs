//! AIVAT variance-reduced win rate estimation.
//!
//! Burch, Johanson & Bowling, 2018.
//! Uses the known blueprint strategy as a control variate.

/// Compute the AIVAT-adjusted win rate from a sequence of game results.
/// `outcomes` — raw chip outcomes per hand
/// Returns adjusted win rate in bb/hand with tighter confidence interval.
pub fn aivat_win_rate(_outcomes: &[f64], _blueprint_ev: &[f64]) -> f64 {
    todo!()
}
