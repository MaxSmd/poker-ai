//! Local Best Response exploitability lower bound.
//!
//! Lisy & Bowling, IJCAI 2016 workshop.
//! Fixes the target strategy and computes best response on a sampled subtree.

/// Estimate exploitability using Local Best Response.
/// Returns exploitability in bb/hand (lower bound).
pub fn local_best_response(
    _strategy: &crate::solver::regret_table::RegretTable,
    _num_samples: usize,
) -> f64 {
    todo!()
}
