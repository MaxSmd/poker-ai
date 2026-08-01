//! Exact solvers and exact best response — the answers the sampled production
//! trainers are measured against.
//!
//! * [`full_cfr`] — full-tree CFR.  Deterministic and exact; the reference
//!   implementation [`crate::solver::mccfr`] must match on the toy games.
//! * [`best_response`] — exact best response and exploitability by full
//!   traversal.  The scoring function for every convergence gate.
//! * [`predictive`] — optimistic/predictive regret updates over the full tree;
//!   the reference for the production `--optimistic` path.

pub mod best_response;
pub mod full_cfr;
pub mod predictive;
