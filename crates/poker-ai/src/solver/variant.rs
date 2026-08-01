//! Which regret-update regime a solver uses.
//!
//! Shared by every solver in the crate — the production sampled trainers
//! ([`crate::solver::mccfr`] and its SoA/atomic/parallel paths) and the
//! full-traversal validation solver
//! ([`crate::validation::solver::full_cfr::Cfr`]) — so that "vanilla vs DCFR"
//! means the same thing on both sides of the oracle gates.  It lives here rather
//! than beside either solver so neither has to depend on the other.

use super::dcfr::Discount;

/// Which regret-update regime the solver uses.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    /// Undiscounted textbook CFR: regrets accumulate unscaled and the average
    /// strategy weights every iteration equally.  Simplest to trust.
    Vanilla,
    /// Discounted CFR with the full `(α, β, γ)` schedule from
    /// [`crate::solver::dcfr`].
    Dcfr(Discount),
}
