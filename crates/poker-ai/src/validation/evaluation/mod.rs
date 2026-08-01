//! Evaluation oracles: estimators that are correct but too expensive to run in
//! the loop, kept as the reference for the cheap ones that are.
//!
//! * [`aivat`] — the full AIVAT variance-reduction estimator, controlling for
//!   both chance *and* action variance over enumerable games.  It is the
//!   conceptual oracle for [`crate::play::luck`], the chance-only control
//!   variate the real match tracker uses (real NLHE cannot enumerate the action
//!   term, so the shipped estimator drops it — this module is where the term it
//!   drops is written down and checked).
//! * [`local_br`] — sampled best response, general over
//!   [`crate::games::Game`].  Validated against exact exploitability on Leduc,
//!   then used as an upper-bound-free convergence probe on games too large for
//!   [`super::solver::best_response`].

pub mod aivat;
pub mod local_br;
