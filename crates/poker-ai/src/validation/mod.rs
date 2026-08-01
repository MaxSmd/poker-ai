//! Oracles: code that exists **only to check the production code**.
//!
//! Nothing in this module ships.  No path reachable from the `train`, `cluster`
//! or `play` binaries links a single symbol from here — that is the defining
//! property, and it is worth preserving deliberately, because the two halves of
//! this crate have opposite design goals:
//!
//! * **Production** ([`crate::abstraction`], [`crate::games::blueprint`],
//!   [`crate::solver::mccfr`], [`crate::resolving::vector_cfr`],
//!   [`crate::play`], [`crate::evaluation::vector_br`]) is written for scale:
//!   flat Structure-of-Arrays storage, quantized tables, sampled traversals,
//!   vectorized public trees, lock-free threading.  Every one of those choices
//!   trades obviousness for speed or memory.
//! * **Validation** (here) is written for *obviousness*: full-tree traversal,
//!   `HashMap` storage, explicit per-deal enumeration, exact best response.  It
//!   is allowed to be arbitrarily slow, because it only ever runs on games small
//!   enough to solve exactly.
//!
//! The oracle gates pair them up.  Each production optimization is admitted only
//! once it reproduces its slow twin's answer:
//!
//! | Production | is checked against |
//! |---|---|
//! | [`crate::solver::mccfr`] (sampled, SoA, atomic) | [`solver::full_cfr::Cfr`] on [`games::kuhn`] / [`games::leduc`], scored by [`solver::best_response::exploitability`] |
//! | [`crate::resolving::vector_cfr`] (vectorized public tree) | [`resolving::subgame::Subgame`] (explicit per-deal) |
//! | [`crate::play::luck`] (chance-only control variate) | [`evaluation::aivat`] (the full AIVAT estimator) |
//! | [`crate::games::blueprint`] mechanics | [`games::nlhe`] (curated-deal bridge) |
//! | [`crate::solver::mccfr`] convergence on real mechanics | [`evaluation::local_br`] (sampled BR, general over [`crate::games::Game`]) |
//!
//! **If you are optimizing something in this module, stop.**  Slowness here is
//! not a defect; it is what makes the answer trustworthy.  The only reason to
//! touch these files is that the production side grew a behaviour they do not
//! yet check.
//!
//! Everything here depends on production code and never the reverse — production
//! modules reference this one only from `#[cfg(test)]` blocks and doc links.
//! That direction is what keeps "does this ship?" answerable by looking at the
//! path.

pub mod evaluation;
pub mod games;
pub mod resolving;
pub mod solver;
