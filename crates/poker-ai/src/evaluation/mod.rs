//! The metrics that run in the loop: cheap enough to call every checkpoint.
//!
//! Their expensive reference counterparts (AIVAT, sampled BR) live in
//! [`crate::validation::evaluation`].

pub mod exploitability;
pub mod vector_br;
