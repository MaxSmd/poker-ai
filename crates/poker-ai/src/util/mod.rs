//! Small shared utilities used across the crate.
//!
//! These live here rather than inside a feature module so that no consumer has
//! to reach into an unrelated module for a generic helper (e.g. the resolver
//! and several games sharing one FNV hash, or every sampler sharing one PRNG).

pub mod hash;
pub mod rng;
