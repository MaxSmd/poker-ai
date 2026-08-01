//! The production solvers: sampled, memory-bounded, fast.
//!
//! Their exact counterparts — full-traversal CFR, exact best response,
//! predictive updates — live in [`crate::validation::solver`], which is what
//! these are gated against.

pub mod dcfr;
pub mod lean_table;
pub mod mccfr;
pub mod pruning;
pub mod regret_table;
pub mod variant;
