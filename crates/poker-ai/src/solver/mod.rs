//! The production solvers: sampled, memory-bounded, fast.
//!
//! Their exact counterparts — full-traversal CFR, exact best response,
//! predictive updates — live in [`crate::validation::solver`], which is what
//! these are gated against.

/// Widest action fan-out any solver has to hold on the stack.
///
/// The engine's [`ActionList`](poker_core::action::ActionList) caps at 8, so a
/// `[_; MAX_ACTIONS]` buffer covers every information set in every game the
/// production solvers train — which is what lets the traversals keep their
/// per-node scratch (strategy, per-action utilities, RBP streaks) on the stack
/// instead of in a heap `Vec`.  Paths that can be checked up front assert it
/// against the table layout before training starts; the rest assert per node.
pub const MAX_ACTIONS: usize = 8;

pub mod dcfr;
pub mod lean_table;
pub mod mccfr;
pub mod pruning;
pub mod regret_table;
pub mod variant;
