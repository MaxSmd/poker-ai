//! The explicit-deal re-solving stack: continual re-solving written the slow,
//! obvious way.
//!
//! [`subgame::Subgame`] enumerates every deal in a subgame as its own tree node
//! and solves it with full-traversal CFR.  That is hopeless at real scale — it
//! is exactly what [`crate::resolving::vector_cfr`] replaces with one vectorized
//! pass over the public tree — but it is transparently correct, which makes it
//! the oracle the vectorized solver is held to
//! (`vectorized_flop_resolve_agrees_with_explicit_oracle`).
//!
//! The rest is the machinery that turns a solved subgame into a *safe*
//! re-solve: [`gadget`] (the CFV gadget giving the opponent the option to opt
//! out at their blueprint value), [`cfv`] (counterfactual values at the subgame
//! root), [`leaf_eval`] (depth-limit leaf values), [`warm_start`] (seeding
//! regrets from the blueprint) and [`continual`] (chaining re-solves across
//! streets).
//!
//! Belief tracking ([`crate::resolving::belief_state`]) is *not* here: it is
//! production code, used by the live bot on every decision.

pub mod cfv;
pub mod continual;
pub mod gadget;
pub mod leaf_eval;
pub mod subgame;
pub mod warm_start;
