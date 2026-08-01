//! Production re-solving: belief tracking and the vectorized public-tree solver
//! the live bot runs.
//!
//! The explicit-deal stack it replaces — [`Subgame`](crate::validation::resolving::subgame),
//! the CFV gadget, leaf evaluators, warm starts, continual chaining — is the
//! oracle, and lives in [`crate::validation::resolving`].

pub mod belief_state;
pub mod vector_cfr;
