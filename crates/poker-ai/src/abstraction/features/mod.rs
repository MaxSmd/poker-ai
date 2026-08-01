//! Hand-strength and equity features for card abstraction (Phase 2).
//!
//! The quality of card bucketing sets the strategic ceiling of the whole bot,
//! and bucketing clusters on these features.  The atomic primitive is
//! [`river_equity`]: the exact probability that a hand beats a uniformly random
//! opponent hand on a *complete* board.  Everything else — expected hand
//! strength over future runouts, its second moment, draw potential, and the
//! equity-distribution histogram the clusterer actually consumes — is built by
//! averaging that primitive over the possible board completions.
//!
//! These are computed exactly (full enumeration).  Exact is the right choice
//! for correctness and for the river/turn; the flop's ~10⁶-evaluation cost per
//! hand is why Phase 2 caches results by suit-isomorphic key and (later) uses
//! Monte-Carlo rollouts for the widest layers.  Correctness first, speed via the
//! cache second.

mod equity;
mod ochs;
mod sweep;
#[cfg(test)]
mod tests;

// The canonical combo bijection (one crate-wide ordering — see `util::combos`
// for why there must be exactly one); re-exported here because the sweeps
// define and consume it.
pub use crate::util::combos::{combo_cards, combo_index};

pub use equity::{
    draw_potential, ehs, ehs2, ehs_histogram, hand_vs_hand_equity, river_equity,
};
pub use ochs::{board_ochs, ochs_opponent_clusters, OCHS_K};
pub use sweep::{
    board_cfvs, board_equities, board_histograms, board_runout_cfvs, PreparedRunout,
    PreparedShowdown,
};
