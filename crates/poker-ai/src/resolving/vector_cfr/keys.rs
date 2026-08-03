//! Info-key construction for the vectorized solver.
//!
//! The emitted keys must be **byte-identical** to the explicit-deal
//! [`Subgame::info_key`](crate::validation::resolving::subgame), because that oracle scores
//! this solver's output — so the hashing lives in one place, here.

use poker_core::state::NO_CARD;

use crate::util::hash::fnv1a;

/// Key-namespace markers for [`NodeKind::Decision`](super::node::NodeKind).
pub(super) const MARKER_NONE: u8 = 0;
pub(super) const MARKER_CONTINUATION: u8 = 0xFE;
pub(super) const MARKER_GADGET: u8 = 0xA6;

/// The key under which [`VectorResolved::strategy`](super::VectorResolved::strategy) stores a hand's
/// distribution (the explicit `Subgame::info_key`).  `hole` must be sorted
/// ascending; `history` is the action-index path from the resolve root
/// (empty at the root itself).  Betting nodes only — continuation-choice nodes
/// carry an extra marker and are never queried by hole+history.
pub fn subgame_info_key(player: usize, hole: [u8; 2], board: &[u8; 5], history: &[u8]) -> u64 {
    info_key(player, hole, board, history, MARKER_NONE)
}

/// Reproduce [`Subgame::info_key`](crate::validation::resolving::subgame): FNV-1a of
/// `player`, the (sorted) hole, the visible board, a separator, then the action
/// history.  `combo_cards` already returns `a < b`, matching the sort there.
/// A nonzero `marker` byte is appended so a depth-limit continuation choice
/// (`0xFE`) or gadget choice (`0xA6`) can never collide with a betting info
/// set at the same key.
pub(super) fn info_key(player: usize, hole: [u8; 2], board: &[u8; 5], history: &[u8], marker: u8) -> u64 {
    let mut bytes = Vec::with_capacity(8 + history.len());
    bytes.push(player as u8);
    bytes.push(hole[0]);
    bytes.push(hole[1]);
    for &c in board {
        if c != NO_CARD {
            bytes.push(c);
        }
    }
    bytes.push(0xFF);
    bytes.extend_from_slice(history);
    if marker != MARKER_NONE {
        bytes.push(marker);
    }
    fnv1a(&bytes)
}