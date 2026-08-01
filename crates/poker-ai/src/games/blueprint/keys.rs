//! Info-key construction: how a `(player, cards, board, history)` situation
//! becomes the `u64` the solver keys on.
//!
//! Both traversal paths (clone-based `Game` and the zero-alloc `CursorGame`)
//! funnel through [`BlueprintHoldem::info_key_for`], so their keys are
//! byte-identical by construction — the property the bit-identical training
//! gates rely on.  The card half is the per-street bucket (or the raw dense
//! hand index where no abstraction is loaded).

use poker_core::state::GameState;

use super::BlueprintHoldem;
use crate::abstraction::canonical::preflop_index;
use crate::util::hash::Fnv1a;

impl BlueprintHoldem {
    /// The abstracted information key for the situation `(hole, board)` at the
    /// given street: a bucket id when an abstraction covers it, otherwise the
    /// suit-canonical key folded to `u64`.
    pub(super) fn situation_bucket(&self, hole: &[u8; 2], board: &[u8]) -> u64 {
        let visible = board.len();
        if visible == 0 {
            // Pre-flop: the 169 suit-canonical starting-hand classes.
            return preflop_index(hole) as u64;
        }
        let street = visible - 3; // flop = 0, turn = 1, river = 2
        match self.street_buckets.get(street).and_then(Option::as_ref) {
            Some(map) => match map.bucket(hole, board) {
                Some(b) => b as u64,
                // Outside the built set: stay correct by not abstracting.
                None => self.raw_index(street, hole, board),
            },
            None => self.raw_index(street, hole, board),
        }
    }

    /// Unabstracted key for a post-flop street: the raw dense hand index (which
    /// is itself suit-isomorphic and collision-free).  Used when no bucket map
    /// covers the situation — the same role the suit-canonical key played before.
    fn raw_index(&self, street: usize, hole: &[u8; 2], board: &[u8]) -> u64 {
        let mut cards = [0u8; 7];
        cards[0] = hole[0];
        cards[1] = hole[1];
        cards[2..2 + board.len()].copy_from_slice(board);
        self.indexers[street].index(&cards[..2 + board.len()])
    }

    /// Fold the information-set key for the acting player at `gs` with the given
    /// perfect-recall `history` (action indices), streamed straight into FNV-1a
    /// so neither the clone-based nor the cursor-based path allocates a `Vec`.
    pub(super) fn info_key_for(&self, gs: &GameState, history: &[u8]) -> u64 {
        let player = gs.current_player();
        let hole = gs.hole_cards[player];
        let visible = gs.board_cards_count();
        self.key_for_cards(player, hole, &gs.board[..visible], history)
    }

    /// The information key `player` would have holding `hole` at the public
    /// situation `(board, history)` — the shared kernel of [`Game::info_key`]
    /// and the play-time belief updates (which ask "what key — and hence what
    /// blueprint strategy — would the opponent have with *this* hand?").
    pub(super) fn key_for_cards(&self, player: usize, mut hole: [u8; 2], board: &[u8], history: &[u8]) -> u64 {
        hole.sort_unstable();
        let bucket = self.situation_bucket(&hole, board);
        self.key_from_bucket(player, board.len(), bucket, history)
    }

    /// Fold an already-computed card `bucket` into the info key — the hashing
    /// kernel of `key_for_cards`, public so bulk walkers
    /// (`evaluation::vector_br`) can hoist the per-hand bucket computation out
    /// of the per-node loop and still land on identical blueprint keys.
    pub fn key_from_bucket(&self, player: usize, visible: usize, bucket: u64, history: &[u8]) -> u64 {
        let mut h = Fnv1a::new();
        h.write(player as u8);
        h.write(visible as u8);
        h.write_all(&bucket.to_le_bytes());
        h.write(0xFF); // separator so bucket bytes / history can't blur
        h.write_all(history);
        h.finish()
    }
}
