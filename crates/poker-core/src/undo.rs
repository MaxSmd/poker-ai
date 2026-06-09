//! Undo stack for zero-allocation tree traversal.
//!
//! Records a *delta* of the mutable fields that changed during each
//! `apply_action` call so that `undo_action` can restore the state exactly.
//!
//! ## Delta layout (compared to the old full-snapshot approach)
//!
//! The old full-snapshot stored **three** `[u32; MAX_PLAYERS]` arrays (stacks,
//! street_bets, total_committed) regardless of the action.  Only one player's
//! values ever change per action, so we now store:
//!   - The acting player's index (1 byte)
//!   - That player's old stack / street_bet / total_committed (3 × 4 bytes)
//!   - All scalar fields that may change (≈14 bytes)
//!   - An `old_street_bets` array used **only** when the betting round closes
//!     and `advance_street` resets every player's street bet to zero.
//!
//! Typical record size: ~60 bytes vs ~88 bytes for the old snapshot.
//! With a 256-deep pre-allocated stack this is ~15 KB vs ~22 KB per
//! `GameState`.
//!
//! The stack is pre-allocated; `push` and `pop` do not allocate.

use crate::action::Action;
use crate::state::MAX_PLAYERS;

pub const MAX_UNDO_DEPTH: usize = 256;

/// A delta-based undo record for a single `apply_action` call.
///
/// Only the fields that may have changed are stored:
/// - The acting player's per-player values (stack, street_bet, total_committed).
/// - All mutable scalar fields (street, to_act, current_bet, …).
/// - When the betting round closes and a street advances, `street_changed` is
///   set and `old_street_bets` holds the full pre-reset array so that all
///   players' street bets can be restored.
#[derive(Clone, Copy, Debug)]
pub struct UndoRecord {
    pub action: Action,

    /// Index of the player who acted.
    pub player: u8,

    // ── Per-player deltas (acting player only) ──────────────────────────────
    pub old_stack: u32,
    pub old_street_bet: u32,
    pub old_total_committed: u32,

    // ── Scalar fields (may all change via advance_or_next) ──────────────────
    pub old_street: u8,
    pub old_to_act: u8,
    pub old_current_bet: u32,
    pub old_min_raise: u32,
    pub old_folded: u8,
    pub old_allin: u8,
    pub old_last_aggressor: u8,
    pub old_players_to_act: u8,
    pub old_pot: u32,

    // ── Street-transition data ───────────────────────────────────────────────
    /// `true` when `advance_street` fired during this action, resetting all
    /// players' street bets.  When restoring, use `old_street_bets` for the
    /// full array instead of just patching the acting player's slot.
    pub street_changed: bool,
    /// Pre-action values of all players' street bets.  Only meaningful (and
    /// used during undo) when `street_changed` is `true`.
    pub old_street_bets: [u32; MAX_PLAYERS],
}

/// Fixed-capacity undo stack.  All operations are O(1) and allocation-free
/// after the initial `new()` call (which pre-allocates `MAX_UNDO_DEPTH` slots).
#[derive(Clone, Debug)]
pub struct UndoStack {
    records: Vec<UndoRecord>,
}

impl Default for UndoStack {
    fn default() -> Self {
        Self::new()
    }
}

impl UndoStack {
    /// Create a new stack pre-allocated for `MAX_UNDO_DEPTH` entries.
    pub fn new() -> Self {
        Self {
            records: Vec::with_capacity(MAX_UNDO_DEPTH),
        }
    }

    /// Push a record.  Panics if the stack would exceed `MAX_UNDO_DEPTH`.
    pub fn push(&mut self, record: UndoRecord) {
        assert!(
            self.records.len() < MAX_UNDO_DEPTH,
            "undo stack overflow — game tree too deep"
        );
        self.records.push(record);
    }

    /// Pop the most recent record, or `None` if the stack is empty.
    pub fn pop(&mut self) -> Option<UndoRecord> {
        self.records.pop()
    }

    /// Mark the most-recently-pushed record as having triggered a street
    /// transition.  Called by `apply_action` after `advance_or_next` when
    /// `self.street` changed.
    pub fn mark_street_changed(&mut self) {
        if let Some(rec) = self.records.last_mut() {
            rec.street_changed = true;
        }
    }

    /// Current depth of the stack.
    pub fn depth(&self) -> usize {
        self.records.len()
    }

    /// Remove all records (does not release memory).
    pub fn clear(&mut self) {
        self.records.clear();
    }
}
