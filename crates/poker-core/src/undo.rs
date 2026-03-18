//! Undo stack for zero-allocation tree traversal.
//!
//! Records a complete snapshot of all mutable fields before each `apply_action`
//! call so that `undo_action` can restore the state exactly.  Each record is a
//! full copy (not a delta), which keeps the implementation simple and correct
//! at the cost of ~88 bytes per entry.  With a 256-deep pre-allocated stack
//! this amounts to ~22 KB per `GameState` — acceptable for CFR traversal.
//!
//! The stack is pre-allocated; `push` and `pop` do not allocate.

use crate::action::Action;
use crate::state::MAX_PLAYERS;

pub const MAX_UNDO_DEPTH: usize = 256;

/// A complete snapshot of all mutable `GameState` fields before a single
/// `apply_action` call.  Restoring the state is done by overwriting all
/// mutable fields with these values.
#[derive(Clone, Copy, Debug)]
pub struct UndoRecord {
    pub action: Action,
    pub stacks: [u32; MAX_PLAYERS],
    pub street_bets: [u32; MAX_PLAYERS],
    pub total_committed: [u32; MAX_PLAYERS],
    pub street: u8,
    pub to_act: u8,
    pub current_bet: u32,
    pub min_raise: u32,
    pub folded: u8,
    pub allin: u8,
    pub last_aggressor: u8,
    pub players_to_act: u8,
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

    /// Current depth of the stack.
    pub fn depth(&self) -> usize {
        self.records.len()
    }

    /// Remove all records (does not release memory).
    pub fn clear(&mut self) {
        self.records.clear();
    }
}
