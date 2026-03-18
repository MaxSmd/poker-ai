//! Undo stack for zero-allocation tree traversal.
//!
//! Records the minimal delta needed to reverse each `apply_action` call.

/// A record of what changed when an action was applied.
pub struct UndoRecord {
    // TODO: fields to reverse a single apply_action
}

/// Fixed-capacity undo stack.
pub struct UndoStack {
    records: Vec<UndoRecord>,
}

impl UndoStack {
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    pub fn push(&mut self, record: UndoRecord) {
        self.records.push(record);
    }

    pub fn pop(&mut self) -> Option<UndoRecord> {
        self.records.pop()
    }
}
