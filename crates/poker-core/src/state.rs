//! GameState: packed representation of 6-max NLHE state.
//!
//! Uses bitmasks for active players, folded players, and board cards.
//! `apply_action` and `undo_action` do not heap-allocate.

/// Packed game state for 6-max No-Limit Hold'em.
pub struct GameState {
    // TODO: implement packed fields (stacks, bets, board cards, street, active/folded masks)
}

impl GameState {
    /// Apply an action, pushing an undo record onto the undo stack.
    pub fn apply_action(&mut self, _action: crate::action::Action) {
        todo!()
    }

    /// Undo the last applied action.
    pub fn undo_action(&mut self) {
        todo!()
    }
}
