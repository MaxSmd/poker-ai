//! The `Game` and `CursorGame` implementations.
//!
//! `Game` is the clone-based reference path (used by the exact/validation
//! tooling); `CursorGame` is the zero-allocation hot path that walks one
//! `GameState` in place with `apply_action`/`undo_action` and carries an inline
//! perfect-recall history.  They are kept bit-identical — same action menus,
//! same raise bookkeeping, same info keys — and gated as such in the tests.

use poker_core::action::{Action, ActionList};

use super::{BlueprintCursor, BlueprintHoldem, BlueprintState, MAX_DEPTH};
use crate::games::Game;

impl Game for BlueprintHoldem {
    type State = BlueprintState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> BlueprintState {
        BlueprintState { gs: None, history: Vec::new(), street_raises: 0 }
    }

    fn is_terminal(&self, state: &BlueprintState) -> bool {
        state.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, state: &BlueprintState) -> bool {
        state.gs.is_none()
    }

    fn is_chance_enumerable(&self, _state: &BlueprintState) -> bool {
        false
    }

    fn utility(&self, state: &BlueprintState, player: usize) -> f64 {
        let gs = state.gs.as_ref().expect("utility at a play node");
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    /// Unsupported: the deal space is not enumerable.  The solver reaches
    /// children through [`sample_chance`](Game::sample_chance) instead.
    fn chance_outcomes(&self, _state: &BlueprintState) -> Vec<(BlueprintState, f64)> {
        unimplemented!("BlueprintHoldem chance is not enumerable; use sample_chance")
    }

    fn sample_chance(
        &self,
        _state: &BlueprintState,
        next_unit: impl FnMut() -> f64,
    ) -> BlueprintState {
        BlueprintState { gs: Some(self.deal(next_unit)), history: Vec::new(), street_raises: 0 }
    }

    fn current_player(&self, state: &BlueprintState) -> usize {
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, state: &BlueprintState) -> usize {
        let gs = state.gs.as_ref().expect("num_actions at a play node");
        self.capped_legal(gs, state.street_raises).len()
    }

    fn apply(&self, state: &BlueprintState, action: usize) -> BlueprintState {
        let gs = state.gs.as_ref().expect("apply at a play node");
        let act = self.capped_legal(gs, state.street_raises)[action];
        let (old_street, old_bet) = (gs.street, gs.current_bet);
        let mut next_gs = gs.clone();
        next_gs.apply_action(act);
        let street_raises = Self::next_raises(state.street_raises, old_street, old_bet, &next_gs);
        let mut history = state.history.clone();
        history.push(action as u8);
        BlueprintState { gs: Some(next_gs), history, street_raises }
    }

    fn info_key(&self, state: &BlueprintState) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        self.info_key_for(gs, &state.history)
    }
}


impl crate::games::CursorGame for BlueprintHoldem {
    type Cursor = BlueprintCursor;
    type Action = Action;
    type Actions = ActionList;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> BlueprintCursor {
        BlueprintCursor {
            gs: None,
            history: [0; MAX_DEPTH],
            depth: 0,
            street_raises: 0,
            raises_at: [0; MAX_DEPTH],
        }
    }

    fn is_terminal(&self, c: &BlueprintCursor) -> bool {
        c.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, c: &BlueprintCursor) -> bool {
        c.gs.is_none()
    }

    fn utility(&self, c: &BlueprintCursor, player: usize) -> f64 {
        let gs = c.gs.as_ref().expect("utility at a play node");
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    fn current_player(&self, c: &BlueprintCursor) -> usize {
        c.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn legal(&self, c: &BlueprintCursor) -> ActionList {
        self.capped_legal(c.gs.as_ref().expect("legal at a play node"), c.street_raises)
    }

    fn info_key(&self, c: &BlueprintCursor) -> u64 {
        let gs = c.gs.as_ref().expect("info_key at a play node");
        self.info_key_for(gs, &c.history[..c.depth])
    }

    fn apply(&self, c: &mut BlueprintCursor, a: usize, action: Action) {
        let gs = c.gs.as_mut().expect("apply at a play node");
        let (old_street, old_bet) = (gs.street, gs.current_bet);
        gs.apply_action(action);
        c.raises_at[c.depth] = c.street_raises;
        c.street_raises = Self::next_raises(c.street_raises, old_street, old_bet, gs);
        c.history[c.depth] = a as u8;
        c.depth += 1;
    }

    fn undo(&self, c: &mut BlueprintCursor) {
        c.depth -= 1;
        c.street_raises = c.raises_at[c.depth];
        c.gs.as_mut().expect("undo at a play node").undo_action();
    }

    fn sample_chance(&self, c: &mut BlueprintCursor, next_unit: impl FnMut() -> f64) {
        c.gs = Some(self.deal(next_unit));
        c.depth = 0;
        c.street_raises = 0;
    }

    fn undo_chance(&self, c: &mut BlueprintCursor) {
        c.gs = None;
        c.depth = 0;
        c.street_raises = 0;
    }
}
