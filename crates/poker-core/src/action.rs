//! Action enum and legal action generation.
//!
//! [`Action::Raise`] carries the **total street-bet level** the player is
//! moving to, *not* an increment.  For example, if `current_bet` is 20 and
//! the player raises to 60, the action is `Raise(60)`.
//!
//! Abstract bet sizes follow the blueprint action abstraction defined in
//! `betting.rs`.  `AllIn` is always legal when a player has chips remaining
//! and the all-in amount is at least a legal action (call or raise).

use crate::betting::abstract_raise_amounts;
use crate::state::GameState;

/// A player action in No-Limit Hold'em.
///
/// # Notes on `Call` vs `AllIn`
///
/// When the required call amount equals or exceeds the player's remaining
/// stack, `legal_actions` emits `AllIn` rather than `Call`.  Both actions
/// commit the same chips in that scenario, but `AllIn` is the canonical
/// representation so that downstream code (CFR, abstraction) can distinguish
/// a *committing* call from a voluntary over-bet without additional checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Fold,
    Check,
    Call,
    /// Raise *to* this total street-bet level (not a raise-by amount).
    Raise(u32),
    AllIn,
}

/// Stack-allocated list of legal actions — avoids heap allocation in the CFR
/// hot path.  Supports up to 8 actions (fold/check/call + up to 4 abstract
/// raises + all-in), which is sufficient for any configured bet abstraction.
///
/// Derefs to `&[Action]`, so all slice methods (`contains`, `iter`, `len`,
/// `is_empty`, …) work without additional code.
#[derive(Clone, Copy, Debug)]
pub struct ActionList {
    buf: [Action; 8],
    len: usize,
}

impl ActionList {
    fn new() -> Self {
        Self { buf: [Action::Fold; 8], len: 0 }
    }

    fn push(&mut self, a: Action) {
        debug_assert!(self.len < 8, "ActionList capacity exceeded");
        self.buf[self.len] = a;
        self.len += 1;
    }

    /// Build a list from a slice of actions — for abstraction layers that filter
    /// the engine's legal set (e.g. a betting-abstraction raise cap). Panics in
    /// debug if given more than the 8-action capacity.
    pub fn from_actions(actions: &[Action]) -> Self {
        let mut l = Self::new();
        for &a in actions {
            l.push(a);
        }
        l
    }
}

impl std::ops::Deref for ActionList {
    type Target = [Action];
    #[inline]
    fn deref(&self) -> &[Action] {
        &self.buf[..self.len]
    }
}

impl AsRef<[Action]> for ActionList {
    #[inline]
    fn as_ref(&self) -> &[Action] {
        self
    }
}

/// Returns the legal abstract actions available for the current player in `state`.
///
/// Uses the blueprint action abstraction from [`betting`]:
/// - Fold / Check / Call are always considered when applicable.
/// - Raise sizes are drawn from the abstract pot-fraction sizes for the street.
/// - AllIn is always appended when it represents a bet or raise above the
///   current bet and the player has chips remaining.
///
/// Returns a stack-allocated [`ActionList`] — no heap allocation occurs.
#[inline]
pub fn legal_actions(state: &GameState) -> ActionList {
    let mut actions = ActionList::new();

    if state.is_terminal() {
        return actions;
    }

    let p = state.to_act as usize;
    let to_call = state.current_bet.saturating_sub(state.street_bets[p]);
    // Maximum total street-bet this player can reach (all chips in).
    let max_bet = state.stacks[p] + state.street_bets[p];

    // --- passive options ---
    if to_call == 0 {
        actions.push(Action::Check);
    } else {
        actions.push(Action::Fold);
        if state.stacks[p] > to_call {
            actions.push(Action::Call);
        } else {
            // Call is effectively all-in — only offer AllIn, not Call.
            actions.push(Action::AllIn);
            return actions;
        }
    }

    // --- aggressive options ---
    // At this point max_bet > current_bet is guaranteed: if to_call == 0
    // the player has chips (active players always do); if to_call > 0 we
    // already returned when stacks[p] <= to_call, so stacks[p] > to_call
    // implies max_bet = stacks[p] + street_bets[p] > current_bet.
    let min_raise_total = state.current_bet + state.min_raise;
    let pot = state.pot;
    let (abstract_bets, n) =
        abstract_raise_amounts(pot, state.current_bet, state.min_raise, state.street);

    let mut allin_added = false;

    for &bet_level in &abstract_bets[..n] {
        if bet_level < min_raise_total {
            continue;
        }
        if bet_level >= max_bet {
            if !allin_added {
                actions.push(Action::AllIn);
                allin_added = true;
            }
            break;
        }
        actions.push(Action::Raise(bet_level));
    }

    if !allin_added && max_bet >= min_raise_total {
        actions.push(Action::AllIn);
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::make_card;
    use crate::state::{GameState, MAX_PLAYERS, NO_CARD};

    fn make_game(num_players: u8) -> GameState {
        let stacks = [1000u32; MAX_PLAYERS];
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        for (i, h) in holes.iter_mut().take(num_players as usize).enumerate() {
            *h = [make_card(i as u8, 0), make_card(i as u8, 1)];
        }
        let board = [
            make_card(8, 2),
            make_card(9, 2),
            make_card(10, 2),
            make_card(11, 2),
            make_card(12, 2),
        ];
        GameState::new(num_players, 10, 5, stacks, holes, board, 0)
    }

    #[test]
    fn preflop_utg_has_fold_call_raise() {
        let gs = make_game(6);
        let actions = legal_actions(&gs);
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(actions.iter().any(|a| matches!(a, Action::Raise(_))));
    }

    #[test]
    fn check_available_when_no_bet() {
        let mut gs = make_game(2);
        gs.apply_action(Action::Call);
        let actions = legal_actions(&gs);
        assert!(actions.contains(&Action::Check));
        assert!(!actions.contains(&Action::Fold));
    }

    #[test]
    fn allin_always_present_when_chips_remain() {
        let gs = make_game(6);
        let actions = legal_actions(&gs);
        assert!(actions.contains(&Action::AllIn));
    }

    #[test]
    fn no_actions_at_terminal() {
        let mut gs = make_game(2);
        gs.apply_action(Action::Fold);
        assert!(gs.is_terminal());
        assert!(legal_actions(&gs).is_empty());
    }
}
