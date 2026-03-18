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

/// Returns the legal abstract actions available for the current player in `state`.
///
/// Uses the blueprint action abstraction from [`betting`]:
/// - Fold / Check / Call are always considered when applicable.
/// - Raise sizes are drawn from the abstract pot-fraction sizes for the street.
/// - AllIn is always appended when it represents a bet or raise above the
///   current bet and the player has chips remaining.
///
/// The returned `Vec` is pre-allocated with capacity 8 to avoid reallocations
/// in typical cases.
pub fn legal_actions(state: &GameState) -> Vec<Action> {
    let mut actions: Vec<Action> = Vec::with_capacity(8);

    if state.is_terminal() {
        return actions;
    }

    let p = state.to_act as usize;
    let to_call = state.current_bet.saturating_sub(state.street_bets[p]);
    // Maximum total street-bet this player can make (all chips in).
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
    // Minimum legal raise total: current_bet + min_raise.
    let min_raise_total = state.current_bet + state.min_raise;

    if max_bet <= state.current_bet {
        // Player can't raise (no chips above the call amount).
        return actions;
    }

    let pot = state.pot();
    let (abstract_bets, n) = abstract_raise_amounts(pot, state.current_bet, state.street, state.big_blind);

    let mut allin_added = false;

    for &bet_level in &abstract_bets[..n] {
        if bet_level < min_raise_total {
            // Below minimum raise — skip.
            continue;
        }
        if bet_level >= max_bet {
            // Would require going all-in or beyond.
            if !allin_added {
                actions.push(Action::AllIn);
                allin_added = true;
            }
            break;
        }
        actions.push(Action::Raise(bet_level));
    }

    // All-in is always a valid aggressive action if it's at least a call/min-raise.
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
        for i in 0..num_players as usize {
            holes[i] = [make_card(i as u8, 0), make_card(i as u8 + 1, 1)];
        }
        let board = [
            make_card(2, 0),
            make_card(3, 1),
            make_card(4, 2),
            make_card(5, 3),
            make_card(6, 0),
        ];
        GameState::new(num_players, 10, stacks, holes, board, 0)
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
        // HU: SB calls, then BB has no bet to face.
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
