//! Tests for the packed game state: blinds, action flow, undo fidelity,
//! street advancement, and — the load-bearing ones — chip conservation and
//! side-pot correctness across all-in paths.

use super::*;
use crate::action::Action;
use crate::evaluator::make_card;

fn make_card_u8(rank: u8, suit: u8) -> u8 {
    make_card(rank, suit)
}

fn default_board() -> [u8; 5] {
    [
        make_card_u8(0, 0), // 2c
        make_card_u8(1, 1), // 3d
        make_card_u8(2, 2), // 4h
        make_card_u8(3, 3), // 5s
        make_card_u8(4, 0), // 6c
    ]
}

fn default_holes() -> [[u8; 2]; MAX_PLAYERS] {
    let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
    h[0] = [make_card_u8(12, 0), make_card_u8(12, 1)]; // AA
    h[1] = [make_card_u8(11, 0), make_card_u8(11, 1)]; // KK
    h[2] = [make_card_u8(10, 0), make_card_u8(10, 1)]; // QQ
    h[3] = [make_card_u8(9, 0), make_card_u8(9, 1)];   // JJ
    h[4] = [make_card_u8(8, 0), make_card_u8(8, 1)];   // TT
    h[5] = [make_card_u8(7, 0), make_card_u8(7, 1)];   // 99
    h
}

fn make_game(num_players: u8) -> GameState {
    let stacks = [1000u32; MAX_PLAYERS];
    GameState::new(num_players, 10, 5, stacks, default_holes(), default_board(), 0)
}

#[test]
fn blinds_posted_correctly() {
    let gs = make_game(6);
    // SB = position 1, BB = position 2 (button = 0)
    assert_eq!(gs.street_bets[1], 5);  // SB
    assert_eq!(gs.street_bets[2], 10); // BB
    assert_eq!(gs.current_bet, 10);
    assert_eq!(gs.street, 0);
}

#[test]
fn fold_reduces_non_folded() {
    let mut gs = make_game(6);
    let before = gs.count_non_folded();
    gs.apply_action(Action::Fold);
    assert_eq!(gs.count_non_folded(), before - 1);
}

#[test]
fn undo_restores_state() {
    let mut gs = make_game(6);
    let to_act_before = gs.to_act;
    let stacks_before = gs.stacks;
    let folded_before = gs.folded;

    gs.apply_action(Action::Fold);
    assert_ne!(gs.folded, folded_before);

    gs.undo_action();
    assert_eq!(gs.to_act, to_act_before);
    assert_eq!(gs.stacks, stacks_before);
    assert_eq!(gs.folded, folded_before);
}

#[test]
fn call_moves_chips() {
    let mut gs = make_game(6);
    let p = gs.to_act as usize;
    let stack_before = gs.stacks[p];
    gs.apply_action(Action::Call);
    // Player called the BB (10 chips), so stack decreased by 10.
    assert_eq!(gs.stacks[p], stack_before - 10);
}

#[test]
fn terminal_after_all_fold() {
    let mut gs = make_game(2);
    assert!(!gs.is_terminal());
    gs.apply_action(Action::Fold);
    assert!(gs.is_terminal());
}

#[test]
fn payoff_last_player_wins_pot() {
    let mut gs = make_game(2);
    // Heads-up: both post (SB+BB=15), then SB folds immediately.
    gs.apply_action(Action::Fold);
    assert!(gs.is_terminal());
    let payoffs = gs.terminal_payoffs();
    let total: i32 = payoffs.iter().sum();
    assert_eq!(total, 0, "payoffs must sum to zero");
    // Winner should have positive payoff.
    assert!(payoffs.iter().any(|&p| p > 0));
}

#[test]
fn street_advances_after_round() {
    let mut gs = make_game(2);
    assert_eq!(gs.street, 0);
    // Heads-up preflop: button (player 0, SB) acts first.
    // Both call/check to close the street.
    gs.apply_action(Action::Call);  // SB (button, player 0) calls
    gs.apply_action(Action::Check); // BB (player 1) checks
    // Street should now be flop (1).
    assert_eq!(gs.street, 1);
}

#[test]
fn raise_resets_players_to_act() {
    let mut gs = make_game(6);
    // UTG (first to act preflop) raises using an abstract action.
    let actions = crate::action::legal_actions(&gs);
    let raise_action = actions.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
    gs.apply_action(*raise_action);
    // All active players except the raiser now need to act.
    // players_to_act >= 1 (there are opponents).
    assert!(gs.players_to_act >= 1);
}

// ── Chip conservation helper ─────────────────────────────────────────────

fn chip_total(gs: &GameState) -> u32 {
    gs.stacks.iter().sum::<u32>() + gs.total_committed.iter().sum::<u32>()
}

// ── Full-hand integration test ───────────────────────────────────────────

/// Play a complete hand (preflop → flop → turn → river → showdown) and
/// assert that chips are conserved at every step and payoffs sum to zero.
#[test]
fn full_hand_chip_conservation() {
    let mut gs = make_game(2);
    let initial_chips = chip_total(&gs);

    // Preflop: SB (button=0) calls, BB checks.
    gs.apply_action(Action::Call);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after preflop call");
    gs.apply_action(Action::Check);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after preflop check");
    assert_eq!(gs.street, 1, "should be on flop");

    // Flop: check, check.
    gs.apply_action(Action::Check);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after flop check 1");
    gs.apply_action(Action::Check);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after flop check 2");
    assert_eq!(gs.street, 2, "should be on turn");

    // Turn: check, check.
    gs.apply_action(Action::Check);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after turn check 1");
    gs.apply_action(Action::Check);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after turn check 2");
    assert_eq!(gs.street, 3, "should be on river");

    // River: check, check → showdown.
    gs.apply_action(Action::Check);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after river check 1");
    gs.apply_action(Action::Check);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after river check 2");
    assert!(gs.is_terminal(), "should be terminal after river");

    let payoffs = gs.terminal_payoffs();
    assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");
}

// ── Adversarial edge cases ────────────────────────────────────────────────

/// Everyone goes all-in preflop — hand must terminate correctly with chips conserved.
#[test]
fn all_in_preflop_two_players() {
    let mut gs = make_game(2);
    let initial_chips = chip_total(&gs);

    gs.apply_action(Action::AllIn);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after p0 all-in");
    gs.apply_action(Action::AllIn);
    assert_eq!(chip_total(&gs), initial_chips, "conservation after p1 all-in");
    assert!(gs.is_terminal(), "hand should be terminal when both all-in");

    let payoffs = gs.terminal_payoffs();
    assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");
}

/// 3-way all-in with different stack sizes — verify side pots and chip conservation.
#[test]
fn three_way_allin_different_stacks() {
    // Player 0: button (stack 100), Player 1: SB (stack 200), Player 2: BB (stack 300).
    let mut stacks = [0u32; MAX_PLAYERS];
    stacks[0] = 100;
    stacks[1] = 200;
    stacks[2] = 300;
    let holes = default_holes();
    let board = default_board();
    let big_blind = 10u32;
    let mut gs = GameState::new(3, big_blind, big_blind / 2, stacks, holes, board, 0);
    let initial_chips = chip_total(&gs);

    // Drive everyone all-in: UTG (player 3%3=0 is button, so UTG is player (0+3)%3=0)
    // Actually button=0, SB=(0+1)%3=1, BB=(0+2)%3=2, UTG=(0+3)%3=0=button. Wait, that
    // wraps to 0. Let me think again. n=3, button=0, SB=1, BB=2, UTG=(0+3)%3=0.
    // So UTG is the button position (0). First to act preflop is UTG=player 0.
    gs.apply_action(Action::AllIn); // player 0 all-in (100 chips)
    assert_eq!(chip_total(&gs), initial_chips);
    gs.apply_action(Action::AllIn); // player 1 all-in (200 chips)
    assert_eq!(chip_total(&gs), initial_chips);
    gs.apply_action(Action::AllIn); // player 2 all-in (300 chips)
    assert_eq!(chip_total(&gs), initial_chips);
    assert!(gs.is_terminal(), "should be terminal when all players all-in");

    let payoffs = gs.terminal_payoffs();
    assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");
    // Total chips redistributed must equal total initial chips.
    let total_returned: i32 = payoffs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let committed = gs.total_committed[i] as i32;
            committed + p
        })
        .sum();
    assert_eq!(total_returned as u32, initial_chips, "all chips must be returned");
}

/// Min-raise then re-raise — min_raise must track the largest raise increment.
#[test]
fn min_raise_then_reraise() {
    let mut gs = make_game(2);
    // Preflop HU: button (p0, SB) is first to act.
    // Pick the first abstract raise available.
    let actions = crate::action::legal_actions(&gs);
    let first_raise = *actions.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
    gs.apply_action(first_raise);
    let first_bet = gs.current_bet;
    let mr_after_first = gs.min_raise;
    assert!(mr_after_first >= 10, "min_raise should be >= BB after first raise");

    // BB re-raises using a legal abstract action.
    let actions2 = crate::action::legal_actions(&gs);
    let reraise = *actions2.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
    gs.apply_action(reraise);
    assert!(gs.current_bet > first_bet, "current_bet should increase on re-raise");
    assert!(gs.min_raise >= mr_after_first, "min_raise should not decrease on re-raise");
}

/// Odd-chip allocation: when a side pot doesn't divide evenly among winners,
/// the remainder goes to the winner seated closest to the button's left,
/// matching standard casino rules (Robert's Rules of Poker §15).
///
/// Setup: 3 players, button=0.  SB=p1 posts 5, BB=p2 posts 10.  p0 (UTG)
/// goes all-in for 30.  p1 folds (5 committed).  p2 goes all-in for 20 more.
///
/// Committed: p0=30, p1=5 (folded), p2=30.  Total = 65.
///
/// Both p0 and p2 hold an identical board hand (royal flush on board) → tied.
///
/// Side pot tiers:
///   Tier 1 (level 5):  3 contributors × 5 = 15.  Eligible: p0, p2.
///                       15 / 2 = 7 remainder 1.  ← odd-chip path exercised
///                       First winner left of button (button=0): p2 (offset 2).
///                       p2 gets 8, p0 gets 7.
///   Tier 2 (level 30): 2 contributors × 25 = 50.  50 / 2 = 25 each.
///
/// Expected payoffs: p0=2, p1=−5, p2=3.
#[test]
fn odd_chip_goes_to_first_winner_left_of_button() {
    let mut stacks = [0u32; MAX_PLAYERS];
    stacks[0] = 30; // UTG / button
    stacks[1] = 6;  // SB — posts 5, keeps 1
    stacks[2] = 30; // BB

    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    // Hole cards are low clubs; royal-flush board gives everyone the same hand.
    holes[0] = [make_card_u8(0, 0), make_card_u8(1, 0)]; // 2c 3c
    holes[1] = [make_card_u8(2, 0), make_card_u8(3, 0)]; // 4c 5c
    holes[2] = [make_card_u8(4, 0), make_card_u8(5, 0)]; // 6c 7c

    let board = [
        make_card_u8(12, 3), // As
        make_card_u8(11, 3), // Ks
        make_card_u8(10, 3), // Qs
        make_card_u8(9, 3),  // Js
        make_card_u8(8, 3),  // Ts  → royal flush on board; everyone ties
    ];

    // button=0, SB=p1, BB=p2, UTG=p0 (acts first preflop).
    let mut gs = GameState::new(3, 10, 5, stacks, holes, board, 0);

    gs.apply_action(Action::AllIn); // p0: commits 30, current_bet=30
    // p1: to_call=25, stacks=1 < 25 → only Fold or AllIn available.
    gs.apply_action(Action::Fold);  // p1 folds; 5 committed stays in pot
    // p2: to_call=20, stacks=20, stacks NOT > to_call → AllIn only.
    gs.apply_action(Action::AllIn); // p2: commits 20 more, total=30

    assert!(gs.is_terminal(), "should be terminal (remaining players all-in)");
    assert_eq!(gs.total_committed[0], 30);
    assert_eq!(gs.total_committed[1],  5);
    assert_eq!(gs.total_committed[2], 30);

    let payoffs = gs.terminal_payoffs();
    assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");

    // p2 is first winner left of button=0 (offset 2 before p0 at offset 3).
    assert_eq!(payoffs[0],  2, "p0: 7+25−30");
    assert_eq!(payoffs[1], -5, "p1 folded, loses blind");
    assert_eq!(payoffs[2],  3, "p2 gets odd chip: 8+25−30");
}

/// BB special case: everyone limps preflop, BB gets the option to raise.
#[test]
fn bb_gets_option_when_everyone_limps() {
    let mut gs = make_game(3);
    // button=0, SB=1, BB=2, UTG=0
    // UTG (player 0) calls (limps).
    gs.apply_action(Action::Call);
    // SB (player 1) calls (limps, tops up to BB).
    gs.apply_action(Action::Call);
    // Now BB (player 2) should still have the option — game is NOT terminal and
    // it is BB's turn.  BB checks.
    assert!(!gs.is_terminal(), "game should not be terminal before BB acts");
    assert_eq!(gs.to_act, 2, "BB (player 2) should be next to act");
    gs.apply_action(Action::Check); // BB exercises option by checking.
    assert_eq!(gs.street, 1, "street should advance to flop after BB checks");
}
