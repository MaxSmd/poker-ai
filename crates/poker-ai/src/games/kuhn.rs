//! Kuhn Poker — the smallest non-trivial poker game, used as the first
//! correctness oracle for the solver.
//!
//! ## Rules
//!
//! Three cards: Jack, Queen, King (encoded `0 < 1 < 2`).  Two players each ante
//! 1 chip and are dealt one private card.  Player 0 acts first.  Each decision
//! is binary:
//!
//! * **action 0** = check (when facing no bet) / fold (when facing a bet)
//! * **action 1** = bet (when facing no bet) / call (when facing a bet)
//!
//! A single bet/raise of 1 chip is allowed.  The betting histories are:
//!
//! ```text
//!   (empty)         P0 to act
//!   [check]         P1 to act
//!   [bet]           P1 to act
//!   [check,check]   showdown, pot 2
//!   [check,bet]     P0 to act
//!   [check,bet,fold] P1 wins P0's ante
//!   [check,bet,call] showdown, pot 4
//!   [bet,fold]      P0 wins P1's ante
//!   [bet,call]      showdown, pot 4
//! ```
//!
//! ## Known solution
//!
//! Kuhn Poker has a known family of Nash equilibria with game value
//! **−1/18 ≈ −0.0556** to the first player.  At equilibrium the exploitability
//! (NashConv / 2) is zero, so a correct solver must drive it below any small ε.
//! This is validation protocol step 1 in the plan.

use super::Game;

/// Action index for check / fold.
const CHECK_FOLD: usize = 0;
/// Action index for bet / call.
const BET_CALL: usize = 1;

/// The exact game value to player 0 under optimal play.
pub const GAME_VALUE_P0: f64 = -1.0 / 18.0;

/// A Kuhn Poker node.
///
/// Before the deal (`dealt == false`) the state is the single chance node.
/// After the deal, `cards[p]` holds player `p`'s card and `history` records the
/// actions taken so far.
#[derive(Clone, Debug)]
pub struct KuhnState {
    dealt: bool,
    cards: [u8; 2],
    /// Actions taken so far (each `CHECK_FOLD` or `BET_CALL`).
    history: Vec<u8>,
}

/// The Kuhn Poker game.
pub struct Kuhn;

impl KuhnState {
    /// True at one of the showdown / fold terminal histories.
    fn is_terminal_history(&self) -> bool {
        matches!(
            self.history.as_slice(),
            [0, 0] | [0, 1, 0] | [0, 1, 1] | [1, 0] | [1, 1]
        )
    }

    /// Player 0's utility at a terminal history (player 1's is the negation).
    fn terminal_utility_p0(&self) -> f64 {
        let p0_wins = self.cards[0] > self.cards[1];
        match self.history.as_slice() {
            // Both checked: showdown for the antes (pot 2, net ±1).
            [0, 0] => if p0_wins { 1.0 } else { -1.0 },
            // P0 checked, P1 bet, P0 folded: P1 takes P0's ante.
            [0, 1, 0] => -1.0,
            // P0 checked, P1 bet, P0 called: showdown for pot 4 (net ±2).
            [0, 1, 1] => if p0_wins { 2.0 } else { -2.0 },
            // P0 bet, P1 folded: P0 takes P1's ante.
            [1, 0] => 1.0,
            // P0 bet, P1 called: showdown for pot 4 (net ±2).
            [1, 1] => if p0_wins { 2.0 } else { -2.0 },
            _ => unreachable!("terminal_utility_p0 called on non-terminal history"),
        }
    }
}

impl Game for Kuhn {
    type State = KuhnState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> KuhnState {
        KuhnState { dealt: false, cards: [0, 0], history: Vec::new() }
    }

    fn is_terminal(&self, state: &KuhnState) -> bool {
        state.dealt && state.is_terminal_history()
    }

    fn is_chance(&self, state: &KuhnState) -> bool {
        !state.dealt
    }

    fn utility(&self, state: &KuhnState, player: usize) -> f64 {
        let u0 = state.terminal_utility_p0();
        if player == 0 { u0 } else { -u0 }
    }

    fn chance_outcomes(&self, _state: &KuhnState) -> Vec<(KuhnState, f64)> {
        // All 6 ordered deals of two distinct cards from {0,1,2}, uniform.
        let mut out = Vec::with_capacity(6);
        for a in 0u8..3 {
            for b in 0u8..3 {
                if a != b {
                    out.push((
                        KuhnState { dealt: true, cards: [a, b], history: Vec::new() },
                        1.0 / 6.0,
                    ));
                }
            }
        }
        out
    }

    fn current_player(&self, state: &KuhnState) -> usize {
        // Even-length histories are P0's turn, odd are P1's.  The only length-2
        // decision node is [check, bet], which is correctly P0 again.
        state.history.len() % 2
    }

    fn num_actions(&self, _state: &KuhnState) -> usize {
        2
    }

    fn apply(&self, state: &KuhnState, action: usize) -> KuhnState {
        debug_assert!(action == CHECK_FOLD || action == BET_CALL);
        let mut next = state.clone();
        next.history.push(action as u8);
        next
    }

    fn info_key(&self, state: &KuhnState) -> u64 {
        // Unique per (acting player's card, public history).  The history bits
        // already encode whose turn it is, so the two players never collide.
        let player = self.current_player(state);
        let card = state.cards[player] as u64;
        // Encode history as a base-3 number with a leading 1 sentinel so that
        // distinct lengths (e.g. [] vs [check]) map to distinct codes.
        let mut hist_code: u64 = 1;
        for &a in &state.history {
            hist_code = hist_code * 3 + (a as u64 + 1);
        }
        (card << 32) | hist_code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::best_response::{exploitability, profile_value};
    use crate::solver::cfr::{Cfr, Variant};
    use crate::solver::dcfr::Discount;

    #[test]
    fn tree_has_12_info_sets() {
        // 2 decision points per player × 3 cards × 2 players = 12 info sets.
        let mut solver = Cfr::new(Kuhn, Variant::Vanilla);
        solver.train(1);
        assert_eq!(solver.num_info_sets(), 12, "Kuhn has exactly 12 information sets");
    }

    #[test]
    fn vanilla_cfr_converges_to_equilibrium() {
        // Vanilla CFR converges at O(1/√T), so it needs more iterations than
        // DCFR to cross the same exploitability threshold — that slower rate is
        // precisely why the plan adopts DCFR for the blueprint.
        let mut solver = Cfr::new(Kuhn, Variant::Vanilla);
        solver.train(100_000);
        let avg = solver.average_strategy();

        let expl = exploitability(&Kuhn, &avg);
        assert!(expl < 1e-3, "vanilla CFR exploitability {expl} should be < 1e-3");

        let value = profile_value(&Kuhn, &avg, 0);
        assert!(
            (value - GAME_VALUE_P0).abs() < 5e-3,
            "game value {value} should be near -1/18 = {GAME_VALUE_P0}"
        );
    }

    #[test]
    fn dcfr_converges_to_equilibrium() {
        let mut solver = Cfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED));
        solver.train(20_000);
        let avg = solver.average_strategy();

        let expl = exploitability(&Kuhn, &avg);
        assert!(expl < 1e-3, "DCFR exploitability {expl} should be < 1e-3");

        let value = profile_value(&Kuhn, &avg, 0);
        assert!(
            (value - GAME_VALUE_P0).abs() < 5e-3,
            "game value {value} should be near -1/18 = {GAME_VALUE_P0}"
        );
    }

    #[test]
    fn dcfr_has_better_last_iterate_than_vanilla() {
        // DCFR's signature property is *last-iterate* convergence: its current
        // strategy approaches equilibrium, whereas vanilla CFR's last iterate
        // keeps oscillating and only its time-average converges.  (DCFR's
        // average-strategy advantage is a variance-suppression effect that only
        // appears under sampling, not in this full-traversal setting.)
        let iters = 2_000;

        let mut vanilla = Cfr::new(Kuhn, Variant::Vanilla);
        vanilla.train(iters);
        let last_vanilla = exploitability(&Kuhn, &vanilla.current_strategy());

        let mut dcfr = Cfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED));
        dcfr.train(iters);
        let last_dcfr = exploitability(&Kuhn, &dcfr.current_strategy());

        assert!(
            last_dcfr < last_vanilla,
            "DCFR last iterate ({last_dcfr}) should beat vanilla last iterate ({last_vanilla})"
        );
    }
}
