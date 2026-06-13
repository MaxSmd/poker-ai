//! Heads-up No-Limit Hold'em bridge — the first time the solver meets the real
//! `poker-core` engine (Phase 1.5, the thin-bridge step).
//!
//! Full NLHE has an intractable chance space (every hole-card and board deal),
//! so a blueprint needs sampling plus card abstraction.  Before any of that,
//! this module proves the *wiring*: that the `Game` trait can drive
//! `poker_core::GameState` end-to-end — legal actions, action application,
//! terminal payoffs with real 7-card evaluation — and that CFR converges over
//! it.
//!
//! ## How it stays tractable *and* real
//!
//! Rather than a full deal, the game enumerates a small **curated set of
//! concrete deals**.  This keeps the chance space enumerable (so the validated
//! full-traversal solver and exact best response apply unchanged) while
//! preserving the essential feature of poker: **hidden information**.  A player
//! keys information sets on its *own* hole cards, so when several deals share a
//! player's hand but differ in the opponent's, that player genuinely cannot
//! tell them apart — exactly the uncertainty CFR must resolve.
//!
//! This is a real-mechanics integration target, not an abstraction: every bet
//! size, side-pot, and showdown comes from `poker-core`.  Replacing the curated
//! deal set with sampled deals plus bucketed keys is the next step (Phase 2).

use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};
use poker_core::{legal_actions, make_card};

use super::Game;
use crate::util::hash::fnv1a;

/// One concrete heads-up deal: both players' hole cards and the full board.
#[derive(Clone, Debug)]
pub struct Deal {
    pub holes: [[u8; 2]; 2],
    pub board: [u8; 5],
}

/// A heads-up NLHE game over a fixed set of equally-likely deals.
pub struct HeadsUpHoldem {
    deals: Vec<Deal>,
    stacks: [u32; MAX_PLAYERS],
    big_blind: u32,
    small_blind: u32,
    button: u8,
}

/// A node: the pre-deal chance root (`gs == None`) or a play node wrapping a
/// concrete `GameState` plus the perfect-recall action history.
#[derive(Clone, Debug)]
pub struct NlheState {
    gs: Option<GameState>,
    /// Legal-action indices taken so far — gives every info set a perfect-recall
    /// key without depending on `GameState`'s summarized betting fields.
    history: Vec<u8>,
}

impl HeadsUpHoldem {
    /// Construct from explicit deals and table parameters.
    pub fn new(deals: Vec<Deal>, stack: u32, big_blind: u32, small_blind: u32, button: u8) -> Self {
        let mut stacks = [0u32; MAX_PLAYERS];
        stacks[0] = stack;
        stacks[1] = stack;
        Self { deals, stacks, big_blind, small_blind, button }
    }

    /// A small demonstration game: each player holds one of two possible hands
    /// on a shared board, giving every player a genuine two-way uncertainty
    /// about the opponent.  Four equally-likely deals.
    pub fn demo() -> Self {
        // Board: A♣ K♦ 7♥ 2♠ 9♣  (ranks 12,11,5,0,7 / suits 0,1,2,3,0)
        let board = [
            make_card(12, 0),
            make_card(11, 1),
            make_card(5, 2),
            make_card(0, 3),
            make_card(7, 0),
        ];
        // Player 0 holds QQ or 88; player 1 holds JJ or 33 — none collide.
        let p0 = [[make_card(10, 0), make_card(10, 1)], [make_card(6, 2), make_card(6, 3)]];
        let p1 = [[make_card(9, 0), make_card(9, 1)], [make_card(1, 2), make_card(1, 3)]];
        let mut deals = Vec::new();
        for a in 0..2 {
            for b in 0..2 {
                deals.push(Deal { holes: [p0[a], p1[b]], board });
            }
        }
        Self::new(deals, 40, 10, 5, 0)
    }

    fn build_state(&self, deal: &Deal) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = deal.holes[0];
        holes[1] = deal.holes[1];
        GameState::new(2, self.big_blind, self.small_blind, self.stacks, holes, deal.board, self.button)
    }
}

impl Game for HeadsUpHoldem {
    type State = NlheState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> NlheState {
        NlheState { gs: None, history: Vec::new() }
    }

    fn is_terminal(&self, state: &NlheState) -> bool {
        state.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, state: &NlheState) -> bool {
        state.gs.is_none()
    }

    fn utility(&self, state: &NlheState, player: usize) -> f64 {
        let gs = state.gs.as_ref().expect("utility at a play node");
        // Chip delta relative to the starting stack, in big blinds.
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    fn chance_outcomes(&self, _state: &NlheState) -> Vec<(NlheState, f64)> {
        let p = 1.0 / self.deals.len() as f64;
        self.deals
            .iter()
            .map(|deal| (NlheState { gs: Some(self.build_state(deal)), history: Vec::new() }, p))
            .collect()
    }

    fn current_player(&self, state: &NlheState) -> usize {
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, state: &NlheState) -> usize {
        let gs = state.gs.as_ref().expect("num_actions at a play node");
        legal_actions(gs).len()
    }

    fn apply(&self, state: &NlheState, action: usize) -> NlheState {
        let gs = state.gs.as_ref().expect("apply at a play node");
        let act = legal_actions(gs)[action];
        let mut next_gs = gs.clone();
        next_gs.apply_action(act);
        let mut history = state.history.clone();
        history.push(action as u8);
        NlheState { gs: Some(next_gs), history }
    }

    fn info_key(&self, state: &NlheState) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        let player = gs.current_player();

        // Own hole cards (suit-order canonicalized) + the public board visible
        // on this street + the perfect-recall action history.
        let mut hole = gs.hole_cards[player];
        hole.sort_unstable();
        let visible = gs.board_cards_count();

        let mut bytes = Vec::with_capacity(8 + visible + state.history.len());
        bytes.push(player as u8);
        bytes.push(hole[0]);
        bytes.push(hole[1]);
        bytes.push(visible as u8);
        bytes.extend_from_slice(&gs.board[..visible]);
        bytes.push(0xFF); // separator so board/history can't blur together
        bytes.extend_from_slice(&state.history);
        fnv1a(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::best_response::exploitability;
    use crate::solver::cfr::{Cfr, Variant};
    use crate::solver::dcfr::Discount;
    use crate::solver::mccfr::Mccfr;

    #[test]
    fn chance_root_expands_to_all_deals() {
        let game = HeadsUpHoldem::demo();
        let root = game.root();
        assert!(game.is_chance(&root));
        let outs = game.chance_outcomes(&root);
        assert_eq!(outs.len(), 4, "demo game has 4 curated deals");
        for (st, p) in &outs {
            assert!((p - 0.25).abs() < 1e-12);
            assert!(!game.is_chance(st), "post-deal nodes are play nodes");
        }
    }

    #[test]
    fn payoffs_are_zero_sum_at_every_terminal() {
        // Walk the whole tree and check poker-core's payoffs conserve chips.
        let game = HeadsUpHoldem::demo();
        fn walk(g: &HeadsUpHoldem, s: &NlheState) {
            if g.is_terminal(s) {
                let u0 = g.utility(s, 0);
                let u1 = g.utility(s, 1);
                assert!((u0 + u1).abs() < 1e-9, "payoffs must sum to zero: {u0} + {u1}");
                return;
            }
            if g.is_chance(s) {
                for (c, _) in g.chance_outcomes(s) {
                    walk(g, &c);
                }
            } else {
                for a in 0..g.num_actions(s) {
                    walk(g, &g.apply(s, a));
                }
            }
        }
        walk(&game, &game.root());
    }

    #[test]
    fn cfr_produces_valid_strategy_over_real_mechanics() {
        // The wiring smoke test: a short CFR run must complete and yield a valid
        // probability distribution over poker-core's legal actions at every
        // discovered info set.
        let game = HeadsUpHoldem::demo();
        let mut solver = Cfr::new(game, Variant::Dcfr(Discount::RECOMMENDED));
        solver.train(200);
        assert!(solver.num_info_sets() > 0, "should discover info sets");
        for (_key, probs) in solver.average_strategy() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got sum {sum}");
            assert!(probs.iter().all(|&p| p >= 0.0));
        }
    }

    #[test]
    fn mccfr_runs_over_real_mechanics() {
        let game = HeadsUpHoldem::demo();
        let mut solver = Mccfr::new(game, Variant::Vanilla).with_baseline();
        solver.train(2_000);
        assert!(solver.num_info_sets() > 0);
    }

    /// CFR converges to equilibrium over the real betting/evaluation engine.
    /// Ignored (heavier); run with:
    ///   cargo test -p poker-ai --release -- --ignored nlhe
    #[test]
    #[ignore]
    fn cfr_converges_on_real_mechanics() {
        let game = HeadsUpHoldem::demo();
        let mut solver = Cfr::new(game, Variant::Vanilla);
        solver.train(50_000);
        let game = HeadsUpHoldem::demo();
        let expl = exploitability(&game, &solver.average_strategy());
        assert!(expl < 0.02, "exploitability {expl} bb should be small after convergence");
    }
}
