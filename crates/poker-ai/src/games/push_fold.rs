//! Heads-up push/fold NLHE — the first *converging* blueprint over real
//! mechanics (Phase 1.5).
//!
//! [`super::blueprint::BlueprintHoldem`] is the full game, but it cannot
//! converge locally: without a complete postflop card abstraction (the
//! cloud-scale equity precompute), every randomly-dealt board mints fresh,
//! once-visited postflop information sets and the tree never plateaus.  Push/fold
//! removes the problem at the root: at every decision the only choices are
//! **fold** or **commit all-in**.  The flat-call / limp line — the sole source
//! of postflop play — is gone, so the tree is two levels deep at *any* stack
//! depth:
//!
//! ```text
//!   SB:  fold  | shove
//!   BB (vs shove):  fold | call
//! ```
//!
//! That is ~169 + 169 information sets (one decision per suit-canonical starting
//! hand per player), it plateaus, and it has a well-known Nash solution to
//! validate against — exactly the plan's "prove it on a known-solution game
//! first" discipline, now over the real `poker-core` engine: the all-in runout
//! and showdown come from the real evaluator, and payoffs are real chip deltas.
//!
//! Chance (the full deal) is sampled, not enumerated, so the game reuses the
//! same [`Game::sample_chance`] path the blueprint introduced.

use poker_core::action::Action;
use poker_core::legal_actions;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use super::nlhe::fnv1a;
use super::Game;
use crate::abstraction::canonical::canonical_key;

/// Cards consumed by a heads-up deal: 2 hole cards each + 5 board.
const DEAL_CARDS: usize = 9;

/// A heads-up push/fold NLHE game with sampled deals.
pub struct PushFoldHoldem {
    stacks: [u32; MAX_PLAYERS],
    big_blind: u32,
    small_blind: u32,
    button: u8,
}

/// A node: the pre-deal chance root (`gs == None`) or a play node.
#[derive(Clone, Debug)]
pub struct PushFoldState {
    gs: Option<GameState>,
    /// Perfect-recall action history (`0 = fold, 1 = commit`).
    history: Vec<u8>,
}

impl PushFoldHoldem {
    /// A game with equal starting stacks (`stack` chips each).  A realistic
    /// short-stack scenario uses, e.g., `stack = 25 * big_blind`.
    pub fn new(stack: u32, big_blind: u32, small_blind: u32, button: u8) -> Self {
        let mut stacks = [0u32; MAX_PLAYERS];
        stacks[0] = stack;
        stacks[1] = stack;
        Self { stacks, big_blind, small_blind, button }
    }

    /// Deal both hands + the full board from a freshly shuffled deck.
    fn deal(&self, mut next_unit: impl FnMut() -> f64) -> GameState {
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        let last = 51;
        for i in 0..DEAL_CARDS {
            let span = 52 - i;
            let j = (i + (next_unit() * span as f64) as usize).min(last);
            deck.swap(i, j);
        }
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [deck[0], deck[1]];
        holes[1] = [deck[2], deck[3]];
        let board = [deck[4], deck[5], deck[6], deck[7], deck[8]];
        GameState::new(2, self.big_blind, self.small_blind, self.stacks, holes, board, self.button)
    }

    /// The two-action push/fold menu at a decision node: `[Fold, commit]`, where
    /// *commit* is `AllIn` if available, else `Call` (calling a shove is itself
    /// all-in).  Restricting to this menu is what removes postflop play.
    fn menu(gs: &GameState) -> [Action; 2] {
        let acts = legal_actions(gs);
        let mut commit = None;
        let mut has_fold = false;
        for &a in acts.iter() {
            match a {
                Action::Fold => has_fold = true,
                Action::AllIn => commit = Some(Action::AllIn),
                Action::Call if commit.is_none() => commit = Some(Action::Call),
                _ => {}
            }
        }
        debug_assert!(has_fold, "push/fold decision nodes always allow folding");
        [Action::Fold, commit.expect("a chips-committing action is always available")]
    }
}

impl Game for PushFoldHoldem {
    type State = PushFoldState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> PushFoldState {
        PushFoldState { gs: None, history: Vec::new() }
    }

    fn is_terminal(&self, state: &PushFoldState) -> bool {
        state.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, state: &PushFoldState) -> bool {
        state.gs.is_none()
    }

    fn is_chance_enumerable(&self, _state: &PushFoldState) -> bool {
        false
    }

    fn utility(&self, state: &PushFoldState, player: usize) -> f64 {
        let gs = state.gs.as_ref().expect("utility at a play node");
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    fn chance_outcomes(&self, _state: &PushFoldState) -> Vec<(PushFoldState, f64)> {
        unimplemented!("PushFoldHoldem chance is not enumerable; use sample_chance")
    }

    fn sample_chance(
        &self,
        _state: &PushFoldState,
        next_unit: impl FnMut() -> f64,
    ) -> PushFoldState {
        PushFoldState { gs: Some(self.deal(next_unit)), history: Vec::new() }
    }

    fn current_player(&self, state: &PushFoldState) -> usize {
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, _state: &PushFoldState) -> usize {
        2
    }

    fn apply(&self, state: &PushFoldState, action: usize) -> PushFoldState {
        let gs = state.gs.as_ref().expect("apply at a play node");
        let act = Self::menu(gs)[action];
        let mut next_gs = gs.clone();
        next_gs.apply_action(act);
        let mut history = state.history.clone();
        history.push(action as u8);
        PushFoldState { gs: Some(next_gs), history }
    }

    fn info_key(&self, state: &PushFoldState) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        let player = gs.current_player();
        let mut hole = gs.hole_cards[player];
        hole.sort_unstable();
        // Pre-flop only: the 169 suit-canonical starting-hand classes, plus the
        // perfect-recall history (distinguishes SB-open from BB-vs-shove).
        let class = canonical_key(&hole, &[]);
        let mut bytes = Vec::with_capacity(4 + class.len() + state.history.len());
        bytes.push(player as u8);
        bytes.extend_from_slice(&class);
        bytes.push(0xFF);
        bytes.extend_from_slice(&state.history);
        fnv1a(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::cfr::Variant;
    use crate::solver::dcfr::Discount;
    use crate::solver::mccfr::Mccfr;
    use poker_core::make_card;
    use std::collections::HashMap;

    #[test]
    fn tree_is_two_levels_and_bounded() {
        // The defining property: info-set count plateaus (no postflop).
        let game = PushFoldHoldem::new(50, 2, 1, 0);
        let mut s = Mccfr::with_seed(game, Variant::Vanilla, 1);
        s.train(40_000);
        let a = s.num_info_sets();
        s.train(60_000);
        let b = s.num_info_sets();
        assert_eq!(a, b, "info-set count must plateau ({a} -> {b})");
        // Two players × 169 classes, minus hands never reached; comfortably < 400.
        assert!(b <= 400, "push/fold has ~338 info sets, got {b}");
        assert!(b > 100, "should discover most starting-hand classes, got {b}");
    }

    #[test]
    fn payoffs_are_zero_sum() {
        // Drive a few sampled hands to terminal and check chips are conserved.
        let game = PushFoldHoldem::new(50, 2, 1, 0);
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..50 {
            let mut st = game.sample_chance(&game.root(), &mut next);
            while !game.is_terminal(&st) {
                // Always take the "commit" branch so we reach showdowns too.
                let a = if game.num_actions(&st) == 2 { 1 } else { 0 };
                st = game.apply(&st, a);
            }
            let (u0, u1) = (game.utility(&st, 0), game.utility(&st, 1));
            assert!((u0 + u1).abs() < 1e-9, "payoffs must sum to zero: {u0} + {u1}");
        }
    }

    #[test]
    fn strategy_is_monotone_premium_shoves_more_than_trash() {
        // A sanity check against the known solution shape: the SB shoves a strong
        // hand more often than a weak one.  We read the opening (no-history) SB
        // node for AA vs 72o.
        let game = PushFoldHoldem::new(40, 2, 1, 0);
        let mut s = Mccfr::with_seed(game, Variant::Dcfr(Discount::RECOMMENDED), 1);
        s.train(150_000);
        let avg = s.average_strategy();

        let game = PushFoldHoldem::new(40, 2, 1, 0);
        let shove_prob = |hole: [u8; 2]| -> f64 {
            // Reconstruct the SB opening info key for this exact hand.
            let mut h = hole;
            h.sort_unstable();
            let class = canonical_key(&h, &[]);
            let mut bytes = Vec::new();
            bytes.push(0u8); // SB is player 0 (button)
            bytes.extend_from_slice(&class);
            bytes.push(0xFF); // empty history
            let key = fnv1a(&bytes);
            avg.get(&key).map(|p| p[1]).unwrap_or(0.0) // p[1] = commit/shove
        };
        let _ = &game;
        let aces = shove_prob([make_card(12, 0), make_card(12, 1)]);
        let trash = shove_prob([make_card(5, 0), make_card(0, 1)]); // 7-2 offsuit
        assert!(aces > trash, "AA shove {aces} should exceed 72o shove {trash}");
        assert!(aces > 0.9, "AA should shove almost always, got {aces}");
    }

    /// The push/fold equilibrium is computed without the curated-deal trick, so
    /// it doubles as a reusable fixture for later checks.
    #[test]
    fn average_strategy_is_valid_distribution() {
        let game = PushFoldHoldem::new(30, 2, 1, 0);
        let mut s = Mccfr::with_seed(game, Variant::Vanilla, 1);
        s.train(10_000);
        let avg: HashMap<u64, Vec<f64>> = s.average_strategy();
        assert!(!avg.is_empty());
        for probs in avg.values() {
            assert_eq!(probs.len(), 2);
            assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }
}
