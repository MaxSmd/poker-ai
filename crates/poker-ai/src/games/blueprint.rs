//! Sampled, card-abstracted heads-up NLHE — the real blueprint target
//! (Phase 2 → Phase 3).
//!
//! The curated-deal bridge ([`super::nlhe`]) proved the wiring by enumerating a
//! handful of concrete deals.  A blueprint cannot enumerate: the chance space is
//! every hole-card and board combination, ~10^9 deals before the betting tree
//! even begins.  This module closes the two gaps the bridge left open, exactly
//! the pieces the plan reserves for here:
//!
//!  1. **Sampled chance.**  [`Game::sample_chance`] deals a fresh random board
//!     and both hands by partial Fisher–Yates over a 52-card deck, so the
//!     solver never materializes the outcome list.  [`is_chance_enumerable`]
//!     returns `false`, which routes external-sampling MCCFR onto that path
//!     (and, correctly, makes the full-traversal solver and exact best response
//!     inapplicable — there is no finite tree to walk).
//!
//!  2. **Card abstraction in the key.**  Information sets are keyed on the
//!     *bucket* of the situation, not the raw cards: a per-street [`BucketMap`]
//!     ([`crate::abstraction`]) collapses strategically-similar `(hole, board)`
//!     situations together, which is what makes the regret table finite.
//!     Pre-flop uses the 169 suit-canonical hand classes directly; a street
//!     with no loaded abstraction falls back to its suit-canonical key (correct,
//!     just unabstracted).
//!
//! [`is_chance_enumerable`]: Game::is_chance_enumerable

use poker_core::legal_actions;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use super::nlhe::fnv1a;
use super::Game;
use crate::abstraction::bucket_map::BucketMap;
use crate::abstraction::canonical::canonical_key;

/// Number of cards consumed by a heads-up deal: 2 hole cards each + 5 board.
const DEAL_CARDS: usize = 9;

/// A heads-up NLHE game with sampled deals and per-street card abstraction.
pub struct BlueprintHoldem {
    stacks: [u32; MAX_PLAYERS],
    big_blind: u32,
    small_blind: u32,
    button: u8,
    /// Information abstraction for the post-flop streets, indexed
    /// `flop = 0, turn = 1, river = 2`.  `None` ⇒ that street is unabstracted.
    street_buckets: [Option<BucketMap>; 3],
}

/// A node: the pre-deal chance root (`gs == None`) or a play node wrapping a
/// concrete `GameState` plus the perfect-recall action history.
#[derive(Clone, Debug)]
pub struct BlueprintState {
    gs: Option<GameState>,
    history: Vec<u8>,
}

impl BlueprintHoldem {
    /// A game with equal starting stacks and no card abstraction loaded
    /// (every street keyed by its suit-canonical situation).
    pub fn new(stack: u32, big_blind: u32, small_blind: u32, button: u8) -> Self {
        let mut stacks = [0u32; MAX_PLAYERS];
        stacks[0] = stack;
        stacks[1] = stack;
        Self { stacks, big_blind, small_blind, button, street_buckets: [None, None, None] }
    }

    /// Attach a street's information abstraction (`flop = 0, turn = 1,
    /// river = 2`).
    pub fn with_street_bucket(mut self, street: usize, buckets: BucketMap) -> Self {
        self.street_buckets[street] = Some(buckets);
        self
    }

    /// Deal both hands and the full board from a freshly shuffled deck, drawing
    /// uniform units from `next_unit`.  Partial Fisher–Yates: only the first
    /// `DEAL_CARDS` positions are resolved.
    fn deal(&self, mut next_unit: impl FnMut() -> f64) -> GameState {
        // Cards are encoded `rank << 2 | suit`, so 0..52 enumerates the deck.
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        for i in 0..DEAL_CARDS {
            let span = 52 - i;
            let j = i + (next_unit() * span as f64) as usize;
            deck.swap(i, j.min(51));
        }
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [deck[0], deck[1]];
        holes[1] = [deck[2], deck[3]];
        let board = [deck[4], deck[5], deck[6], deck[7], deck[8]];
        GameState::new(2, self.big_blind, self.small_blind, self.stacks, holes, board, self.button)
    }

    /// The abstracted information key for the situation `(hole, board)` at the
    /// given street: a bucket id when an abstraction covers it, otherwise the
    /// suit-canonical key folded to `u64`.
    fn situation_bucket(&self, hole: &[u8; 2], board: &[u8]) -> u64 {
        let visible = board.len();
        if visible == 0 {
            // Pre-flop: the 169 suit-canonical starting-hand classes.
            return fnv1a(&canonical_key(hole, &[]));
        }
        let street = visible - 3; // flop = 0, turn = 1, river = 2
        match self.street_buckets.get(street).and_then(Option::as_ref) {
            Some(map) => match map.bucket(hole, board) {
                Some(b) => b as u64,
                // Outside the clustered set: stay correct by not abstracting.
                None => fnv1a(&canonical_key(hole, board)),
            },
            None => fnv1a(&canonical_key(hole, board)),
        }
    }
}

impl Game for BlueprintHoldem {
    type State = BlueprintState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> BlueprintState {
        BlueprintState { gs: None, history: Vec::new() }
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
        BlueprintState { gs: Some(self.deal(next_unit)), history: Vec::new() }
    }

    fn current_player(&self, state: &BlueprintState) -> usize {
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, state: &BlueprintState) -> usize {
        let gs = state.gs.as_ref().expect("num_actions at a play node");
        legal_actions(gs).len()
    }

    fn apply(&self, state: &BlueprintState, action: usize) -> BlueprintState {
        let gs = state.gs.as_ref().expect("apply at a play node");
        let act = legal_actions(gs)[action];
        let mut next_gs = gs.clone();
        next_gs.apply_action(act);
        let mut history = state.history.clone();
        history.push(action as u8);
        BlueprintState { gs: Some(next_gs), history }
    }

    fn info_key(&self, state: &BlueprintState) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        let player = gs.current_player();
        let mut hole = gs.hole_cards[player];
        hole.sort_unstable();
        let visible = gs.board_cards_count();
        let bucket = self.situation_bucket(&hole, &gs.board[..visible]);

        let mut bytes = Vec::with_capacity(11 + state.history.len());
        bytes.push(player as u8);
        bytes.push(visible as u8);
        bytes.extend_from_slice(&bucket.to_le_bytes());
        bytes.push(0xFF); // separator so bucket bytes / history can't blur
        bytes.extend_from_slice(&state.history);
        fnv1a(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::{make_card, rank_of, suit_of};

    use crate::solver::cfr::Variant;
    use crate::solver::dcfr::Discount;
    use crate::solver::mccfr::Mccfr;

    /// Suit-rotate a card by `+1 (mod 4)` — for asserting suit isomorphism.
    fn rotate_suit(c: u8) -> u8 {
        make_card(rank_of(c), (suit_of(c) + 1) % 4)
    }

    /// A tiny deterministic unit source for driving `sample_chance` directly.
    fn unit_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed | 1;
        move || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (v >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn sampled_deal_uses_nine_distinct_real_cards() {
        let game = BlueprintHoldem::new(100, 2, 1, 0);
        let root = game.root();
        assert!(game.is_chance(&root));
        assert!(!game.is_chance_enumerable(&root));

        for seed in 0..200u64 {
            let st = game.sample_chance(&root, unit_stream(seed));
            let gs = st.gs.as_ref().unwrap();
            let mut cards = Vec::new();
            cards.extend_from_slice(&gs.hole_cards[0]);
            cards.extend_from_slice(&gs.hole_cards[1]);
            cards.extend_from_slice(&gs.board);
            assert_eq!(cards.len(), DEAL_CARDS);
            assert!(cards.iter().all(|&c| c < 52), "every dealt card is a real card");
            cards.sort_unstable();
            cards.dedup();
            assert_eq!(cards.len(), DEAL_CARDS, "no card is dealt twice (seed {seed})");
        }
    }

    #[test]
    fn preflop_key_collapses_suit_isomorphic_hands() {
        // Two pre-flop situations that differ only by a global suit rotation must
        // share an information key (same 169-class), and they must differ from a
        // genuinely different starting hand.
        let game = BlueprintHoldem::new(100, 2, 1, 0);
        let mk = |holes: [[u8; 2]; 2]| {
            let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
            h[0] = holes[0];
            h[1] = holes[1];
            let board = [NO_CARD; 5];
            let gs = GameState::new(2, 2, 1, game.stacks, h, board, 0);
            BlueprintState { gs: Some(gs), history: Vec::new() }
        };
        // A♠K♠ vs 7♦7♣  →  rotate every suit  →  A♥K♥ vs 7♣7♠.
        let base = mk([[make_card(12, 0), make_card(11, 0)], [make_card(5, 1), make_card(5, 2)]]);
        let rot = mk([
            [rotate_suit(make_card(12, 0)), rotate_suit(make_card(11, 0))],
            [rotate_suit(make_card(5, 1)), rotate_suit(make_card(5, 2))],
        ]);
        // The acting pre-flop player is the same in both; keys must match.
        assert_eq!(game.info_key(&base), game.info_key(&rot));

        // A different starting hand (Q♠J♠) keys differently.
        let other = mk([[make_card(10, 0), make_card(9, 0)], [make_card(5, 1), make_card(5, 2)]]);
        assert_ne!(game.info_key(&base), game.info_key(&other));
    }

    #[test]
    fn mccfr_runs_over_sampled_blueprint() {
        // The keystone smoke test: external sampling drives the real engine
        // through sampled deals + bucketed keys, completes, and produces valid
        // probability distributions at every discovered info set.
        let game = BlueprintHoldem::new(40, 2, 1, 0);
        let mut solver = Mccfr::new(game, Variant::Dcfr(Discount::RECOMMENDED));
        solver.train(2_000);
        assert!(solver.num_info_sets() > 0, "should discover info sets");
        for (_key, probs) in solver.average_strategy() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
            assert!(probs.iter().all(|&p| p >= 0.0));
        }
    }

    #[test]
    fn baseline_mccfr_runs_over_sampled_blueprint() {
        // The VR-MCCFR chance baseline must gracefully no-op on a non-enumerable
        // chance node (no outcome list to index) yet still train cleanly.
        let game = BlueprintHoldem::new(40, 2, 1, 0);
        let mut solver = Mccfr::new(game, Variant::Vanilla).with_baseline();
        solver.train(1_000);
        assert!(solver.num_info_sets() > 0);
    }

    #[test]
    fn is_deterministic_for_fixed_seed() {
        let run = || {
            let game = BlueprintHoldem::new(40, 2, 1, 0);
            let mut s = Mccfr::with_seed(game, Variant::Vanilla, 99);
            s.train(1_000);
            s.num_info_sets()
        };
        assert_eq!(run(), run(), "same seed must visit the same info sets");
    }
}
