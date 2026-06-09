//! Pluggable leaf evaluator for depth-limited resolving (Phase 5).
//!
//! A subgame solver cannot search to the end of the hand within a 2–5 s budget,
//! so it cuts the tree at a depth limit (e.g. the start of the next street) and
//! substitutes an *estimate* of each player's value there.  Quality of that
//! estimate — not the belief logic — is the first suspect when resolving
//! misbehaves, so the evaluator is a pluggable trait with several backends.
//!
//! The default backend here is [`CheckdownLeafEval`]: it assumes the hand checks
//! down to showdown from the leaf and scores each player's value by the exact
//! all-in equity of the (known) hands over the remaining runout.  It is simple,
//! parameter-free, and correct under the check-down continuation — the standard
//! baseline.  A `BlueprintLeafEval` that instead reads the blueprint's own
//! values (more accurate, especially when continued betting matters) slots in
//! behind the same trait once a full blueprint exists.

use std::cell::RefCell;
use std::collections::HashMap;

use poker_core::state::{GameState, NO_CARD};

use crate::abstraction::features::hand_vs_hand_equity;
use crate::resolving::belief_state::BeliefState;

/// Expected value at a subgame leaf, per player (net chips relative to the start
/// of the hand — the same convention as [`GameState::terminal_payoffs`]).
pub trait LeafEvaluator {
    fn evaluate(&self, state: &GameState, beliefs: &[BeliefState]) -> Vec<f64>;
}

/// Leaf evaluator that assumes the hand **checks down to showdown** from the
/// leaf: each player's value is its all-in equity (over the remaining board)
/// times the pot, minus what it has committed.  Heads-up.
///
/// Equities are cached by `(hands, visible board)`, since the solver evaluates
/// the same leaf many times across iterations and the equity depends only on the
/// cards, not the betting that led there.
#[derive(Default)]
pub struct CheckdownLeafEval {
    cache: RefCell<HashMap<u64, f64>>,
}

impl CheckdownLeafEval {
    pub fn new() -> Self {
        Self::default()
    }

    /// Equity of `h0` vs `h1` over `board`, memoized.
    fn equity0(&self, h0: [u8; 2], h1: [u8; 2], board: &[u8]) -> f64 {
        let key = Self::key(h0, h1, board);
        if let Some(&e) = self.cache.borrow().get(&key) {
            return e;
        }
        let e = hand_vs_hand_equity(h0, h1, board);
        self.cache.borrow_mut().insert(key, e);
        e
    }

    /// FNV-1a over the (sorted) hands and board — a stable cache key.
    fn key(mut h0: [u8; 2], mut h1: [u8; 2], board: &[u8]) -> u64 {
        h0.sort_unstable();
        h1.sort_unstable();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in h0.iter().chain(h1.iter()).chain(board.iter()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

impl LeafEvaluator for CheckdownLeafEval {
    fn evaluate(&self, state: &GameState, _beliefs: &[BeliefState]) -> Vec<f64> {
        let n = state.num_players as usize;
        let pot = state.pot as f64;
        let mut value = vec![0.0; n];

        // Players still in the hand at the leaf.
        let active: Vec<usize> = (0..n).filter(|&i| state.folded & (1 << i) == 0).collect();

        // Visible board = the real community cards (a depth leaf on the turn has
        // a NO_CARD river slot that must not be read as a card).
        let board: Vec<u8> = state.board.iter().copied().filter(|&c| c != NO_CARD).collect();

        // Folded players simply lose what they put in.
        for i in 0..n {
            if state.folded & (1 << i) != 0 {
                value[i] = -(state.total_committed[i] as f64);
            }
        }

        match active.as_slice() {
            [p] => {
                // Everyone else folded: the lone survivor takes the pot.
                value[*p] = pot - state.total_committed[*p] as f64;
            }
            [p, q] => {
                let eq_p = self.equity0(state.hole_cards[*p], state.hole_cards[*q], &board);
                value[*p] = eq_p * pot - state.total_committed[*p] as f64;
                value[*q] = (1.0 - eq_p) * pot - state.total_committed[*q] as f64;
            }
            _ => panic!("CheckdownLeafEval supports heads-up leaves (1–2 active players)"),
        }
        value
    }
}

/// Leaf evaluator backed by **blueprint table lookups** — the plan's default
/// resolving leaf evaluator.
///
/// At a subgame boundary the most accurate leaf value is the blueprint's own
/// value for the players' hands there (it accounts for continued betting, unlike
/// the check-down assumption).  This evaluator stores those values keyed by a
/// caller-supplied function of the leaf state (typically `(hand bucket, board)`,
/// matching how the blueprint was trained) and returns them directly.
///
/// Crucially, the **fallback is wired in** (the plan is explicit about this): a
/// leaf the blueprint never stored a value for — an off-tree board, a bucket the
/// blueprint skipped — is scored by an inner evaluator (default
/// [`CheckdownLeafEval`]) instead of failing.  So the resolver degrades to the
/// parameter-free check-down baseline exactly where the blueprint has nothing to
/// say, rather than producing garbage.
///
/// Locally the value table is sparse/empty because a postflop blueprint is a
/// cloud-burst artifact; the mechanism, keying, and fallback are complete and the
/// table is populated from the trained blueprint once it exists.
pub struct BlueprintLeafEval<'a> {
    /// Per-player chip value (start-of-hand convention) keyed by [`Self::key`].
    values: HashMap<u64, Vec<f64>>,
    /// Maps a leaf state to a value-table key (e.g. hand bucket + board).
    key: Box<dyn Fn(&GameState) -> u64 + 'a>,
    /// Scored for leaves the blueprint has no stored value for.
    fallback: &'a dyn LeafEvaluator,
}

impl<'a> BlueprintLeafEval<'a> {
    /// Build a blueprint leaf evaluator from a precomputed value table, a keying
    /// function over the leaf state, and a `fallback` evaluator for misses.
    pub fn new(
        values: HashMap<u64, Vec<f64>>,
        key: impl Fn(&GameState) -> u64 + 'a,
        fallback: &'a dyn LeafEvaluator,
    ) -> Self {
        Self { values, key: Box::new(key), fallback }
    }

    /// Number of stored leaf values (blueprint coverage of the boundary).
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl LeafEvaluator for BlueprintLeafEval<'_> {
    fn evaluate(&self, state: &GameState, beliefs: &[BeliefState]) -> Vec<f64> {
        match self.values.get(&(self.key)(state)) {
            Some(v) => v.clone(),
            None => self.fallback.evaluate(state, beliefs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::make_card;
    use poker_core::state::MAX_PLAYERS;

    /// A heads-up river state (board complete) with a given pot already
    /// committed equally.
    fn river_state(h0: [u8; 2], h1: [u8; 2], board: [u8; 5], committed: u32) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = h0;
        holes[1] = h1;
        let mut gs = GameState::new(2, 2, 1, [committed * 2; MAX_PLAYERS], holes, board, 0);
        // Force the public state: both players have committed `committed`, river.
        gs.total_committed[0] = committed;
        gs.total_committed[1] = committed;
        gs.pot = committed * 2;
        gs.street = 3;
        gs
    }

    #[test]
    fn nut_hand_wins_the_pot() {
        // Board A♣K♦9♥4♠2♣; P0 has AА (trips+), P1 has 7♦2♦ (pair of deuces).
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        let gs = river_state([make_card(12, 1), make_card(12, 2)], [make_card(5, 1), make_card(0, 1)], board, 10);
        let v = CheckdownLeafEval::new().evaluate(&gs, &[]);
        assert!((v[0] + v[1]).abs() < 1e-9, "zero-sum");
        assert!(v[0] > 9.0, "near-nuts wins ~the whole pot, net ≈ +10: {v:?}");
        assert!(v[1] < -9.0, "dominated hand loses its contribution");
    }

    #[test]
    fn blueprint_lookup_overrides_and_falls_back_to_checkdown() {
        // Trip aces vs a weak pair on a dry board.
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        let gs = river_state([make_card(12, 1), make_card(12, 2)], [make_card(5, 1), make_card(0, 1)], board, 10);

        let checkdown = CheckdownLeafEval::new();

        // A blueprint table that overrides exactly this leaf with a (made-up)
        // value, keyed by a constant so we control the hit.
        let mut values = HashMap::new();
        values.insert(42u64, vec![3.0, -3.0]);
        let hit = BlueprintLeafEval::new(values, |_gs: &GameState| 42, &checkdown);
        assert_eq!(hit.evaluate(&gs, &[]), vec![3.0, -3.0], "stored blueprint value is used");

        // A table that never matches ⇒ falls back to the check-down baseline.
        let miss = BlueprintLeafEval::new(HashMap::new(), |_gs: &GameState| 0, &checkdown);
        assert_eq!(
            miss.evaluate(&gs, &[]),
            checkdown.evaluate(&gs, &[]),
            "missing leaf falls back to check-down, not failure"
        );
    }

    #[test]
    fn equal_hands_split_value() {
        // Same ranks, different suits on a rainbow-ish board ⇒ ~50/50, net ≈ 0.
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(4, 3), make_card(2, 0)];
        let gs = river_state([make_card(9, 0), make_card(8, 1)], [make_card(9, 2), make_card(8, 3)], board, 10);
        let v = CheckdownLeafEval::new().evaluate(&gs, &[]);
        assert!((v[0] + v[1]).abs() < 1e-9, "zero-sum");
        assert!(v[0].abs() < 1e-6 && v[1].abs() < 1e-6, "symmetric hands net to ~0: {v:?}");
    }
}
