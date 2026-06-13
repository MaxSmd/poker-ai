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
///
/// ## Multi-valued leaves (depth-limited solving, Brown et al. 2018)
///
/// A single leaf value assumes **one** opponent continuation past the depth
/// limit, and the searcher overfits to it — the standard depth-limited-solving
/// exploit.  The fix is to let the opponent **choose** among `K` continuations
/// at the leaf (Pluribus: normal / fold- / call- / raise-biased copies of the
/// blueprint), so the searcher must be robust to the opponent adapting.  An
/// evaluator advertises `K` via [`num_continuations`](Self::num_continuations)
/// and supplies the `K` value vectors via [`continuations`](Self::continuations).
/// The default `K = 1` reproduces the single-continuation behaviour exactly, so
/// the subgame tree is unchanged unless an evaluator opts in.
pub trait LeafEvaluator {
    /// The "normal" continuation value, per player — the single value used when
    /// `K = 1` and the first entry of [`continuations`](Self::continuations).
    fn evaluate(&self, state: &GameState, beliefs: &[BeliefState]) -> Vec<f64>;

    /// Number of opponent continuations `K` offered at each leaf (default 1).
    fn num_continuations(&self) -> usize {
        1
    }

    /// The `K` per-player value vectors the opponent may choose among at this
    /// leaf.  Default: the single normal continuation, so `K = 1` evaluators
    /// need not override it.
    fn continuations(&self, state: &GameState, beliefs: &[BeliefState]) -> Vec<Vec<f64>> {
        vec![self.evaluate(state, beliefs)]
    }
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

/// Multi-valued check-down leaf: offers the opponent `K` continuations at the
/// leaf instead of a single check-down, the depth-limited-solving fix of finding
/// #1 (Brown et al. 2018).
///
/// **This is a constructed stand-in, not a blueprint-derived continuation.** The
/// poker-faithful four continuations (normal / fold- / call- / raise-biased
/// copies of the blueprint) need a trained blueprint and live in
/// [`BlueprintLeafEval`]; locally that table is empty.  To exercise — and test —
/// the *mechanism* without a blueprint, each continuation here is a different
/// **rest-of-hand pot scale**: both players put `scale · pot` more in (a notional
/// bet-and-call past the leaf), the realized check-down equity `e` unchanged.
/// Player `p`'s value under scale `s` is
///
/// ```text
/// v_p(s) = e·(P + sP) − (c_p + sP/2) = e·P − c_p + s·P·(e − 0.5)
/// ```
///
/// so inflating the pot is **+EV exactly when `e > 0.5`**: a strong hand wants a
/// big pot, a weak hand wants the smallest.  The opponent's best continuation is
/// therefore **hand-dependent**, which is precisely the adversarial structure a
/// single-continuation leaf ignores (and gets exploited for).  `scale = 0` is the
/// plain check-down, so `scales[0] = 0.0` reproduces [`CheckdownLeafEval`].  A
/// give-up ("fold") line is always dominated by checking down for the chooser, so
/// it is represented by the `0.0` floor rather than a separately stored option.
pub struct MultiContinuationLeaf {
    /// Rest-of-hand pot scales, one per continuation; `scales[0]` should be `0.0`
    /// (the normal check-down).  Default models check / small-bet / big-bet /
    /// overbet aggression of the remainder.
    scales: Vec<f64>,
    cache: RefCell<HashMap<u64, f64>>,
}

impl Default for MultiContinuationLeaf {
    fn default() -> Self {
        // normal, call-biased, raise-biased, overbet/raise-biased: the chooser
        // picks the pot size that best fits its hand strength.
        Self::with_scales(vec![0.0, 0.75, 1.5, 3.0])
    }
}

impl MultiContinuationLeaf {
    /// The default four-continuation evaluator (check / small / big / overbet).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build with explicit rest-of-hand pot scales (`scales[0]` is the normal
    /// continuation and should be `0.0`).
    pub fn with_scales(scales: Vec<f64>) -> Self {
        assert!(!scales.is_empty(), "need at least one continuation");
        Self { scales, cache: RefCell::new(HashMap::new()) }
    }

    /// Equity of `h0` vs `h1` over `board`, memoized (same key scheme as
    /// [`CheckdownLeafEval`]).
    fn equity0(&self, h0: [u8; 2], h1: [u8; 2], board: &[u8]) -> f64 {
        let key = CheckdownLeafEval::key(h0, h1, board);
        if let Some(&e) = self.cache.borrow().get(&key) {
            return e;
        }
        let e = hand_vs_hand_equity(h0, h1, board);
        self.cache.borrow_mut().insert(key, e);
        e
    }

    /// Per-player value vector at scale `s` for a leaf `state` (heads-up).
    fn value_at(&self, state: &GameState, board: &[u8], s: f64) -> Vec<f64> {
        let n = state.num_players as usize;
        let pot = state.pot as f64;
        let mut value = vec![0.0; n];
        let active: Vec<usize> = (0..n).filter(|&i| state.folded & (1 << i) == 0).collect();
        for i in 0..n {
            if state.folded & (1 << i) != 0 {
                value[i] = -(state.total_committed[i] as f64);
            }
        }
        match active.as_slice() {
            [p] => value[*p] = pot - state.total_committed[*p] as f64,
            [p, q] => {
                let e = self.equity0(state.hole_cards[*p], state.hole_cards[*q], board);
                let add = s * pot; // total extra chips in the pot past the leaf
                let inflated = pot + add;
                value[*p] = e * inflated - (state.total_committed[*p] as f64 + add / 2.0);
                value[*q] = (1.0 - e) * inflated - (state.total_committed[*q] as f64 + add / 2.0);
            }
            _ => panic!("MultiContinuationLeaf supports heads-up leaves (1–2 active players)"),
        }
        value
    }
}

impl LeafEvaluator for MultiContinuationLeaf {
    fn evaluate(&self, state: &GameState, _beliefs: &[BeliefState]) -> Vec<f64> {
        let board: Vec<u8> = state.board.iter().copied().filter(|&c| c != NO_CARD).collect();
        self.value_at(state, &board, self.scales[0])
    }

    fn num_continuations(&self) -> usize {
        self.scales.len()
    }

    fn continuations(&self, state: &GameState, _beliefs: &[BeliefState]) -> Vec<Vec<f64>> {
        let board: Vec<u8> = state.board.iter().copied().filter(|&c| c != NO_CARD).collect();
        self.scales.iter().map(|&s| self.value_at(state, &board, s)).collect()
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
    /// **Multi-valued** leaf table (finding #1): the `K` continuation value
    /// vectors per leaf key (normal / fold- / call- / raise-biased blueprint
    /// copies).  Empty until populated from a trained blueprint (cloud artifact);
    /// when empty the evaluator is single-continuation and identical to before.
    continuation_values: HashMap<u64, Vec<Vec<f64>>>,
    /// Continuations advertised via [`LeafEvaluator::num_continuations`] when the
    /// `K`-table is in use; falls back to the inner evaluator's `K` otherwise.
    k: usize,
    /// Maps a leaf state to a value-table key (e.g. hand bucket + board).
    key: Box<dyn Fn(&GameState) -> u64 + 'a>,
    /// Scored for leaves the blueprint has no stored value for.
    fallback: &'a dyn LeafEvaluator,
}

impl<'a> BlueprintLeafEval<'a> {
    /// Build a blueprint leaf evaluator from a precomputed value table, a keying
    /// function over the leaf state, and a `fallback` evaluator for misses.
    /// Single-continuation (`K` follows the fallback).
    pub fn new(
        values: HashMap<u64, Vec<f64>>,
        key: impl Fn(&GameState) -> u64 + 'a,
        fallback: &'a dyn LeafEvaluator,
    ) -> Self {
        let k = fallback.num_continuations();
        Self { values, continuation_values: HashMap::new(), k, key: Box::new(key), fallback }
    }

    /// Build a **multi-valued** blueprint leaf evaluator: `continuation_values`
    /// holds the `k` continuation vectors per leaf key.  `values` (the normal,
    /// single value) is derived as continuation 0 so [`LeafEvaluator::evaluate`]
    /// and the `K = 1` consumers still work.
    pub fn with_continuations(
        continuation_values: HashMap<u64, Vec<Vec<f64>>>,
        k: usize,
        key: impl Fn(&GameState) -> u64 + 'a,
        fallback: &'a dyn LeafEvaluator,
    ) -> Self {
        assert!(k >= 1, "need at least one continuation");
        let values = continuation_values
            .iter()
            .map(|(&key, conts)| (key, conts[0].clone()))
            .collect();
        Self { values, continuation_values, k, key: Box::new(key), fallback }
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

    fn num_continuations(&self) -> usize {
        self.k
    }

    fn continuations(&self, state: &GameState, beliefs: &[BeliefState]) -> Vec<Vec<f64>> {
        if let Some(v) = self.continuation_values.get(&(self.key)(state)) {
            return v.clone();
        }
        // Miss: fall back, but normalize to exactly `k` continuations so the
        // chooser node always has the same action count (a leaf the blueprint
        // never stored simply offers the fallback's value under every choice).
        let mut conts = self.fallback.continuations(state, beliefs);
        if conts.is_empty() {
            conts.push(self.fallback.evaluate(state, beliefs));
        }
        while conts.len() < self.k {
            conts.push(conts[0].clone());
        }
        conts.truncate(self.k);
        conts
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
    fn multi_continuation_normal_matches_checkdown_and_is_zero_sum() {
        // scales[0] = 0.0 ⇒ continuation 0 is exactly the check-down value, and
        // every continuation is zero-sum (net chips conserved past the leaf).
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        let gs = river_state([make_card(12, 1), make_card(12, 2)], [make_card(5, 1), make_card(0, 1)], board, 10);

        let multi = MultiContinuationLeaf::new();
        let check = CheckdownLeafEval::new();
        let conts = multi.continuations(&gs, &[]);
        assert_eq!(conts.len(), 4, "default K = 4");
        assert_eq!(multi.evaluate(&gs, &[]), check.evaluate(&gs, &[]), "normal == check-down");
        assert_eq!(conts[0], check.evaluate(&gs, &[]), "continuation 0 == check-down");
        for c in &conts {
            assert!((c[0] + c[1]).abs() < 1e-9, "continuation must be zero-sum: {c:?}");
        }
    }

    #[test]
    fn multi_continuation_best_choice_is_hand_dependent() {
        // The whole point: a strong hand wants the biggest pot, a weak hand the
        // smallest — so the opponent's best continuation depends on its hand,
        // which a single-continuation leaf cannot represent.
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        let multi = MultiContinuationLeaf::new();

        // P0 holds trips (very strong, e ≈ 1): more pot is strictly better.
        let strong = river_state([make_card(12, 1), make_card(12, 2)], [make_card(5, 1), make_card(3, 1)], board, 10);
        let cs = multi.continuations(&strong, &[]);
        let best_strong = (0..cs.len()).max_by(|&a, &b| cs[a][0].partial_cmp(&cs[b][0]).unwrap()).unwrap();
        assert_eq!(best_strong, cs.len() - 1, "strong hand prefers the largest pot");

        // P0 holds air (e ≈ 0): the smallest pot (normal) is best.
        let weak = river_state([make_card(5, 1), make_card(3, 1)], [make_card(12, 1), make_card(12, 2)], board, 10);
        let cw = multi.continuations(&weak, &[]);
        let best_weak = (0..cw.len()).max_by(|&a, &b| cw[a][0].partial_cmp(&cw[b][0]).unwrap()).unwrap();
        assert_eq!(best_weak, 0, "weak hand prefers the smallest pot (normal)");
    }

    #[test]
    fn blueprint_k_table_is_used_and_falls_back_to_k_continuations() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        let gs = river_state([make_card(12, 1), make_card(12, 2)], [make_card(5, 1), make_card(0, 1)], board, 10);
        let checkdown = CheckdownLeafEval::new();

        // A K = 3 continuation table that overrides exactly this leaf.
        let mut table: HashMap<u64, Vec<Vec<f64>>> = HashMap::new();
        table.insert(7u64, vec![vec![1.0, -1.0], vec![2.0, -2.0], vec![-3.0, 3.0]]);
        let hit = BlueprintLeafEval::with_continuations(table, 3, |_g: &GameState| 7, &checkdown);
        assert_eq!(hit.num_continuations(), 3);
        assert_eq!(
            hit.continuations(&gs, &[]),
            vec![vec![1.0, -1.0], vec![2.0, -2.0], vec![-3.0, 3.0]],
            "stored K continuations are used"
        );
        assert_eq!(hit.evaluate(&gs, &[]), vec![1.0, -1.0], "evaluate == continuation 0");

        // A miss must still yield exactly K vectors (the fallback's value under
        // every choice), so the chooser node has a stable action count.
        let miss = BlueprintLeafEval::with_continuations(HashMap::new(), 3, |_g: &GameState| 0, &checkdown);
        let conts = miss.continuations(&gs, &[]);
        assert_eq!(conts.len(), 3, "miss normalizes to K continuations");
        for c in &conts {
            assert_eq!(*c, checkdown.evaluate(&gs, &[]), "each falls back to check-down");
        }
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
