//! Depth-limited subgame solver (Phase 5 resolving).
//!
//! Resolving turns the public state at the resolve root into a *subgame* and
//! solves it from scratch within a time budget, so the bot can answer bet sizes
//! the blueprint never abstracted.  This models the subgame as a [`Game`] and
//! hands it to the CFR⁺ core ([`PredictiveSolver`]):
//!
//!  * **Chance** = the deal of both players' hole cards from their belief ranges
//!    ([`BeliefState`]), with card removal — i.e. the standard *range vs range*
//!    root.
//!  * **Play** = real `poker-core` betting from the root public state, so any
//!    bet size is handled exactly.
//!  * **Leaves** = either a real terminal (fold/showdown), scored by
//!    `poker-core`, or the **depth limit** (the tree is cut when the street
//!    advances past the limit), scored by a pluggable [`LeafEvaluator`].  A river
//!    subgame has no depth cut and is solved exactly to showdown.
//!
//! Because the range-vs-range chance is *enumerable*, the resolved strategy can
//! be checked with the exact best response in [`crate::solver::best_response`]
//! — the validation anchor used in the tests.
//!
//! Scale note: enumerating `range × range` deals is tractable for the small
//! ranges resolving narrows to, but full 1326×1326 ranges need the vectorized
//! public-tree formulation (each public node carrying a value vector over all
//! hands) rather than this explicit-deal tree — a later optimization.

use std::collections::HashMap;
use std::time::Instant;

use poker_core::legal_actions;
use poker_core::state::{GameState, NO_CARD};

use crate::games::nlhe::fnv1a;
use crate::games::Game;
use crate::resolving::belief_state::BeliefState;
use crate::resolving::leaf_eval::LeafEvaluator;
use crate::solver::cfr::{Cfr, Variant};
use crate::solver::predictive::PredictiveSolver;

/// One enumerated range-vs-range deal.
#[derive(Clone, Debug)]
struct Deal {
    h0: [u8; 2],
    h1: [u8; 2],
    prob: f64,
}

/// Enumerate the deals consistent with both belief ranges: every `(h0, h1)` with
/// no shared cards and none on the board, weighted by the product of the
/// marginals and renormalized.
fn deals_from_beliefs(template: &GameState, b0: &BeliefState, b1: &BeliefState) -> Vec<Deal> {
    let mut board_mask = 0u64;
    for &c in &template.board {
        if c != NO_CARD {
            board_mask |= 1 << c;
        }
    }
    let mut deals = Vec::new();
    let mut total = 0.0;
    for (h0, p0) in b0.iter_nonzero() {
        let m0 = (1u64 << h0[0]) | (1u64 << h0[1]);
        if m0 & board_mask != 0 {
            continue;
        }
        for (h1, p1) in b1.iter_nonzero() {
            let m1 = (1u64 << h1[0]) | (1u64 << h1[1]);
            if m1 & board_mask != 0 || m0 & m1 != 0 {
                continue;
            }
            deals.push(Deal { h0, h1, prob: p0 * p1 });
            total += p0 * p1;
        }
    }
    if total > 0.0 {
        for d in &mut deals {
            d.prob /= total;
        }
    }
    deals
}

/// A node in the subgame: the pre-deal chance root (`gs == None`) or a play node.
#[derive(Clone, Debug)]
pub struct SubgameNode {
    gs: Option<GameState>,
    history: Vec<u8>,
}

/// A depth-limited heads-up subgame as a [`Game`].
pub struct Subgame<'a> {
    template: GameState,
    deals: Vec<Deal>,
    leaf_eval: &'a dyn LeafEvaluator,
    big_blind: f64,
}

impl<'a> Subgame<'a> {
    /// Build a subgame rooted at `template` (the resolve-root public state) over
    /// `beliefs[0]` / `beliefs[1]`.  The depth limit is set implicitly by the
    /// template's board: the tree is cut at any node whose street wants a board
    /// card the template does not have (a `NO_CARD` slot), and that leaf is
    /// scored by `leaf_eval`.  A complete (river) board has no cut and is solved
    /// exactly to showdown.
    pub fn new(template: GameState, beliefs: &[BeliefState], leaf_eval: &'a dyn LeafEvaluator) -> Self {
        assert_eq!(beliefs.len(), 2, "heads-up resolving needs two belief ranges");
        let deals = deals_from_beliefs(&template, &beliefs[0], &beliefs[1]);
        let big_blind = template.big_blind as f64;
        Self { template, deals, leaf_eval, big_blind }
    }

    /// Number of enumerated deals (the chance breadth).
    pub fn num_deals(&self) -> usize {
        self.deals.len()
    }

    /// True when the engine's current street wants a board card the template does
    /// not have — i.e. play has advanced past the known board (a normal
    /// street-close at the depth limit, or an all-in run-out beyond it).  Such a
    /// node is a leaf scored by the evaluator, since its real showdown value
    /// depends on cards we are deliberately not searching.
    fn needs_leaf(&self, gs: &GameState) -> bool {
        gs.board[..gs.board_cards_count()].contains(&NO_CARD)
    }
}

impl Game for Subgame<'_> {
    type State = SubgameNode;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> SubgameNode {
        SubgameNode { gs: None, history: Vec::new() }
    }

    fn is_terminal(&self, state: &SubgameNode) -> bool {
        match &state.gs {
            Some(gs) => gs.is_terminal() || self.needs_leaf(gs),
            None => false,
        }
    }

    fn is_chance(&self, state: &SubgameNode) -> bool {
        state.gs.is_none()
    }

    fn utility(&self, state: &SubgameNode, player: usize) -> f64 {
        let gs = state.gs.as_ref().expect("utility at a play node");
        let chips = if self.needs_leaf(gs) {
            // Play advanced past the known board: estimate (the engine cannot
            // score a showdown it has no cards for).
            self.leaf_eval.evaluate(gs, &[])[player]
        } else {
            // Complete board and a real terminal (fold or river showdown): exact.
            gs.terminal_payoffs()[player] as f64
        };
        chips / self.big_blind
    }

    fn chance_outcomes(&self, _state: &SubgameNode) -> Vec<(SubgameNode, f64)> {
        self.deals
            .iter()
            .map(|d| {
                let mut gs = self.template.clone();
                gs.hole_cards[0] = d.h0;
                gs.hole_cards[1] = d.h1;
                (SubgameNode { gs: Some(gs), history: Vec::new() }, d.prob)
            })
            .collect()
    }

    fn current_player(&self, state: &SubgameNode) -> usize {
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, state: &SubgameNode) -> usize {
        let gs = state.gs.as_ref().expect("num_actions at a play node");
        legal_actions(gs).len()
    }

    fn apply(&self, state: &SubgameNode, action: usize) -> SubgameNode {
        let gs = state.gs.as_ref().expect("apply at a play node");
        let act = legal_actions(gs)[action];
        let mut next = gs.clone();
        next.apply_action(act);
        let mut history = state.history.clone();
        history.push(action as u8);
        SubgameNode { gs: Some(next), history }
    }

    fn info_key(&self, state: &SubgameNode) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        let player = gs.current_player();
        let mut hole = gs.hole_cards[player];
        hole.sort_unstable();

        let mut bytes = Vec::with_capacity(8 + state.history.len());
        bytes.push(player as u8);
        bytes.push(hole[0]);
        bytes.push(hole[1]);
        for &c in &gs.board {
            if c != NO_CARD {
                bytes.push(c);
            }
        }
        bytes.push(0xFF); // separator so board / history can't blur together
        bytes.extend_from_slice(&state.history);
        fnv1a(&bytes)
    }
}

/// Resolved subgame output.
pub struct Resolved {
    /// Strategy per information set (the resolver's deployable last iterate).
    pub strategy: HashMap<u64, Vec<f64>>,
    /// Number of enumerated deals (chance breadth).
    pub deals: usize,
    /// Number of information sets discovered.
    pub info_sets: usize,
}

/// Which regret minimizer resolves the subgame.
///
/// The plan's default is **predictive** (CFR⁺): in the near-two-player,
/// full-traversal regime a subgame becomes once folds collapse the active set,
/// CFR⁺'s fast last iterate buys the best strategy per second.  But the plan also
/// requires a **DCFR fallback**: predictive RM⁺'s strong guarantees are a 2p0s
/// result, so a *multiway* subgame (several opponents still in) should fall back
/// to DCFR, which is empirically robust where the predictive guarantee erodes.
/// Both consume the identical subgame tree, so the fallback is a one-line switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    /// CFR⁺ last iterate — the default for heads-up / near-heads-up subgames.
    Predictive,
    /// Discounted CFR average — the robust fallback for multiway subgames.
    Dcfr,
}

/// Depth-limited subgame solver: builds the subgame and solves it with the chosen
/// regret minimizer (predictive CFR⁺ by default, DCFR for the multiway fallback).
pub struct SubgameSolver {
    /// Streets to solve before cutting to leaf estimates (1–2 is realistic).
    pub depth_limit: u32,
    /// Wall-clock budget per resolving call.
    pub time_budget_ms: u64,
    /// Regret minimizer used to resolve.
    pub kind: SolverKind,
    /// Optional blueprint warm-start (predictive only): seed regrets so the first
    /// iterate is the blueprint instead of uniform.  See [`crate::resolving::warm_start`].
    warm_start: Option<HashMap<u64, Vec<f64>>>,
}

impl SubgameSolver {
    /// A predictive (CFR⁺) subgame solver — the default.
    pub fn new(depth_limit: u32, time_budget_ms: u64) -> Self {
        Self { depth_limit, time_budget_ms, kind: SolverKind::Predictive, warm_start: None }
    }

    /// Select the regret minimizer (e.g. [`SolverKind::Dcfr`] for a multiway
    /// fallback).
    pub fn with_solver(mut self, kind: SolverKind) -> Self {
        self.kind = kind;
        self
    }

    /// Warm-start the (predictive) solver's regrets from a blueprint, expressed
    /// over the subgame's own information sets.  Ignored on the DCFR path.
    pub fn with_warm_start(mut self, seed_regrets: HashMap<u64, Vec<f64>>) -> Self {
        self.warm_start = Some(seed_regrets);
        self
    }

    /// Resolve the subgame rooted at `root` over the given `beliefs`, training
    /// until the wall-clock budget is spent.  Returns the deployable strategy
    /// (CFR⁺ last iterate, or DCFR average on the fallback path).
    pub fn solve(
        &self,
        root: &GameState,
        beliefs: &[BeliefState],
        leaf_eval: &dyn LeafEvaluator,
    ) -> Resolved {
        let subgame = Subgame::new(root.clone(), beliefs, leaf_eval);
        let deals = subgame.num_deals();
        match self.kind {
            SolverKind::Predictive => {
                let mut solver = PredictiveSolver::new(subgame);
                if let Some(seed) = &self.warm_start {
                    solver.warm_start(seed.clone());
                }
                let start = Instant::now();
                loop {
                    solver.train(32);
                    if start.elapsed().as_millis() >= self.time_budget_ms as u128 {
                        break;
                    }
                }
                Resolved { strategy: solver.current_strategy(), deals, info_sets: solver.num_info_sets() }
            }
            SolverKind::Dcfr => {
                let mut solver = Cfr::new(subgame, Variant::Dcfr(crate::solver::dcfr::Discount::RECOMMENDED));
                let start = Instant::now();
                loop {
                    solver.train(32);
                    if start.elapsed().as_millis() >= self.time_budget_ms as u128 {
                        break;
                    }
                }
                Resolved { strategy: solver.average_strategy(), deals, info_sets: solver.num_info_sets() }
            }
        }
    }

    /// Deterministic resolve for a fixed iteration count (used by tests and when
    /// reproducibility matters more than a wall-clock budget).
    pub fn solve_for_iters(
        &self,
        root: &GameState,
        beliefs: &[BeliefState],
        leaf_eval: &dyn LeafEvaluator,
        iters: u64,
    ) -> Resolved {
        let subgame = Subgame::new(root.clone(), beliefs, leaf_eval);
        let deals = subgame.num_deals();
        match self.kind {
            SolverKind::Predictive => {
                let mut solver = PredictiveSolver::new(subgame);
                if let Some(seed) = &self.warm_start {
                    solver.warm_start(seed.clone());
                }
                solver.train(iters);
                Resolved { strategy: solver.current_strategy(), deals, info_sets: solver.num_info_sets() }
            }
            SolverKind::Dcfr => {
                let mut solver = Cfr::new(subgame, Variant::Dcfr(crate::solver::dcfr::Discount::RECOMMENDED));
                solver.train(iters);
                Resolved { strategy: solver.average_strategy(), deals, info_sets: solver.num_info_sets() }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolving::leaf_eval::CheckdownLeafEval;
    use crate::resolving::warm_start::warm_start_regrets;
    use crate::solver::best_response::exploitability;
    use poker_core::action::Action;
    use poker_core::make_card;
    use poker_core::state::MAX_PLAYERS;

    /// Drive a fresh hand to the start of `target_street` by checking/calling
    /// through, producing a clean public root via real poker-core mechanics.
    /// Hole cards are placeholders (overwritten per deal in the subgame).
    fn public_root(board: [u8; 5], stack: u32, target_street: u8) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        // Placeholders distinct from the board and each other (skip empty slots).
        let mut used = 0u64;
        for &c in &board {
            if c != NO_CARD {
                used |= 1 << c;
            }
        }
        let mut spare = (0u8..52).filter(|&c| used & (1 << c) == 0);
        holes[0] = [spare.next().unwrap(), spare.next().unwrap()];
        holes[1] = [spare.next().unwrap(), spare.next().unwrap()];

        let mut gs = GameState::new(2, 2, 1, [stack; MAX_PLAYERS], holes, board, 0);
        while gs.street < target_street && !gs.is_terminal() {
            let acts = legal_actions(&gs);
            // Prefer Check; otherwise Call — never put extra money in.
            let act = if acts.iter().any(|&a| a == Action::Check) {
                Action::Check
            } else {
                Action::Call
            };
            gs.apply_action(act);
        }
        gs
    }

    fn river_board() -> [u8; 5] {
        // A♣ K♦ 9♥ 4♠ 2♣
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)]
    }

    #[test]
    fn deals_respect_card_removal() {
        let root = public_root(river_board(), 20, 3);
        // Ranges that include hands overlapping the board and each other.
        let b0 = BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)], river_board()[0..2].try_into().unwrap()]);
        let b1 = BeliefState::from_hands(&[[make_card(10, 0), make_card(10, 1)], [make_card(12, 1), make_card(8, 0)]]);
        let leaf = CheckdownLeafEval::new();
        let sg = Subgame::new(root, &[b0, b1], &leaf);
        // No deal may reuse a board card or share a card across the two hands.
        let board_mask: u64 = river_board().iter().fold(0, |m, &c| m | (1 << c));
        for d in &sg.deals {
            let m0 = (1u64 << d.h0[0]) | (1u64 << d.h0[1]);
            let m1 = (1u64 << d.h1[0]) | (1u64 << d.h1[1]);
            assert_eq!(m0 & board_mask, 0, "hole on board");
            assert_eq!(m1 & board_mask, 0, "hole on board");
            assert_eq!(m0 & m1, 0, "hands share a card");
        }
        assert!(sg.num_deals() > 0);
    }

    #[test]
    fn payoffs_are_zero_sum_everywhere() {
        let root = public_root(river_board(), 20, 3);
        let b0 = BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)], // trip aces
            [make_card(10, 0), make_card(9, 0)],  // weak
        ]);
        let b1 = BeliefState::from_hands(&[
            [make_card(11, 0), make_card(11, 2)], // trip kings
            [make_card(8, 0), make_card(8, 1)],   // pair
        ]);
        let leaf = CheckdownLeafEval::new();
        let sg = Subgame::new(root, &[b0, b1], &leaf);

        fn walk(g: &Subgame, s: &SubgameNode) {
            if g.is_terminal(s) {
                let (u0, u1) = (g.utility(s, 0), g.utility(s, 1));
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
        walk(&sg, &sg.root());
    }

    #[test]
    fn river_subgame_resolves_to_low_exploitability() {
        // The end-to-end resolver check: belief ranges → subgame → CFR+, and the
        // resolved strategy is near-optimal *within the subgame* (measured by the
        // exact best response, which the enumerable chance makes feasible).
        let root = public_root(river_board(), 20, 3);
        let b0 = BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)], // nuts-ish (trips)
            [make_card(6, 0), make_card(5, 0)],   // air
        ]);
        let b1 = BeliefState::from_hands(&[
            [make_card(8, 0), make_card(8, 1)],   // medium pair (bluff-catcher)
            [make_card(10, 0), make_card(9, 1)],  // weak
        ]);
        let leaf = CheckdownLeafEval::new();

        let solver = SubgameSolver::new(1, 0);
        let resolved = solver.solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 3_000);
        assert!(resolved.info_sets > 0);

        // Rebuild an identical subgame to score exploitability of the strategy.
        let sg = Subgame::new(public_root(river_board(), 20, 3), &[b0, b1], &leaf);
        let expl = exploitability(&sg, &resolved.strategy);
        assert!(expl < 0.05, "resolved river subgame exploitability {expl} bb should be small");
    }

    #[test]
    fn turn_subgame_uses_the_leaf_evaluator() {
        // A turn root with depth 1 ⇒ the tree is cut at the river and scored by
        // the check-down evaluator.  It must still be a well-formed, zero-sum,
        // solvable game.
        let turn_board =
            [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD];
        let root = public_root(turn_board, 20, 2);
        assert_eq!(root.street, 2, "root should be on the turn");

        let b0 = BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)], [make_card(6, 0), make_card(5, 0)]]);
        let b1 = BeliefState::from_hands(&[[make_card(8, 0), make_card(8, 1)], [make_card(10, 0), make_card(9, 1)]]);
        let leaf = CheckdownLeafEval::new();

        let solver = SubgameSolver::new(1, 0);
        let resolved = solver.solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 1_000);
        assert!(resolved.info_sets > 0, "turn subgame should discover info sets");

        // Strategies are valid distributions.
        for probs in resolved.strategy.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
        }
    }

    // ----- Phase 5 completion: warm-start, DCFR fallback, comparison, stress -----

    /// A clean river root with an arbitrarily inflated pot — the public state the
    /// resolver receives after an **off-tree** (e.g. overbet) line on a prior
    /// street put `extra_each` extra chips in from each player.  Built by real
    /// mechanics, then the (street-start, nobody-owes) pot is scaled up while
    /// conserving chips, so it is a valid public state the abstraction never
    /// would have produced.
    fn river_root_with_extra_pot(board: [u8; 5], stack: u32, extra_each: u32) -> GameState {
        let mut gs = public_root(board, stack, 3);
        for i in 0..2 {
            gs.total_committed[i] += extra_each;
            gs.pot += extra_each;
            gs.stacks[i] -= extra_each;
        }
        gs
    }

    fn duel_ranges() -> (BeliefState, BeliefState) {
        let b0 = BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)], // trips (nuts-ish)
            [make_card(6, 0), make_card(5, 0)],   // air
        ]);
        let b1 = BeliefState::from_hands(&[
            [make_card(8, 0), make_card(8, 1)],   // bluff-catcher
            [make_card(10, 0), make_card(9, 1)],  // weak
        ]);
        (b0, b1)
    }

    #[test]
    fn off_tree_overbet_pot_river_resolves_low_exploitability() {
        // The resolver does not need the bet size in any abstraction: it resolves
        // from whatever public state it is handed.  Here a big off-tree pot is
        // already in (a prior overbet line); the river subgame must still resolve
        // near-optimally (exact BR, complete board).
        let (b0, b1) = duel_ranges();
        let leaf = CheckdownLeafEval::new();
        let root = river_root_with_extra_pot(river_board(), 60, 40);
        assert!(root.pot >= 84, "pot should be inflated by the off-tree line: {}", root.pot);

        let resolved = SubgameSolver::new(1, 0).solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 3_000);
        let sg = Subgame::new(river_root_with_extra_pot(river_board(), 60, 40), &[b0, b1], &leaf);
        let expl = exploitability(&sg, &resolved.strategy);
        assert!(expl < 0.05, "off-tree-pot river resolved to {expl} bb, should be small");
    }

    #[test]
    fn dcfr_fallback_resolves_the_subgame() {
        // The multiway fallback path (plan caveat): the same subgame tree solved
        // with DCFR instead of predictive RM⁺.  Validated heads-up (we have no
        // multiway subgame yet) — it must reach a near-optimal strategy too.
        let (b0, b1) = duel_ranges();
        let leaf = CheckdownLeafEval::new();
        let root = public_root(river_board(), 20, 3);

        let resolved = SubgameSolver::new(1, 0)
            .with_solver(SolverKind::Dcfr)
            .solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 4_000);

        let sg = Subgame::new(public_root(river_board(), 20, 3), &[b0, b1], &leaf);
        let expl = exploitability(&sg, &resolved.strategy);
        assert!(expl < 0.05, "DCFR fallback resolved to {expl} bb, should be small");
    }

    #[test]
    fn predictive_matches_or_beats_dcfr_on_the_subgame() {
        // The Phase 5 deliverable: a recorded comparison of predictive vs DCFR
        // subgame solving at an equal budget.  Both reach a good strategy; the
        // predictive (CFR⁺) last iterate should be at least as good as DCFR's
        // average — the reason the resolver defaults to it.
        let (b0, b1) = duel_ranges();
        let leaf = CheckdownLeafEval::new();
        let iters = 2_000;

        let pred = SubgameSolver::new(1, 0)
            .solve_for_iters(&public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf, iters);
        let dcfr = SubgameSolver::new(1, 0)
            .with_solver(SolverKind::Dcfr)
            .solve_for_iters(&public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf, iters);

        let expl_pred = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf), &pred.strategy);
        let expl_dcfr = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &[b0, b1], &leaf), &dcfr.strategy);

        // Recorded comparison (visible with `--nocapture`).
        println!("subgame resolve @ {iters} iters — predictive: {expl_pred:.5} bb, DCFR: {expl_dcfr:.5} bb");
        assert!(expl_pred < 0.05 && expl_dcfr < 0.05, "both solvers should resolve well");
        assert!(
            expl_pred <= expl_dcfr + 1e-3,
            "predictive ({expl_pred}) should be at least as good as DCFR ({expl_dcfr})"
        );
    }

    #[test]
    fn warm_start_speeds_convergence() {
        // Warm-starting from a blueprint (here a converged strategy on the *same*
        // subgame, so the info-set keys match) reaches a far lower exploitability
        // in a handful of iterations than a cold (uniform) start does.
        let (b0, b1) = duel_ranges();
        let leaf = CheckdownLeafEval::new();
        let beliefs = [b0.clone(), b1.clone()];

        // A near-equilibrium "blueprint" for this subgame.
        let blueprint = SubgameSolver::new(1, 0)
            .solve_for_iters(&public_root(river_board(), 20, 3), &beliefs, &leaf, 4_000)
            .strategy;
        let seed = warm_start_regrets(&blueprint, 50.0);

        let few = 3;
        let cold = SubgameSolver::new(1, 0).solve_for_iters(&public_root(river_board(), 20, 3), &beliefs, &leaf, few);
        let warm = SubgameSolver::new(1, 0)
            .with_warm_start(seed)
            .solve_for_iters(&public_root(river_board(), 20, 3), &beliefs, &leaf, few);

        let expl_cold = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &beliefs, &leaf), &cold.strategy);
        let expl_warm = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &beliefs, &leaf), &warm.strategy);
        println!("after {few} iters — cold: {expl_cold:.5} bb, warm-started: {expl_warm:.5} bb");
        assert!(expl_warm < expl_cold, "warm start ({expl_warm}) should beat cold ({expl_cold}) at {few} iters");
    }

    #[test]
    fn flop_subgame_cuts_at_turn_and_resolves() {
        // A flop root (board = 3 cards + two NO_CARD slots): play resolves the
        // flop betting — including off-tree all-in lines — and is cut at the turn,
        // scored by the check-down leaf evaluator.  Must be well-formed: info sets
        // discovered, valid distributions, zero-sum.
        let flop_board =
            [make_card(12, 0), make_card(11, 1), make_card(7, 2), NO_CARD, NO_CARD];
        let root = public_root(flop_board, 20, 1);
        assert_eq!(root.street, 1, "root should be on the flop");

        let b0 = BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)], [make_card(6, 0), make_card(5, 0)]]);
        let b1 = BeliefState::from_hands(&[[make_card(8, 0), make_card(8, 1)], [make_card(10, 0), make_card(9, 1)]]);
        let leaf = CheckdownLeafEval::new();

        let resolved = SubgameSolver::new(1, 0).solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 1_000);
        assert!(resolved.info_sets > 0, "flop subgame should discover info sets");
        for probs in resolved.strategy.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
        }

        // Zero-sum at every leaf (terminal or depth-cut).
        let sg = Subgame::new(root, &[b0, b1], &leaf);
        fn walk(g: &Subgame, s: &SubgameNode) {
            if g.is_terminal(s) {
                assert!((g.utility(s, 0) + g.utility(s, 1)).abs() < 1e-9, "zero-sum at leaf");
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
        walk(&sg, &sg.root());
    }

    #[test]
    fn check_raise_line_is_in_the_subgame_tree() {
        // The resolver solves over real betting, so check-raise lines (a common
        // resolving failure mode) are genuinely in the tree, not abstracted away.
        // Confirm a [check, then aggressive] line is reachable for some deal.
        let (b0, b1) = duel_ranges();
        let leaf = CheckdownLeafEval::new();
        let sg = Subgame::new(public_root(river_board(), 40, 3), &[b0, b1], &leaf);

        // From the first deal, the first player checks; the second bets/raises;
        // the first then raises = a check-raise.
        let deal = sg.chance_outcomes(&sg.root())[0].0.clone();
        let acts0 = {
            let gs = deal.gs.as_ref().unwrap();
            poker_core::legal_actions(gs)
        };
        let check_idx = acts0.iter().position(|&a| a == Action::Check).expect("first player can check");
        let after_check = sg.apply(&deal, check_idx);

        let acts1 = {
            let gs = after_check.gs.as_ref().unwrap();
            poker_core::legal_actions(gs)
        };
        let bet_idx = acts1
            .iter()
            .position(|&a| matches!(a, Action::Raise(_)) || a == Action::AllIn)
            .expect("second player can bet into the check");
        let after_bet = sg.apply(&after_check, bet_idx);

        let acts2 = {
            let gs = after_bet.gs.as_ref().unwrap();
            poker_core::legal_actions(gs)
        };
        assert!(
            acts2.iter().any(|&a| matches!(a, Action::Raise(_)) || a == Action::AllIn),
            "the checker can raise over the bet — a check-raise line exists in the subgame"
        );
    }
}
