//! Depth-limited subgame solver.
//!
//! Resolving turns the public state at the resolve root into a *subgame* and
//! solves it from scratch within a time budget, so the bot can answer bet sizes
//! the blueprint never abstracted.  This models the subgame as a [`Game`] and
//! hands it to the CFR⁺ core ([`PredictiveSolver`](crate::validation::solver::predictive::PredictiveSolver)):
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
//! be checked with the exact best response in [`crate::validation::solver::best_response`]
//! — the validation anchor used in the tests.
//!
//! Scale note: enumerating `range × range` deals is tractable for the small
//! ranges resolving narrows to, but full 1326×1326 ranges need the vectorized
//! public-tree formulation (each public node carrying a value vector over all
//! hands) rather than this explicit-deal tree — a later optimization.

mod solver;
#[cfg(test)]
mod tests;

use poker_core::legal_actions;
use poker_core::state::{GameState, NO_CARD};

use crate::games::Game;
use crate::resolving::belief_state::BeliefState;
use crate::validation::resolving::leaf_eval::LeafEvaluator;
use crate::util::hash::fnv1a;

pub use solver::{Resolved, SolverKind, SubgameSolver};

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
    /// Multi-valued leaf state.  At a depth-limit leaf with `K > 1`
    /// continuations this is `None` first — a decision node where the opponent
    /// picks a continuation — then `Some(i)` once chosen, a terminal scored by
    /// continuation `i`.  Always `None` when `K = 1` (no continuation node).
    continuation: Option<u8>,
}

impl SubgameNode {
    /// Build the initial play node for a deal: `template` with the two players'
    /// hole cards set (`holes[p]` to player `p`).  This is the deal-rooted state
    /// the [`Subgame`] places under its chance root — exposed so the re-solving
    /// gadget ([`crate::validation::resolving::gadget`]) can root its Follow subtree on the
    /// same betting tree.
    pub fn deal(template: &GameState, holes: [[u8; 2]; 2]) -> Self {
        let mut gs = template.clone();
        gs.hole_cards[0] = holes[0];
        gs.hole_cards[1] = holes[1];
        SubgameNode { gs: Some(gs), history: Vec::new(), continuation: None }
    }

    /// Hole cards of `player` at a play node (`None` at the pre-deal chance root).
    /// Used by counterfactual-value extraction to group deals by a player's hand.
    pub fn hole_cards(&self, player: usize) -> Option<[u8; 2]> {
        self.gs.as_ref().map(|gs| gs.hole_cards[player])
    }
}

/// A depth-limited heads-up subgame as a [`Game`].
pub struct Subgame<'a> {
    deals: Vec<Deal>,
    /// Chance children precomputed once from `deals`: a deal-rooted state plus
    /// its probability.  The root is visited every CFR⁺ iteration, so building
    /// these here (rather than in `chance_outcomes`) keeps the per-deal
    /// `template` clone + hole-card assignment out of the hot loop.
    outcomes: Vec<(SubgameNode, f64)>,
    leaf_eval: &'a dyn LeafEvaluator,
    big_blind: f64,
    /// Number of opponent continuations `K` offered at each depth-limit leaf
    /// `K = 1` ⇒ leaves are plain terminals (legacy behaviour).
    k: usize,
    /// The player who chooses the continuation at a leaf — the opponent of the
    /// resolve-root actor, whose post-leaf adaptation the resolve must be robust
    /// to.  Fixed for the whole subgame.
    chooser: usize,
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
        // The continuation chooser is the opponent of whoever acts at the root.
        let chooser = 1 - template.current_player();
        let k = leaf_eval.num_continuations().max(1);
        let outcomes = deals
            .iter()
            .map(|d| {
                let mut gs = template.clone();
                gs.hole_cards[0] = d.h0;
                gs.hole_cards[1] = d.h1;
                (SubgameNode { gs: Some(gs), history: Vec::new(), continuation: None }, d.prob)
            })
            .collect();
        Self { deals, outcomes, leaf_eval, big_blind, k, chooser }
    }

    /// Number of enumerated deals (the chance breadth).
    pub fn num_deals(&self) -> usize {
        self.deals.len()
    }

    /// Build a **play-only context** rooted at `template` — the same betting
    /// tree, leaf evaluation, and `info_key` behaviour as [`Self::new`], but with
    /// no enumerated deals (`chance_outcomes` is unused).  The re-solving gadget
    /// ([`crate::validation::resolving::gadget`]) drives its own chance and delegates each
    /// play node's [`Game`] methods to this context, so gadget play info sets
    /// share the exact keyspace of a plain [`Subgame`] resolve.
    pub fn play_context(template: &GameState, leaf_eval: &'a dyn LeafEvaluator) -> Self {
        let big_blind = template.big_blind as f64;
        let chooser = 1 - template.current_player();
        let k = leaf_eval.num_continuations().max(1);
        Self { deals: Vec::new(), outcomes: Vec::new(), leaf_eval, big_blind, k, chooser }
    }

    /// The precomputed chance children `(deal-rooted node, probability)` — the
    /// per-deal roots counterfactual-value extraction iterates over.
    pub fn outcomes(&self) -> &[(SubgameNode, f64)] {
        &self.outcomes
    }

    /// True when the engine's current street wants a board card the template does
    /// not have — i.e. play has advanced past the known board (a normal
    /// street-close at the depth limit, or an all-in run-out beyond it).  Such a
    /// node is a leaf scored by the evaluator, since its real showdown value
    /// depends on cards we are deliberately not searching.
    fn needs_leaf(&self, gs: &GameState) -> bool {
        gs.board[..gs.board_cards_count()].contains(&NO_CARD)
    }

    /// A depth-limit leaf with `K > 1` continuations that the opponent has not
    /// yet chosen between — a decision node for [`Self::chooser`], not a terminal
    /// False when `K = 1` (leaves are plain terminals).
    fn pending_continuation(&self, state: &SubgameNode) -> bool {
        self.k > 1
            && state.continuation.is_none()
            && state.gs.as_ref().is_some_and(|gs| self.needs_leaf(gs))
    }
}

impl Game for Subgame<'_> {
    type State = SubgameNode;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> SubgameNode {
        SubgameNode { gs: None, history: Vec::new(), continuation: None }
    }

    fn is_terminal(&self, state: &SubgameNode) -> bool {
        match &state.gs {
            // A multi-valued depth-limit leaf is a decision node until the
            // opponent has chosen a continuation; then it is terminal.
            Some(_) if self.pending_continuation(state) => false,
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
            // score a showdown it has no cards for).  With K > 1 the opponent has
            // chosen continuation `i`; with K = 1 it is the normal continuation.
            let conts = self.leaf_eval.continuations(gs, &[]);
            let i = state.continuation.unwrap_or(0) as usize;
            conts[i.min(conts.len() - 1)][player]
        } else {
            // Complete board and a real terminal (fold or river showdown): exact.
            gs.terminal_payoffs()[player] as f64
        };
        chips / self.big_blind
    }

    fn chance_outcomes(&self, _state: &SubgameNode) -> Vec<(SubgameNode, f64)> {
        // Precomputed in `Subgame::new`; the root is visited every iteration.
        self.outcomes.clone()
    }

    fn current_player(&self, state: &SubgameNode) -> usize {
        if self.pending_continuation(state) {
            // The opponent of the resolve-root actor chooses the continuation.
            return self.chooser;
        }
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, state: &SubgameNode) -> usize {
        let gs = state.gs.as_ref().expect("num_actions at a play node");
        if self.pending_continuation(state) {
            // One action per continuation the opponent may pick at this leaf.
            return self.leaf_eval.continuations(gs, &[]).len();
        }
        legal_actions(gs).len()
    }

    fn apply(&self, state: &SubgameNode, action: usize) -> SubgameNode {
        let gs = state.gs.as_ref().expect("apply at a play node");
        if self.pending_continuation(state) {
            // Record the chosen continuation; the node is now terminal.
            return SubgameNode {
                gs: Some(gs.clone()),
                history: state.history.clone(),
                continuation: Some(action as u8),
            };
        }
        let act = legal_actions(gs)[action];
        let mut next = gs.clone();
        next.apply_action(act);
        let mut history = state.history.clone();
        history.push(action as u8);
        SubgameNode { gs: Some(next), history, continuation: None }
    }

    fn info_key(&self, state: &SubgameNode) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        // At a continuation-choice node the actor is the fixed chooser, who keys
        // on its OWN hand (perfect recall: the continuation may depend on it).
        let continuation = self.pending_continuation(state);
        let player = if continuation { self.chooser } else { gs.current_player() };
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
        // Marker so a continuation-choice info set can never collide with a
        // betting info set at the same (player, hand, board, history).
        if continuation {
            bytes.push(0xFE);
        }
        fnv1a(&bytes)
    }
}
