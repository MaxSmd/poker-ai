//! Vectorized range-vs-range subgame solving.
//!
//! The explicit-deal [`Subgame`](crate::validation::resolving::subgame::Subgame) enumerates
//! every `(h0, h1)` pair as a chance child — up to 1326×1326 ≈ 1.76 M
//! traversals.  The **public-tree** formulation here walks the shared betting
//! tree *once* and carries a length-[`NUM_COMBOS`] counterfactual-value vector
//! per node: hands are a vector dimension, not chance branches.  Card removal
//! between the two players is deferred to the terminals, where the showdown over
//! two ranges is the reach-weighted O(n log n) sweep
//! [`board_cfvs`](crate::abstraction::features::board_cfvs) rather than a
//! 1326×1326 pairwise loop.  The result is ~100–1000× the resolver throughput,
//! which is what makes full-range (not narrowed) resolves and deeper limits
//! affordable.
//!
//! The explicit-deal `Subgame` stays the **correctness oracle**: this solver
//! emits its average strategy under the *same* `info_key` (player + hand +
//! board + history), so
//! [`exploitability`](crate::validation::solver::best_response::exploitability) scores the
//! vectorized result inside the explicit game and the two must agree.
//!
//! **Scope.** This solves *complete-board* (river) subgames — the full
//! 1326×1326 range-vs-range case — with exact showdown and fold terminals, and
//! *turn* subgames with the river depth-cut: any node that reaches the undealt
//! river (the turn betting closing, or a turn all-in) is a leaf whose
//! check-down showdown is averaged over the 44 live river runouts
//! (`board_runout_cfvs`).  That average reproduces the explicit oracle's
//! `CheckdownLeafEval` value exactly, so the same exploitability cross-check
//! validates turn and flop resolves.  Depth-limit leaves with `K > 1`
//! continuations are supported via the continuation-choice node (see
//! `build.rs`).

mod build;
mod keys;
mod node;
mod solve;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Mutex;

use poker_core::legal_actions;
use poker_core::state::{GameState, NO_CARD};

use crate::abstraction::features::{combo_cards, PreparedRunout, PreparedShowdown};
use crate::resolving::belief_state::{BeliefState, NUM_COMBOS};

use keys::MARKER_GADGET;
use node::{NodeKind, NodeStore};

pub use keys::subgame_info_key;

/// A solved vectorized subgame (mirrors [`Resolved`](crate::validation::resolving::subgame::Resolved)).
pub struct VectorResolved {
    /// Average strategy keyed by the explicit `Subgame::info_key` (hand + history)
    /// so it validates against the explicit-deal oracle.
    pub strategy: HashMap<u64, Vec<f64>>,
    /// Information sets emitted (hand × public decision node, nonzero reach).
    pub info_sets: usize,
    /// Public decision nodes (the betting tree size — *independent* of range
    /// breadth, the whole point of the vectorized form).
    pub public_nodes: usize,
}


/// The vectorized public-tree solver.
pub struct VectorCfr {
    kinds: Vec<NodeKind>,
    /// Per-decision-node regret/strategy blocks.  Each is written by exactly
    /// ONE node, so the locks are permanently uncontended — they exist to let
    /// the traversal hand disjoint stores to concurrent subtree tasks without
    /// `unsafe` or index rebasing, not to arbitrate real contention.
    stores: Vec<Mutex<NodeStore>>,
    root: usize,
    reach0: [f64; NUM_COMBOS],
    reach1: [f64; NUM_COMBOS],
    board: [u8; 5],
    /// Pre-sorted river runouts for a turn resolve (`None` for a complete-board
    /// river resolve, which has no `RunoutShowdown` leaves).  Built once and
    /// shared across every iteration and every turn leaf.
    runout: Option<PreparedRunout>,
    /// Pre-sorted complete boards for `Showdown` leaves: one entry for a river
    /// resolve, one per live river card for a full-river turn resolve.  Built
    /// once at construction so no iteration ever re-sorts a board.
    prepared: Vec<PreparedShowdown>,
    /// Solve the river betting exactly inside a turn resolve (a `Chance` node
    /// per street close) instead of cutting at the reveal with a check-down /
    /// continuation leaf.  Only affects turn roots; a flop root still cuts at
    /// the turn reveal, and all-in run-outs stay exact check-downs either way.
    full_river: bool,
    /// `combo_cards(h)` for every `h` — the chance-mask hot path decodes each
    /// combo per branch per iteration, so it is precomputed once.
    cards: Vec<[u8; 2]>,
    big_blind: f64,
    /// Maximum raises in the subgame (`u32::MAX` = the engine's unbounded
    /// re-raising).  Deep-stacked resolving needs a finite cap for the same
    /// reason the blueprint does: geometric raise chains blow the tree up.
    raise_cap: u32,
    /// Rest-of-hand pot scales for the depth-limit continuation choice (finding
    /// #1); `[0.0]` (length 1) is the plain single check-down (no chooser node).
    /// `scales[0]` should be `0.0` (the normal continuation).  Mirrors
    /// [`MultiContinuationLeaf`](crate::validation::resolving::leaf_eval).
    scales: Vec<f64>,
    /// The fixed continuation chooser — the opponent of the resolve-root actor,
    /// whose post-leaf adaptation the resolve must be robust to (only used when
    /// `scales.len() > 1`).
    chooser: usize,
    /// The betting-tree root *before* any gadget wrap (`== root` without one);
    /// CFV extraction always evaluates from here.
    inner_root: usize,
    /// Carried opponent CFVs (bb, per opponent hand in `features::combo_index`
    /// order) when this is a gadget-constrained continual resolve.
    carried: Option<Box<[f64; NUM_COMBOS]>>,
    /// Recycled per-hand traversal buffers — see [`solve::Scratch`].  Held on
    /// the solver (not per `run` call) so the pool stays warm across iterations.
    scratch: solve::Scratch,
    t: u64,
}

impl VectorCfr {

    /// Build the public tree rooted at `root` (a river *or* turn public state)
    /// over the two belief ranges.  Solved with CFR⁺ (RM⁺ + linear averaging).
    pub fn new(root: &GameState, beliefs: &[BeliefState]) -> Self {
        Self::new_capped(root, beliefs, u32::MAX)
    }

    /// [`new`](Self::new) with the subgame's aggression bounded at `raise_cap`
    /// raises: past the cap voluntary aggression (raise / voluntary all-in) is
    /// pruned, exactly mirroring the blueprint's betting abstraction
    /// (`BlueprintHoldem::capped_legal`) — a forced all-in call always stays.
    pub fn new_capped(root: &GameState, beliefs: &[BeliefState], raise_cap: u32) -> Self {
        Self::new_capped_multi(root, beliefs, raise_cap, vec![0.0])
    }

    /// [`new_capped`](Self::new_capped) with a **multi-valued depth-limit leaf**
    /// at each turn/flop depth cut the opponent picks among the
    /// `scales` continuations (rest-of-hand pot inflations), so the resolve is
    /// robust to the opponent adapting past the leaf rather than overfitting one
    /// check-down.  `scales[0]` should be `0.0` (the normal check-down); a single
    /// `[0.0]` reproduces [`new_capped`](Self::new_capped) with no chooser nodes.
    pub fn new_capped_multi(
        root: &GameState,
        beliefs: &[BeliefState],
        raise_cap: u32,
        scales: Vec<f64>,
    ) -> Self {
        Self::new_full(root, beliefs, raise_cap, scales, false)
    }

    /// The general constructor.  With `full_river` (turn roots only), a turn
    /// street close deals the river as an explicit `NodeKind::Chance` and
    /// solves the **real river betting** below it — no leaf model at all on
    /// that boundary — instead of cutting with a check-down / continuation
    /// leaf.  All-in run-outs (no betting left) stay exact check-downs, and a
    /// flop root still cuts at the turn reveal with the `scales` leaf.
    pub fn new_full(
        root: &GameState,
        beliefs: &[BeliefState],
        raise_cap: u32,
        scales: Vec<f64>,
        full_river: bool,
    ) -> Self {
        assert_eq!(beliefs.len(), 2, "heads-up vectorized resolving needs two ranges");
        assert!(!scales.is_empty(), "need at least one continuation");
        let board = root.board;
        let chooser = 1 - root.current_player();
        let big_blind = root.big_blind as f64;

        // Initial reaches = belief marginals with board cards removed (the
        // explicit deal enumeration drops board-conflicting hands the same way).
        let mut board_mask = 0u64;
        for &c in &board {
            if c != NO_CARD {
                board_mask |= 1 << c;
            }
        }
        // Reach in features ordering: slot `features::combo_index(a,b)` holds the
        // belief probability for cards (a,b), looked up via `BeliefState::prob`
        // (which uses its own ordering internally).
        let seed = |b: &BeliefState| {
            let mut r = [0.0; NUM_COMBOS];
            for (i, slot) in r.iter_mut().enumerate() {
                let [a, c] = combo_cards(i);
                if board_mask & (1 << a) == 0 && board_mask & (1 << c) == 0 {
                    *slot = b.prob(a, c);
                }
            }
            r
        };
        let reach0 = seed(&beliefs[0]);
        let reach1 = seed(&beliefs[1]);

        // A turn root (river slot undealt) needs the runout table for its leaves;
        // a complete-board river root has none.
        let runout = board.contains(&NO_CARD).then(|| PreparedRunout::new(board));

        let mut me = Self {
            kinds: Vec::new(),
            stores: Vec::new(),
            root: 0,
            reach0,
            reach1,
            board,
            runout,
            prepared: Vec::new(),
            full_river,
            cards: (0..NUM_COMBOS).map(combo_cards).collect(),
            big_blind,
            raise_cap,
            scales,
            chooser,
            inner_root: 0,
            carried: None,
            scratch: solve::Scratch::default(),
            t: 0,
        };
        // A complete-board root shares one prepared showdown; incomplete roots
        // get one per dealt river card (full-river mode) or none (leaf cuts).
        let root_prep = if board.contains(&NO_CARD) {
            usize::MAX
        } else {
            me.prepared.push(PreparedShowdown::new(board));
            0
        };
        me.root = me.build(root.clone(), Vec::new(), 0, root_prep);
        me.inner_root = me.root;
        me
    }

    /// Constrain this resolve with the opponent's **carried counterfactual
    /// values** (continual re-solving): the root is wrapped in the re-solving
    /// gadget, a per-hand Follow/Terminate choice for the opponent whose
    /// Terminate banks `cfvs[hand]` (bb).  The solved strategy is then *safe*:
    /// the opponent's best response cannot exceed its carried guarantee, no
    /// matter that our strategy was recomputed since.  Extract fresh values
    /// after solving with [`opponent_cfvs`](Self::opponent_cfvs).
    pub fn with_opponent_gadget(mut self, cfvs: [f64; NUM_COMBOS]) -> Self {
        assert!(self.carried.is_none(), "gadget already applied");
        let term = self.kinds.len();
        self.kinds.push(NodeKind::CfvTerminal);
        let store = self.stores.len();
        self.stores.push(Mutex::new(NodeStore::new(2)));
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Decision {
            player: self.chooser,
            store,
            // Action 0 = Terminate (bank the carry), action 1 = Follow.
            children: vec![term, self.root],
            board: self.board,
            history: Vec::new(),
            marker: MARKER_GADGET,
        });
        self.root = id;
        self.carried = Some(Box::new(cfvs));
        self
    }
}

/// Convenience: build, run, and emit in one call (mirrors `SubgameSolver::solve_for_iters`).
pub fn solve_vectorized(root: &GameState, beliefs: &[BeliefState], iters: u64) -> VectorResolved {
    let mut solver = VectorCfr::new(root, beliefs);
    solver.run(iters);
    solver.into_resolved()
}

/// [`solve_vectorized`] with the subgame's aggression bounded at `raise_cap`
/// raises — required for deep-stacked play-time resolving, where the unbounded
/// re-raise chain makes the public tree explode (see [`VectorCfr::new_capped`]).
pub fn solve_vectorized_capped(
    root: &GameState,
    beliefs: &[BeliefState],
    iters: u64,
    raise_cap: u32,
) -> VectorResolved {
    let mut solver = VectorCfr::new_capped(root, beliefs, raise_cap);
    solver.run(iters);
    solver.into_resolved()
}

/// [`solve_vectorized_capped`] with a **multi-valued depth-limit leaf** (finding
/// #1): the opponent picks among the `scales` continuations at each turn/flop
/// depth cut, making the resolve robust to post-leaf adaptation.  `scales[0]`
/// should be `0.0`; `[0.0]` reproduces [`solve_vectorized_capped`].
pub fn solve_vectorized_multi(
    root: &GameState,
    beliefs: &[BeliefState],
    iters: u64,
    raise_cap: u32,
    scales: Vec<f64>,
) -> VectorResolved {
    let mut solver = VectorCfr::new_capped_multi(root, beliefs, raise_cap, scales);
    solver.run(iters);
    solver.into_resolved()
}

/// A **full-river turn resolve**: the river is dealt as an explicit chance
/// node and the real river betting is solved below it — exact to showdown, no
/// leaf model on the turn/river boundary (all-in run-outs are exact
/// check-downs).  ~48× the tree of a leaf-cut turn resolve; budget iterations
/// accordingly.  For turn roots; other roots behave like
/// [`solve_vectorized_capped`] (`scales` still applies to a flop root's cut).
pub fn solve_vectorized_full_river(
    root: &GameState,
    beliefs: &[BeliefState],
    iters: u64,
    raise_cap: u32,
) -> VectorResolved {
    let mut solver = VectorCfr::new_full(root, beliefs, raise_cap, vec![0.0], true);
    solver.run(iters);
    solver.into_resolved()
}

/// The action menu at the **root** of a capped vectorized subgame —
/// index-aligned with the per-hand distributions [`solve_vectorized_capped`]
/// emits at the root key (empty history).  Play-time resolving samples an
/// index from the resolved distribution and looks the concrete action up here.
pub fn capped_root_actions(root: &GameState, raise_cap: u32) -> Vec<poker_core::action::Action> {
    use poker_core::action::Action;
    let full = legal_actions(root);
    if raise_cap > 0 {
        return full.to_vec();
    }
    let has_passive = full.iter().any(|a| matches!(a, Action::Check | Action::Call));
    full.iter()
        .copied()
        .filter(|a| !(matches!(a, Action::Raise(_)) || (matches!(a, Action::AllIn) && has_passive)))
        .collect()
}
