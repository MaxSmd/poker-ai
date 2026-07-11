//! Vectorized range-vs-range subgame solving (finding #2).
//!
//! The explicit-deal [`Subgame`](crate::resolving::subgame::Subgame) enumerates
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
//! [`exploitability`](crate::solver::best_response::exploitability) scores the
//! vectorized result inside the explicit game and the two must agree.
//!
//! **Scope.** This solves *complete-board* (river) subgames — exactly the
//! 1326×1326 blow-up the finding targets — with exact showdown and fold
//! terminals, and *turn* subgames with the river depth-cut: any node that
//! reaches the undealt river (the turn betting closing, or a turn all-in) is a
//! leaf whose check-down showdown is averaged over the 44 live river runouts
//! (`board_runout_cfvs`).  That average reproduces the explicit oracle's
//! `CheckdownLeafEval` value exactly, so the same exploitability cross-check
//! validates turn and flop resolves.  The K > 1 multi-continuation leaf of
//! finding #1 is the remaining follow-up.

use std::collections::HashMap;

use poker_core::legal_actions;
use poker_core::state::{GameState, NO_CARD};

// `board_cfvs` indexes its reach/output by `features::combo_index`, so the whole
// solver works in that ordering (NOT `belief_state`'s, a different bijection).
use crate::abstraction::features::{board_cfvs, combo_cards, PreparedRunout};
use crate::resolving::belief_state::{BeliefState, NUM_COMBOS};
use crate::util::hash::fnv1a;

/// A solved vectorized subgame (mirrors [`Resolved`](crate::resolving::subgame::Resolved)).
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

/// A terminal or decision node of the public betting tree.
enum NodeKind {
    /// River showdown: each player's value is `board_cfvs` over the opponent's
    /// reach with `half_pot` chips at stake (already in big-blind units).
    Showdown { half_pot: f64 },
    /// Turn/flop depth-limit or all-in leaf: the board is incomplete, so the
    /// showdown is the check-down value averaged over the runout
    /// (`board_runout_cfvs`).  The vectorized analogue of the explicit oracle's
    /// `CheckdownLeafEval`.
    RunoutShowdown { half_pot: f64 },
    /// A fold terminal: card-independent per-player net payoff (bb units),
    /// weighted at solve time by the blocker-corrected opponent reach.
    Fold { payoffs: [f64; 2] },
    /// A betting decision for `player`; `children[a]` is the node after legal
    /// action `a`.  `store` indexes the regret/strategy arrays; `board`/`history`
    /// reproduce the explicit info key when emitting the strategy.
    ///
    /// When `is_continuation`, this is not a betting node but the opponent's
    /// depth-limit **continuation choice** (finding #1): `player` is the fixed
    /// chooser, `children[i]` a `RunoutShowdown` at the `i`-th continuation's
    /// inflated pot, and the emitted key carries the `0xFE` marker so it matches
    /// the explicit oracle's continuation info set (and never a betting one).
    Decision {
        player: usize,
        store: usize,
        children: Vec<usize>,
        board: [u8; 5],
        history: Vec<u8>,
        is_continuation: bool,
    },
}

/// Per-decision-node regret and strategy-sum, one row of `num_actions` per hand.
struct NodeStore {
    num_actions: usize,
    regret: Vec<f64>,      // NUM_COMBOS × num_actions
    strategy_sum: Vec<f64>, // NUM_COMBOS × num_actions
}

impl NodeStore {
    fn new(num_actions: usize) -> Self {
        Self {
            num_actions,
            regret: vec![0.0; NUM_COMBOS * num_actions],
            strategy_sum: vec![0.0; NUM_COMBOS * num_actions],
        }
    }

    /// Regret-matched current strategy, all hands, row-major `[hand][action]`.
    fn strategy(&self) -> Vec<f64> {
        let a = self.num_actions;
        let mut out = vec![0.0; NUM_COMBOS * a];
        for h in 0..NUM_COMBOS {
            let row = &self.regret[h * a..h * a + a];
            let pos: f64 = row.iter().map(|&r| r.max(0.0)).sum();
            let dst = &mut out[h * a..h * a + a];
            if pos > 0.0 {
                for (o, &r) in dst.iter_mut().zip(row) {
                    *o = r.max(0.0) / pos;
                }
            } else {
                dst.fill(1.0 / a as f64);
            }
        }
        out
    }

    /// CFR⁺ / RM⁺ update for this node's player: add the instantaneous
    /// counterfactual regret `child_p[a] − v_p` and **floor the accumulated regret
    /// at 0**, then accumulate the linearly-weighted (`weight = t`) reach-weighted
    /// strategy.  This mirrors [`PredictiveSolver`](crate::solver::predictive),
    /// which the resolver defaults to: RM⁺'s non-negativity keeps low-reach
    /// information sets responsive (DCFR's signed, discounted regret froze them at
    /// uniform once the opponent's strategy went pure — exploitable off-path).
    fn update(&mut self, sigma: &[f64], child_p: &[Vec<f64>], v_p: &[f64], reach_p: &[f64; NUM_COMBOS], t: u64) {
        let a = self.num_actions;
        let weight = t as f64; // linear averaging
        for h in 0..NUM_COMBOS {
            let rp = reach_p[h];
            for (ai, cp) in child_p.iter().enumerate().take(a) {
                let idx = h * a + ai;
                let r = &mut self.regret[idx];
                *r = (*r + cp[h] - v_p[h]).max(0.0);
                self.strategy_sum[idx] += weight * rp * sigma[idx];
            }
        }
    }
}

/// The vectorized public-tree solver.
pub struct VectorCfr {
    kinds: Vec<NodeKind>,
    stores: Vec<NodeStore>,
    root: usize,
    reach0: [f64; NUM_COMBOS],
    reach1: [f64; NUM_COMBOS],
    board: [u8; 5],
    /// Pre-sorted river runouts for a turn resolve (`None` for a complete-board
    /// river resolve, which has no `RunoutShowdown` leaves).  Built once and
    /// shared across every iteration and every turn leaf.
    runout: Option<PreparedRunout>,
    big_blind: f64,
    /// Maximum raises in the subgame (`u32::MAX` = the engine's unbounded
    /// re-raising).  Deep-stacked resolving needs a finite cap for the same
    /// reason the blueprint does: geometric raise chains blow the tree up.
    raise_cap: u32,
    /// Rest-of-hand pot scales for the depth-limit continuation choice (finding
    /// #1); `[0.0]` (length 1) is the plain single check-down (no chooser node).
    /// `scales[0]` should be `0.0` (the normal continuation).  Mirrors
    /// [`MultiContinuationLeaf`](crate::resolving::leaf_eval).
    scales: Vec<f64>,
    /// The fixed continuation chooser — the opponent of the resolve-root actor,
    /// whose post-leaf adaptation the resolve must be robust to (only used when
    /// `scales.len() > 1`).
    chooser: usize,
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
    /// (finding #1): at each turn/flop depth cut the opponent picks among the
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
            big_blind,
            raise_cap,
            scales,
            chooser,
            t: 0,
        };
        me.root = me.build(root.clone(), Vec::new(), 0);
        me
    }

    /// The subgame's legal actions under the raise cap: past `raises` ≥ cap,
    /// drop every `Raise` and any *voluntary* `AllIn` (one where a passive
    /// action exists) — the same filter as `BlueprintHoldem::capped_legal`.
    fn capped_legal(&self, gs: &GameState, raises: u32) -> Vec<poker_core::action::Action> {
        use poker_core::action::Action;
        let full = legal_actions(gs);
        if raises < self.raise_cap {
            return full.to_vec();
        }
        let has_passive = full.iter().any(|a| matches!(a, Action::Check | Action::Call));
        full.iter()
            .copied()
            .filter(|a| !(matches!(a, Action::Raise(_)) || (matches!(a, Action::AllIn) && has_passive)))
            .collect()
    }

    fn build(&mut self, gs: GameState, history: Vec<u8>, raises: u32) -> usize {
        // A node is a leaf when the hand ends (fold / river showdown) or when the
        // current street wants a board card the resolve root does not have — the
        // depth cut of a turn or flop subgame (or an all-in run-out past it).
        let needs_runout = gs.board[..gs.board_cards_count()].contains(&NO_CARD);
        if gs.is_terminal() || needs_runout {
            let active = (0..gs.num_players as usize).filter(|&i| gs.folded & (1 << i) == 0).count();
            let real_cards = gs.board.iter().filter(|&&c| c != NO_CARD).count();
            let half_pot = (gs.pot as f64 / 2.0) / self.big_blind;
            if active <= 1 {
                // Someone folded: the payoff is board-independent and exact.
                let p = gs.terminal_payoffs();
                let id = self.kinds.len();
                self.kinds.push(NodeKind::Fold {
                    payoffs: [p[0] as f64 / self.big_blind, p[1] as f64 / self.big_blind],
                });
                return id;
            }
            if real_cards == 5 {
                // Complete board: exact river showdown.
                let id = self.kinds.len();
                self.kinds.push(NodeKind::Showdown { half_pot });
                return id;
            }
            // Board undealt (depth cut or all-in run-out): check-down showdown
            // averaged over the runout.  With K > 1 continuations, the opponent
            // first chooses among them (finding #1); otherwise a plain leaf.
            if self.scales.len() > 1 {
                return self.build_continuation_chooser(half_pot, gs.board, history);
            }
            let id = self.kinds.len();
            self.kinds.push(NodeKind::RunoutShowdown { half_pot });
            return id;
        }

        let player = gs.current_player();
        let acts = self.capped_legal(&gs, raises);
        let mut children = Vec::with_capacity(acts.len());
        for (i, &act) in acts.iter().enumerate() {
            let old_bet = gs.current_bet;
            let mut next = gs.clone();
            next.apply_action(act);
            let r = if next.current_bet > old_bet { raises + 1 } else { raises };
            let mut h = history.clone();
            h.push(i as u8);
            children.push(self.build(next, h, r));
        }
        let store = self.stores.len();
        self.stores.push(NodeStore::new(acts.len()));
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Decision {
            player,
            store,
            children,
            board: gs.board,
            history,
            is_continuation: false,
        });
        id
    }

    /// Build the depth-limit **continuation-choice** node (finding #1): a
    /// decision owned by the fixed [`chooser`](Self::chooser) with one action per
    /// `scales` entry, whose `i`-th child is a `RunoutShowdown` at the inflated
    /// pot `half_pot·(1 + scales[i])`.  Inflating a check-down pot by `s` scales
    /// the (chop-relative) showdown value by exactly `1 + s`, so a scaled
    /// `RunoutShowdown` reproduces `MultiContinuationLeaf`'s continuation `i`
    /// without a new node kind.
    fn build_continuation_chooser(&mut self, half_pot: f64, board: [u8; 5], history: Vec<u8>) -> usize {
        let mut children = Vec::with_capacity(self.scales.len());
        for i in 0..self.scales.len() {
            let s = self.scales[i];
            let child = self.kinds.len();
            self.kinds.push(NodeKind::RunoutShowdown { half_pot: half_pot * (1.0 + s) });
            children.push(child);
        }
        let store = self.stores.len();
        self.stores.push(NodeStore::new(children.len()));
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Decision {
            player: self.chooser,
            store,
            children,
            board,
            history,
            is_continuation: true,
        });
        id
    }

    /// Run `iters` vectorized CFR iterations (DCFR, **alternating** traverser:
    /// each iteration updates one player's regrets while the other plays its
    /// current strategy — the standard, robustly-converging scheme).
    pub fn run(&mut self, iters: u64) {
        for _ in 0..iters {
            self.t += 1;
            // Player 0 first (t=1): with every node still uniform, the traverser
            // trains its responses against an opponent that reaches *all* nodes,
            // and RM⁺ locks those regrets in — the off-path robustness CFR⁺ needs.
            let traverser = ((self.t - 1) % 2) as usize;
            let (reach0, reach1) = (self.reach0, self.reach1);
            let (reach_tr, reach_op) = if traverser == 0 { (reach0, reach1) } else { (reach1, reach0) };
            Self::cfr(&self.kinds, &mut self.stores, self.root, &reach_tr, &reach_op, traverser, self.t, &self.board, self.runout.as_ref());
        }
    }

    /// Counterfactual value vector for `traverser` (per traverser hand), given
    /// the traverser's reach `reach_tr` and the opponent's reach `reach_op`.
    /// Regrets/strategy are updated only at the traverser's own decision nodes.
    #[allow(clippy::too_many_arguments)]
    fn cfr(
        kinds: &[NodeKind],
        stores: &mut [NodeStore],
        id: usize,
        reach_tr: &[f64; NUM_COMBOS],
        reach_op: &[f64; NUM_COMBOS],
        traverser: usize,
        t: u64,
        board: &[u8; 5],
        runout: Option<&PreparedRunout>,
    ) -> Vec<f64> {
        match &kinds[id] {
            NodeKind::Showdown { half_pot } => {
                // Traverser's value = reach-weighted showdown over the opponent.
                let mut v = [0.0; NUM_COMBOS];
                board_cfvs(*board, reach_op, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::RunoutShowdown { half_pot } => {
                // Turn leaf: the same reach-weighted showdown, averaged over the
                // undealt river (check-down continuation), via the pre-sorted
                // runout table built once for this board.
                let mut v = [0.0; NUM_COMBOS];
                runout
                    .expect("turn resolve must build a runout table for its leaves")
                    .evaluate(reach_op, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::Fold { payoffs } => {
                let vr = valid_reach(board, reach_op);
                vr.iter().map(|&r| payoffs[traverser] * r).collect()
            }
            NodeKind::Decision { player, store, children, .. } => {
                let a = children.len();
                let sigma = stores[*store].strategy();
                let mut v = vec![0.0; NUM_COMBOS];

                if *player == traverser {
                    // Push the traverser's own reach by σ; collect per-action
                    // counterfactual values to form regrets.
                    let mut child_v: Vec<Vec<f64>> = Vec::with_capacity(a);
                    for (ai, &child) in children.iter().enumerate() {
                        let mut rt = *reach_tr;
                        for h in 0..NUM_COMBOS {
                            rt[h] *= sigma[h * a + ai];
                        }
                        let cv = Self::cfr(kinds, stores, child, &rt, reach_op, traverser, t, board, runout);
                        for h in 0..NUM_COMBOS {
                            v[h] += sigma[h * a + ai] * cv[h];
                        }
                        child_v.push(cv);
                    }
                    stores[*store].update(&sigma, &child_v, &v, reach_tr, t);
                } else {
                    // Opponent node: push the opponent's reach by σ (folding it
                    // into the counterfactual weight) and sum over actions.
                    for (ai, &child) in children.iter().enumerate() {
                        let mut ro = *reach_op;
                        for h in 0..NUM_COMBOS {
                            ro[h] *= sigma[h * a + ai];
                        }
                        let cv = Self::cfr(kinds, stores, child, reach_tr, &ro, traverser, t, board, runout);
                        for h in 0..NUM_COMBOS {
                            v[h] += cv[h];
                        }
                    }
                }
                v
            }
        }
    }

    /// Emit the deployable average strategy keyed by the explicit `info_key`.
    pub fn into_resolved(self) -> VectorResolved {
        let mut strategy = HashMap::new();
        let mut public_nodes = 0;
        for kind in &self.kinds {
            let NodeKind::Decision { player, store, board, history, children, is_continuation } = kind
            else {
                continue;
            };
            public_nodes += 1;
            let a = children.len();
            let s = &self.stores[*store];
            for h in 0..NUM_COMBOS {
                let row = &s.strategy_sum[h * a..h * a + a];
                let total: f64 = row.iter().sum();
                if total <= 0.0 {
                    continue; // unreached hand: defaults to uniform in the oracle
                }
                let key = info_key(*player, combo_cards(h), board, history, *is_continuation);
                strategy.insert(key, row.iter().map(|&x| x / total).collect());
            }
        }
        let info_sets = strategy.len();
        VectorResolved { strategy, info_sets, public_nodes }
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

/// Blocker-corrected opponent reach per hero hand: `total − card[a] − card[b] +
/// reach[h]`, zero for hero hands using a board card.  This is the reach mass of
/// opponents that do **not** share a card with the hero (or the board).
fn valid_reach(board: &[u8; 5], reach: &[f64; NUM_COMBOS]) -> [f64; NUM_COMBOS] {
    let mut board_mask = 0u64;
    for &c in board {
        if c != NO_CARD {
            board_mask |= 1 << c;
        }
    }
    let mut total = 0.0;
    let mut card = [0.0; 52];
    for (i, &r) in reach.iter().enumerate() {
        if r == 0.0 {
            continue;
        }
        let [a, b] = combo_cards(i);
        if board_mask & (1 << a) != 0 || board_mask & (1 << b) != 0 {
            continue;
        }
        total += r;
        card[a as usize] += r;
        card[b as usize] += r;
    }
    let mut out = [0.0; NUM_COMBOS];
    for (i, slot) in out.iter_mut().enumerate() {
        let [a, b] = combo_cards(i);
        if board_mask & (1 << a) != 0 || board_mask & (1 << b) != 0 {
            continue;
        }
        *slot = total - card[a as usize] - card[b as usize] + reach[i];
    }
    out
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

/// The key under which [`VectorResolved::strategy`] stores a hand's
/// distribution (the explicit `Subgame::info_key`).  `hole` must be sorted
/// ascending; `history` is the action-index path from the resolve root
/// (empty at the root itself).  Betting nodes only — continuation-choice nodes
/// (finding #1) carry an extra marker and are never queried by hole+history.
pub fn subgame_info_key(player: usize, hole: [u8; 2], board: &[u8; 5], history: &[u8]) -> u64 {
    info_key(player, hole, board, history, false)
}

/// Reproduce [`Subgame::info_key`](crate::resolving::subgame): FNV-1a of
/// `player`, the (sorted) hole, the visible board, a separator, then the action
/// history.  `combo_cards` already returns `a < b`, matching the sort there.
/// `is_continuation` appends the `0xFE` marker so a depth-limit continuation
/// choice can never collide with a betting info set at the same key.
fn info_key(player: usize, hole: [u8; 2], board: &[u8; 5], history: &[u8], is_continuation: bool) -> u64 {
    let mut bytes = Vec::with_capacity(8 + history.len());
    bytes.push(player as u8);
    bytes.push(hole[0]);
    bytes.push(hole[1]);
    for &c in board {
        if c != NO_CARD {
            bytes.push(c);
        }
    }
    bytes.push(0xFF);
    bytes.extend_from_slice(history);
    if is_continuation {
        bytes.push(0xFE);
    }
    fnv1a(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolving::leaf_eval::CheckdownLeafEval;
    use crate::resolving::subgame::Subgame;
    use crate::solver::best_response::exploitability;
    use poker_core::action::Action;
    use poker_core::make_card;
    use poker_core::state::MAX_PLAYERS;

    fn river_board() -> [u8; 5] {
        // A♣ K♦ 9♥ 4♠ 2♣
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)]
    }

    fn turn_board() -> [u8; 5] {
        // A♣ K♦ 9♥ 4♠ + (river undealt)
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD]
    }

    fn flop_board() -> [u8; 5] {
        // A♣ K♦ 9♥ + (turn, river undealt)
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), NO_CARD, NO_CARD]
    }

    /// A clean heads-up public root reached by checking/calling to `target_street`
    /// (no extra money); holes are placeholders overwritten per hand.
    fn public_root_at(board: [u8; 5], stack: u32, target_street: u8) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
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
            let act = if acts.contains(&Action::Check) { Action::Check } else { Action::Call };
            gs.apply_action(act);
        }
        gs
    }

    /// A clean heads-up river public root (check/call to the river).
    fn public_root(board: [u8; 5], stack: u32) -> GameState {
        public_root_at(board, stack, 3)
    }

    fn duel_ranges() -> [BeliefState; 2] {
        [
            BeliefState::from_hands(&[
                [make_card(12, 1), make_card(12, 2)], // trips (nuts-ish)
                [make_card(6, 0), make_card(5, 0)],   // air
            ]),
            BeliefState::from_hands(&[
                [make_card(8, 0), make_card(8, 1)],   // bluff-catcher
                [make_card(10, 0), make_card(9, 1)],  // weak
            ]),
        ]
    }

    #[test]
    fn vectorized_resolve_agrees_with_explicit_oracle() {
        // The headline #2 cross-check: the vectorized public-tree solver and the
        // explicit-deal Subgame solve the SAME river game.  The vectorized
        // strategy, scored inside the explicit oracle by exact best response,
        // must reach the same near-optimal exploitability.
        let beliefs = duel_ranges();
        let resolved = solve_vectorized(&public_root(river_board(), 20), &beliefs, 1_200);
        assert!(resolved.info_sets > 0, "must emit strategy");

        let leaf = CheckdownLeafEval::new(); // unused on a complete board
        let oracle = Subgame::new(public_root(river_board(), 20), &beliefs, &leaf);
        let expl = exploitability(&oracle, &resolved.strategy);
        println!("vectorized river resolve exploitability (in the explicit oracle): {expl:.5} bb");
        assert!(expl < 0.05, "vectorized resolve should be near-optimal in the oracle game, got {expl}");
    }

    #[test]
    fn vectorized_turn_resolve_agrees_with_explicit_oracle() {
        // Turn resolving: the vectorized public-tree solver cuts at the undealt
        // river and scores each turn leaf by the runout-averaged check-down
        // showdown (`RunoutShowdown`).  The explicit-deal `Subgame` with the
        // `CheckdownLeafEval` cuts the SAME tree at the river with the SAME
        // check-down value, so the vectorized strategy, scored by exact best
        // response inside that oracle, must reach the same near-optimal
        // exploitability — exactly the river cross-check, one street earlier.
        let beliefs = duel_ranges();
        let root = public_root_at(turn_board(), 20, 2);
        assert_eq!(root.street, 2, "root should be on the turn");

        let resolved = solve_vectorized(&root, &beliefs, 1_500);
        assert!(resolved.info_sets > 0, "must emit strategy");

        let leaf = CheckdownLeafEval::new();
        let oracle = Subgame::new(public_root_at(turn_board(), 20, 2), &beliefs, &leaf);
        let expl = exploitability(&oracle, &resolved.strategy);
        println!("vectorized turn resolve exploitability (in the explicit oracle): {expl:.5} bb");
        assert!(expl < 0.05, "vectorized turn resolve should be near-optimal in the oracle game, got {expl}");
    }

    #[test]
    fn k_continuation_inserts_a_chooser_node_owned_by_the_opponent() {
        // Fast structural guard for the finding-#1 wiring (the exploitability
        // cross-check below proves the semantics but is minutes-slow): a K > 1
        // turn resolve must insert, at its depth-limit leaf, a continuation-choice
        // Decision owned by the chooser with one `RunoutShowdown` child per scale
        // at a non-decreasing (inflated) pot — and a K = 1 resolve must not.
        let beliefs = duel_ranges();
        let root = public_root_at(turn_board(), 20, 2);
        let chooser = 1 - root.current_player();
        let scales = vec![0.0, 0.75, 1.5, 3.0];

        let solver = VectorCfr::new_capped_multi(&root, &beliefs, u32::MAX, scales.clone());
        let mut choosers = 0;
        for k in &solver.kinds {
            let NodeKind::Decision { player, children, is_continuation: true, .. } = k else {
                continue;
            };
            choosers += 1;
            assert_eq!(*player, chooser, "the opponent chooses the continuation");
            assert_eq!(children.len(), scales.len(), "one action per continuation");
            let pots: Vec<f64> = children
                .iter()
                .map(|&c| match solver.kinds[c] {
                    NodeKind::RunoutShowdown { half_pot } => half_pot,
                    _ => panic!("a continuation child must be a runout showdown"),
                })
                .collect();
            for w in pots.windows(2) {
                assert!(w[1] > w[0], "each later continuation inflates the pot: {pots:?}");
            }
        }
        assert!(choosers > 0, "a K > 1 turn resolve must contain a continuation-choice node");

        // K = 1 stays a plain leaf — no chooser nodes.
        let single = VectorCfr::new(&root, &beliefs);
        assert!(
            !single.kinds.iter().any(|k| matches!(k, NodeKind::Decision { is_continuation: true, .. })),
            "a single-continuation resolve must not insert chooser nodes"
        );

        // The emitted strategy (chooser nodes included) is valid — a handful of
        // iterations suffices, the strategy-sum normalizes at any count.
        let resolved = solve_vectorized_multi(&root, &beliefs, 20, u32::MAX, scales);
        for probs in resolved.strategy.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
        }
    }

    #[test]
    #[ignore = "K-aware turn resolve + two exact-BR passes over the multi-valued \
                oracle is minutes-slow; k_continuation_inserts_a_chooser_node guards the wiring"]
    fn vectorized_multi_continuation_is_more_robust_than_single() {
        // Finding #1, vectorized: a turn resolve that lets the opponent pick among
        // K continuations at the depth-limit leaf is less exploitable — measured
        // IN the explicit K-continuation oracle by exact BR (which may choose
        // continuations adversarially) — than one resolved assuming a single
        // check-down.  This is the depth-limited-solving headline, and it also
        // proves the vectorized chooser nodes key-match the oracle's (else the K=4
        // resolve's continuation policy would be ignored and buy nothing).
        use crate::resolving::leaf_eval::MultiContinuationLeaf;
        let beliefs = duel_ranges();
        let scales = vec![0.0, 0.75, 1.5, 3.0]; // == MultiContinuationLeaf default
        let root = || public_root_at(turn_board(), 20, 2);

        // A: resolved aware of the K = 4 choice.  B: resolved assuming one.
        let a = solve_vectorized_multi(&root(), &beliefs, 2_000, u32::MAX, scales.clone());
        let b = solve_vectorized(&root(), &beliefs, 2_000);

        // Both scored in the SAME multi-valued oracle (the adapting opponent).
        let leaf = MultiContinuationLeaf::with_scales(scales);
        let game = Subgame::new(root(), &beliefs, &leaf);
        let expl_a = exploitability(&game, &a.strategy);
        let expl_b = exploitability(&game, &b.strategy);
        println!(
            "vectorized multi-valued-leaf robustness — K=4-resolved: {expl_a:.5} bb, single-resolved: {expl_b:.5} bb"
        );
        assert!(
            expl_a < expl_b,
            "the continuation-aware resolve ({expl_a}) must be less exploitable than the naive one ({expl_b})"
        );
    }

    #[test]
    #[ignore = "flop's two-card runout + exact-BR oracle is minutes-slow; \
                the 990-divisor is guarded fast by flop_runout_cfvs_matches_hand_vs_hand_equity"]
    fn vectorized_flop_resolve_agrees_with_explicit_oracle() {
        // Flop resolving: the vectorized solver cuts at the undealt turn and
        // scores each flop leaf by the two-card runout average (`RunoutShowdown`
        // over C(45,2)=990 turn+river completions).  The explicit-deal `Subgame`
        // with `CheckdownLeafEval` cuts the SAME tree at the turn with the SAME
        // check-down-over-runout value, so the vectorized strategy scored by exact
        // best response in that oracle must be near-optimal.  A small stack keeps
        // the uncapped betting tree (and thus the count of expensive runout
        // leaves) modest so the two-card runout stays affordable in a unit test.
        let beliefs = duel_ranges();
        let root = public_root_at(flop_board(), 6, 1);
        assert_eq!(root.street, 1, "root should be on the flop");

        let resolved = solve_vectorized(&root, &beliefs, 600);
        assert!(resolved.info_sets > 0, "must emit strategy");

        let leaf = CheckdownLeafEval::new();
        let oracle = Subgame::new(public_root_at(flop_board(), 6, 1), &beliefs, &leaf);
        let expl = exploitability(&oracle, &resolved.strategy);
        println!("vectorized flop resolve exploitability (in the explicit oracle): {expl:.5} bb");
        assert!(expl < 0.05, "vectorized flop resolve should be near-optimal in the oracle game, got {expl}");
    }

    #[test]
    #[ignore = "throughput demonstration; run with --ignored"]
    fn turn_full_range_solve_is_fast() {
        // The runout table is built once and shared, so a full-range turn resolve
        // (both ranges the whole 1081-combo grid) solves at a play-viable rate —
        // NOT the per-iteration evaluate+sort the naive runout would cost.
        use std::time::Instant;
        let mut b0 = BeliefState::uniform();
        let mut b1 = BeliefState::uniform();
        b0.remove_board(&turn_board());
        b1.remove_board(&turn_board());

        let root = public_root_at(turn_board(), 20, 2);
        let build = Instant::now();
        let mut solver = VectorCfr::new(&root, &[b0, b1]);
        let build_ms = build.elapsed().as_millis();
        let solve = Instant::now();
        solver.run(500);
        let solve_ms = solve.elapsed().as_millis();
        let resolved = solver.into_resolved();
        println!(
            "turn full-range resolve: {} public nodes, {} info sets — build {build_ms} ms, 500 iters {solve_ms} ms",
            resolved.public_nodes, resolved.info_sets
        );
        assert!(resolved.info_sets > 1000, "full ranges yield many per-hand info sets");
    }

    #[test]
    fn single_hand_each_is_solved() {
        // One hand per player ⇒ no reach mixing; a clean check of the core
        // recursion against the explicit oracle (which solves this trivially).
        let beliefs = [
            BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)]]), // trips
            BeliefState::from_hands(&[[make_card(8, 0), make_card(8, 1)]]),   // pair
        ];
        let resolved = solve_vectorized(&public_root(river_board(), 20), &beliefs, 1_000);
        let leaf = CheckdownLeafEval::new();
        let oracle = Subgame::new(public_root(river_board(), 20), &beliefs, &leaf);
        let expl = exploitability(&oracle, &resolved.strategy);
        assert!(expl < 0.05, "single-hand-each should solve cleanly, got {expl}");
    }

    #[test]
    #[ignore = "throughput demonstration over full 1326-combo ranges; run with --ignored"]
    fn vectorized_solves_full_ranges_the_explicit_solver_cannot() {
        // Throughput deliverable: full uniform ranges = ~1081×1081 ≈ 1.1 M deals,
        // which the explicit-deal Subgame cannot enumerate, are solved by walking
        // a tiny PUBLIC tree once with a per-hand value vector.  The public node
        // count is independent of range breadth — the whole point.
        use std::time::Instant;
        let mut b0 = BeliefState::uniform();
        let mut b1 = BeliefState::uniform();
        b0.remove_board(&river_board());
        b1.remove_board(&river_board());

        let start = Instant::now();
        let resolved = solve_vectorized(&public_root(river_board(), 20), &[b0, b1], 200);
        let elapsed = start.elapsed();
        println!(
            "vectorized full-range river: {} public nodes, {} info sets, 200 iters in {:?}",
            resolved.public_nodes, resolved.info_sets, elapsed
        );
        // Tiny public tree, but thousands of per-hand info sets emitted.
        assert!(resolved.public_nodes < 100, "betting tree is small regardless of range breadth");
        assert!(resolved.info_sets > 1000, "full ranges yield many per-hand info sets");
    }

    #[test]
    fn raise_cap_bounds_the_public_tree() {
        // Deep-ish stacks with a small pot: the raise chain is what the cap
        // prunes.  The capped tree must be strictly smaller, still solve to
        // valid distributions, and a generous cap must reproduce the uncapped
        // tree exactly.
        let beliefs = duel_ranges();
        let root = public_root(river_board(), 200);
        let capped = {
            let mut s = VectorCfr::new_capped(&root, &beliefs, 1);
            s.run(200);
            s.into_resolved()
        };
        let uncapped = solve_vectorized(&root, &beliefs, 200);
        assert!(
            capped.public_nodes < uncapped.public_nodes,
            "cap-1 tree ({}) must be smaller than uncapped ({})",
            capped.public_nodes,
            uncapped.public_nodes
        );
        for probs in capped.strategy.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
        let generous = VectorCfr::new_capped(&root, &beliefs, 1_000);
        assert_eq!(
            generous.kinds.len(),
            VectorCfr::new(&root, &beliefs).kinds.len(),
            "a cap the tree never reaches changes nothing"
        );
        // The root menu helper is index-aligned with the built tree's root.
        let acts = capped_root_actions(&root, 1);
        let NodeKind::Decision { children, .. } = &VectorCfr::new_capped(&root, &beliefs, 1).kinds
            [VectorCfr::new_capped(&root, &beliefs, 1).root]
        else {
            panic!("root is a decision node");
        };
        assert_eq!(acts.len(), children.len(), "root menu width matches the tree");
    }

    #[test]
    fn strategies_are_valid_distributions() {
        let beliefs = duel_ranges();
        let resolved = solve_vectorized(&public_root(river_board(), 20), &beliefs, 500);
        for probs in resolved.strategy.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
            assert!(probs.iter().all(|&p| p >= 0.0), "no negative probabilities");
        }
    }
}
