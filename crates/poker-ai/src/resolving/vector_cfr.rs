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

// The showdown sweeps index reach/output by `features::combo_index`, so the
// whole solver works in that ordering (NOT `belief_state`'s, a different
// bijection).
use crate::abstraction::features::{combo_cards, PreparedRunout, PreparedShowdown};
use crate::resolving::belief_state::{BeliefState, NUM_COMBOS};
use crate::util::hash::fnv1a;

/// Key-namespace markers for [`NodeKind::Decision`] (see its docs).
const MARKER_NONE: u8 = 0;
const MARKER_CONTINUATION: u8 = 0xFE;
const MARKER_GADGET: u8 = 0xA6;

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
    /// River showdown: each player's value is the reach-weighted sweep over
    /// the opponent with `half_pot` at stake (bb units), on the pre-sorted
    /// complete board `prepared[prep]` (a full-river turn resolve has one
    /// prepared board per river card; a river resolve has exactly one).
    Showdown { half_pot: f64, prep: usize },
    /// Turn/flop depth-limit or all-in leaf: the board is incomplete, so the
    /// showdown is the check-down value averaged over the runout
    /// (`board_runout_cfvs`).  The vectorized analogue of the explicit oracle's
    /// `CheckdownLeafEval`.
    RunoutShowdown { half_pot: f64 },
    /// A fold terminal: card-independent per-player net payoff (bb units),
    /// weighted at solve time by the blocker-corrected opponent reach.
    Fold { payoffs: [f64; 2] },
    /// The river reveal inside a **full-river turn resolve**: one child per
    /// live river card.  Both players' reaches are masked per branch (combos
    /// using the card are impossible) and the sum is divided by 44 — the
    /// exact number of rivers consistent with any two disjoint holdings — so
    /// each hand pair's value is an exact conditional expectation, the same
    /// per-pair convention as `board_runout_cfvs`.
    Chance { children: Vec<(u8, usize)> },
    /// The re-solving gadget's **Terminate** terminal (Burch–Johanson–Bowling
    /// 2014, vectorized): the constrained opponent opts out of the subgame and
    /// banks its carried per-hand counterfactual value instead
    /// ([`VectorCfr::carried`], bb).  The resolver's seat receives the
    /// negation.  Constraining the resolve so Follow can never beat the carry
    /// is what makes re-solving *safe*: the opponent cannot profit from our
    /// strategy having been recomputed since the values were extracted.
    CfvTerminal,
    /// A betting decision for `player`; `children[a]` is the node after legal
    /// action `a`.  `store` indexes the regret/strategy arrays; `board`/`history`
    /// reproduce the explicit info key when emitting the strategy.
    ///
    /// `marker` namespaces the emitted key: `MARKER_NONE` for a betting node;
    /// `MARKER_CONTINUATION` for the opponent's depth-limit **continuation
    /// choice** (finding #1: `player` is the fixed chooser, `children[i]` a
    /// `RunoutShowdown` at the `i`-th continuation's inflated pot, matching the
    /// explicit oracle's continuation info set); `MARKER_GADGET` for the
    /// re-solving gadget's per-hand Follow/Terminate choice (never emitted).
    Decision {
        player: usize,
        store: usize,
        children: Vec<usize>,
        board: [u8; 5],
        history: Vec<u8>,
        marker: u8,
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

    /// The linear-averaged strategy (normalized `strategy_sum`, uniform where
    /// no mass accumulated — matching what `into_resolved` emits), row-major
    /// `[hand][action]`.  Used by the CFV-extraction evaluation pass.
    fn average(&self) -> Vec<f64> {
        let a = self.num_actions;
        let mut out = vec![0.0; NUM_COMBOS * a];
        for h in 0..NUM_COMBOS {
            let row = &self.strategy_sum[h * a..h * a + a];
            let total: f64 = row.iter().sum();
            let dst = &mut out[h * a..h * a + a];
            if total > 0.0 {
                for (o, &s) in dst.iter_mut().zip(row) {
                    *o = s / total;
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

/// The read-only traversal environment — everything `cfr` needs besides the
/// mutable stores, bundled so the recursion's signature stays sane.
struct Env<'a> {
    kinds: &'a [NodeKind],
    board: &'a [u8; 5],
    runout: Option<&'a PreparedRunout>,
    prepared: &'a [PreparedShowdown],
    cards: &'a [[u8; 2]],
    /// Carried opponent CFVs when a gadget wraps the root (`CfvTerminal`).
    carried: Option<&'a [f64; NUM_COMBOS]>,
    /// The constrained opponent (the gadget's owner).
    chooser: usize,
}

/// Values at the gadget's Terminate terminal, for `player`'s hands weighted by
/// the other seat's reach: the owner banks its carried per-hand CFV; the other
/// seat pays it (blocker-corrected inclusion–exclusion over the weighted
/// opponent reach, the same sums as a fold terminal).
fn cfv_terminal_values(env: &Env<'_>, reach_other: &[f64; NUM_COMBOS], player: usize) -> Vec<f64> {
    let cfvs = env.carried.expect("CfvTerminal requires carried CFVs");
    if player == env.chooser {
        let vr = valid_reach(env.board, reach_other);
        cfvs.iter().zip(vr.iter()).map(|(&c, &r)| c * r).collect()
    } else {
        let mut weighted = [0.0f64; NUM_COMBOS];
        for ((w, &r), &c) in weighted.iter_mut().zip(reach_other.iter()).zip(cfvs.iter()) {
            *w = r * c;
        }
        valid_reach(env.board, &weighted).iter().map(|&w| -w).collect()
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
    /// [`MultiContinuationLeaf`](crate::resolving::leaf_eval).
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
        Self::new_full(root, beliefs, raise_cap, scales, false)
    }

    /// The general constructor.  With `full_river` (turn roots only), a turn
    /// street close deals the river as an explicit [`NodeKind::Chance`] and
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
        self.stores.push(NodeStore::new(2));
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

    /// The opponent's per-hand counterfactual values at the (inner) resolve
    /// root under the emitted **average** profile, conditional on each hand
    /// (bb; zero where its prior reach has no mass) — the carry for the next
    /// continual resolve.  Evaluated below any gadget wrap, i.e. assuming
    /// Follow, matching how the explicit `ContinualResolver` refreshes.
    pub fn opponent_cfvs(&self) -> [f64; NUM_COMBOS] {
        let opp = self.chooser;
        let (reach_opp, reach_me) =
            if opp == 0 { (&self.reach0, &self.reach1) } else { (&self.reach1, &self.reach0) };
        let env = Env {
            kinds: &self.kinds,
            board: &self.board,
            runout: self.runout.as_ref(),
            prepared: &self.prepared,
            cards: &self.cards,
            carried: self.carried.as_deref(),
            chooser: self.chooser,
        };
        let raw = Self::eval_average(&env, &self.stores, self.inner_root, reach_me, opp);
        let mass = valid_reach(&self.board, reach_me);
        let mut out = [0.0; NUM_COMBOS];
        for (o, (&v, (&m, &prior))) in
            out.iter_mut().zip(raw.iter().zip(mass.iter().zip(reach_opp.iter())))
        {
            if m > 0.0 && prior > 0.0 {
                *o = v / m;
            }
        }
        out
    }

    /// Expected value per `player` hand under the stored **average** strategy
    /// (both seats), weighted by the other seat's reach — the evaluation
    /// (no-update) counterpart of [`cfr`](Self::cfr) used for CFV extraction.
    fn eval_average(
        env: &Env<'_>,
        stores: &[NodeStore],
        id: usize,
        reach_other: &[f64; NUM_COMBOS],
        player: usize,
    ) -> Vec<f64> {
        match &env.kinds[id] {
            NodeKind::Showdown { half_pot, prep } => {
                let mut v = [0.0; NUM_COMBOS];
                env.prepared[*prep].accumulate(reach_other, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::RunoutShowdown { half_pot } => {
                let mut v = [0.0; NUM_COMBOS];
                env.runout
                    .expect("turn resolve must build a runout table for its leaves")
                    .evaluate(reach_other, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::Fold { payoffs } => {
                let vr = valid_reach(env.board, reach_other);
                vr.iter().map(|&r| payoffs[player] * r).collect()
            }
            NodeKind::CfvTerminal => cfv_terminal_values(env, reach_other, player),
            NodeKind::Chance { children } => {
                let mut v = vec![0.0; NUM_COMBOS];
                for &(c, child) in children {
                    let mut ro = *reach_other;
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] == c || cards[1] == c {
                            ro[h] = 0.0;
                        }
                    }
                    let cv = Self::eval_average(env, stores, child, &ro, player);
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] != c && cards[1] != c {
                            v[h] += cv[h];
                        }
                    }
                }
                for x in &mut v {
                    *x /= 44.0;
                }
                v
            }
            NodeKind::Decision { player: p, store, children, .. } => {
                let a = children.len();
                let sigma = stores[*store].average();
                let mut v = vec![0.0; NUM_COMBOS];
                for (ai, &child) in children.iter().enumerate() {
                    if *p == player {
                        let cv = Self::eval_average(env, stores, child, reach_other, player);
                        for h in 0..NUM_COMBOS {
                            v[h] += sigma[h * a + ai] * cv[h];
                        }
                    } else {
                        let mut ro = *reach_other;
                        for h in 0..NUM_COMBOS {
                            ro[h] *= sigma[h * a + ai];
                        }
                        let cv = Self::eval_average(env, stores, child, &ro, player);
                        for h in 0..NUM_COMBOS {
                            v[h] += cv[h];
                        }
                    }
                }
                v
            }
        }
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

    fn build(&mut self, gs: GameState, history: Vec<u8>, raises: u32, prep: usize) -> usize {
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
                self.kinds.push(NodeKind::Showdown { half_pot, prep });
                return id;
            }
            // Full-river mode, betting still open, only the river missing:
            // deal it as an explicit chance node and solve the real river
            // betting below — the exact replacement for the depth-cut leaf.
            // (A terminal here is an all-in run-out: no betting remains, so
            // the plain runout check-down is already exact.)
            if self.full_river && !gs.is_terminal() && real_cards == 4 {
                return self.build_river_chance(gs, history);
            }
            // Board undealt (depth cut or all-in run-out): check-down showdown
            // averaged over the runout.  With K > 1 continuations, the opponent
            // first chooses among them (finding #1); otherwise a plain leaf.
            // In full-river mode a turn all-in run-out gets NO chooser: with no
            // betting left, the plain check-down is exact and a continuation
            // choice would hand the opponent fictitious post-all-in leverage.
            let exact_runout = self.full_river && real_cards == 4;
            if self.scales.len() > 1 && !exact_runout {
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
            children.push(self.build(next, h, r, prep));
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
            marker: MARKER_NONE,
        });
        id
    }

    /// Deal the river inside a full-river turn resolve: one branch per live
    /// card, each with its own pre-sorted showdown board and the real river
    /// betting tree below.  The action history continues across the reveal
    /// (branches are distinguished by each decision node's own `board`, which
    /// the info key already includes).
    fn build_river_chance(&mut self, gs: GameState, history: Vec<u8>) -> usize {
        let mut used = 0u64;
        for &c in &gs.board[..4] {
            used |= 1 << c;
        }
        let mut children = Vec::with_capacity(48);
        for c in 0..52u8 {
            if used & (1 << c) != 0 {
                continue;
            }
            let mut next = gs.clone();
            next.board[4] = c;
            let prep = self.prepared.len();
            self.prepared.push(PreparedShowdown::new(next.board));
            // The raise counter resets on the new street, mirroring the
            // blueprint's per-street cap semantics.
            children.push((c, self.build(next, history.clone(), 0, prep)));
        }
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Chance { children });
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
            marker: MARKER_CONTINUATION,
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
            let env = Env {
                kinds: &self.kinds,
                board: &self.board,
                runout: self.runout.as_ref(),
                prepared: &self.prepared,
                cards: &self.cards,
                carried: self.carried.as_deref(),
                chooser: self.chooser,
            };
            Self::cfr(&env, &mut self.stores, self.root, &reach_tr, &reach_op, traverser, self.t);
        }
    }

    /// Counterfactual value vector for `traverser` (per traverser hand), given
    /// the traverser's reach `reach_tr` and the opponent's reach `reach_op`.
    /// Regrets/strategy are updated only at the traverser's own decision nodes.
    fn cfr(
        env: &Env<'_>,
        stores: &mut [NodeStore],
        id: usize,
        reach_tr: &[f64; NUM_COMBOS],
        reach_op: &[f64; NUM_COMBOS],
        traverser: usize,
        t: u64,
    ) -> Vec<f64> {
        match &env.kinds[id] {
            NodeKind::Showdown { half_pot, prep } => {
                // Traverser's value = reach-weighted showdown over the opponent,
                // on this leaf's pre-sorted complete board.
                let mut v = [0.0; NUM_COMBOS];
                env.prepared[*prep].accumulate(reach_op, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::RunoutShowdown { half_pot } => {
                // Turn leaf: the same reach-weighted showdown, averaged over the
                // undealt river (check-down continuation), via the pre-sorted
                // runout table built once for this board.
                let mut v = [0.0; NUM_COMBOS];
                env.runout
                    .expect("turn resolve must build a runout table for its leaves")
                    .evaluate(reach_op, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::Fold { payoffs } => {
                let vr = valid_reach(env.board, reach_op);
                vr.iter().map(|&r| payoffs[traverser] * r).collect()
            }
            NodeKind::CfvTerminal => cfv_terminal_values(env, reach_op, traverser),
            NodeKind::Chance { children } => {
                // River reveal: mask both reaches per branch, sum, divide by
                // the per-pair-consistent count (44) — see `NodeKind::Chance`.
                let mut v = vec![0.0; NUM_COMBOS];
                for &(c, child) in children {
                    let mut rt = *reach_tr;
                    let mut ro = *reach_op;
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] == c || cards[1] == c {
                            rt[h] = 0.0;
                            ro[h] = 0.0;
                        }
                    }
                    let cv = Self::cfr(env, stores, child, &rt, &ro, traverser, t);
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] != c && cards[1] != c {
                            v[h] += cv[h];
                        }
                    }
                }
                for x in &mut v {
                    *x /= 44.0;
                }
                v
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
                        let cv = Self::cfr(env, stores, child, &rt, reach_op, traverser, t);
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
                        let cv = Self::cfr(env, stores, child, reach_tr, &ro, traverser, t);
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
            let NodeKind::Decision { player, store, board, history, children, marker } = kind
            else {
                continue;
            };
            if *marker == MARKER_GADGET {
                // The gadget's Follow/Terminate mix is a solving device, not a
                // deployable strategy — never emitted.
                continue;
            }
            public_nodes += 1;
            let a = children.len();
            let s = &self.stores[*store];
            for h in 0..NUM_COMBOS {
                let row = &s.strategy_sum[h * a..h * a + a];
                let total: f64 = row.iter().sum();
                if total <= 0.0 {
                    continue; // unreached hand: defaults to uniform in the oracle
                }
                let key = info_key(*player, combo_cards(h), board, history, *marker);
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
    info_key(player, hole, board, history, MARKER_NONE)
}

/// Reproduce [`Subgame::info_key`](crate::resolving::subgame): FNV-1a of
/// `player`, the (sorted) hole, the visible board, a separator, then the action
/// history.  `combo_cards` already returns `a < b`, matching the sort there.
/// A nonzero `marker` byte is appended so a depth-limit continuation choice
/// (`0xFE`) or gadget choice (`0xA6`) can never collide with a betting info
/// set at the same key.
fn info_key(player: usize, hole: [u8; 2], board: &[u8; 5], history: &[u8], marker: u8) -> u64 {
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
    if marker != MARKER_NONE {
        bytes.push(marker);
    }
    fnv1a(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolving::leaf_eval::CheckdownLeafEval;
    use crate::resolving::subgame::Subgame;
    use crate::solver::best_response::{best_response_value, exploitability, profile_value};
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
            let NodeKind::Decision { player, children, marker: MARKER_CONTINUATION, .. } = k else {
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
            !single.kinds.iter().any(|k| matches!(k, NodeKind::Decision { marker: MARKER_CONTINUATION, .. })),
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

    // ------------------------------------------------------------------
    // Full-river turn resolving (real river betting inside the subgame).
    // ------------------------------------------------------------------

    /// Exact best response in the TRUE turn+river game (real river betting,
    /// no leaf model), over small explicit supports — the independent gate
    /// for the full-river mode.  Support-vector BR: per-BR-hand values, the
    /// profile seat's reach weighted by the resolved strategy (uniform where
    /// it stored nothing), explicit per-pair collision skips at terminals,
    /// direct 7-card rank comparison at showdowns, per-pair river divisor.
    struct TrueBr<'a> {
        strategy: &'a HashMap<u64, Vec<f64>>,
        cap: u32,
        br: usize,
        hands: [&'a [[u8; 2]]; 2],
        big_blind: f64,
    }

    impl TrueBr<'_> {
        /// `(br₀ + br₁)/2` in bb — exploitability of `strategy` in the true
        /// game, deals uniform over non-colliding support pairs.
        fn exploitability(
            root: &GameState,
            hands: [&[[u8; 2]]; 2],
            strategy: &HashMap<u64, Vec<f64>>,
            cap: u32,
        ) -> f64 {
            let mut sum = 0.0;
            for br in 0..2 {
                let me = TrueBr { strategy, cap, br, hands, big_blind: root.big_blind as f64 };
                let opp = 1 - br;
                let reach = vec![1.0f64; hands[opp].len()];
                let v = me.node(&mut root.clone(), &mut Vec::new(), 0, &reach);
                // Normalize by the number of consistent (h, j) pairs.
                let pairs: usize = hands[br]
                    .iter()
                    .map(|h| {
                        hands[opp].iter().filter(|j| !collide(h, j, &root.board)).count()
                    })
                    .sum();
                sum += v.iter().sum::<f64>() / pairs as f64 / me.big_blind;
            }
            sum / 2.0
        }

        fn node(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u32, reach: &[f64]) -> Vec<f64> {
            let acts = test_capped(gs, raises, self.cap);
            let actor = gs.current_player();
            let n = acts.len();
            if actor == self.br {
                let mut out = vec![f64::NEG_INFINITY; self.hands[self.br].len()];
                for (i, &a) in acts.iter().enumerate() {
                    let child = self.descend(gs, hist, raises, reach, a, i);
                    for (o, c) in out.iter_mut().zip(&child) {
                        *o = o.max(*c);
                    }
                }
                out
            } else {
                let mut out = vec![0.0f64; self.hands[self.br].len()];
                for (i, &a) in acts.iter().enumerate() {
                    let mut child_reach = vec![0.0f64; reach.len()];
                    for (j, cr) in child_reach.iter_mut().enumerate() {
                        if reach[j] == 0.0 {
                            continue;
                        }
                        let mut hole = self.hands[actor][j];
                        hole.sort_unstable();
                        let key = subgame_info_key(actor, hole, &gs.board, hist);
                        let sigma = match self.strategy.get(&key) {
                            Some(p) if p.len() == n => p[i],
                            _ => 1.0 / n as f64,
                        };
                        *cr = reach[j] * sigma;
                    }
                    let child = self.descend(gs, hist, raises, &child_reach, a, i);
                    for (o, c) in out.iter_mut().zip(&child) {
                        *o += c;
                    }
                }
                out
            }
        }

        fn descend(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u32, reach: &[f64], act: Action, i: usize) -> Vec<f64> {
            let (old_street, old_bet) = (gs.street, gs.current_bet);
            gs.apply_action(act);
            hist.push(i as u8);
            let new_raises = if gs.street != old_street {
                0
            } else if gs.current_bet > old_bet {
                raises + 1
            } else {
                raises
            };
            let undealt = gs.board[..gs.board_cards_count()].contains(&NO_CARD);
            let out = if gs.is_terminal() {
                if gs.folded != 0 {
                    self.fold_value(gs, reach)
                } else if undealt {
                    self.deal_river(gs, hist, new_raises, reach)
                } else {
                    self.showdown(gs, reach)
                }
            } else if undealt {
                self.deal_river(gs, hist, new_raises, reach)
            } else {
                self.node(gs, hist, new_raises, reach)
            };
            hist.pop();
            gs.undo_action();
            out
        }

        fn deal_river(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u32, reach: &[f64]) -> Vec<f64> {
            let opp = 1 - self.br;
            let mut used = 0u64;
            for &c in &gs.board[..4] {
                used |= 1 << c;
            }
            let mut out = vec![0.0f64; self.hands[self.br].len()];
            for c in 0..52u8 {
                if used & (1 << c) != 0 {
                    continue;
                }
                gs.board[4] = c;
                let mut child_reach = reach.to_vec();
                for (j, cr) in child_reach.iter_mut().enumerate() {
                    let h = self.hands[opp][j];
                    if h[0] == c || h[1] == c {
                        *cr = 0.0;
                    }
                }
                let child = if gs.is_terminal() {
                    self.showdown(gs, &child_reach)
                } else {
                    self.node(gs, hist, raises, &child_reach)
                };
                for (h, (o, cv)) in out.iter_mut().zip(&child).enumerate() {
                    let hb = self.hands[self.br][h];
                    if hb[0] != c && hb[1] != c {
                        *o += cv;
                    }
                }
            }
            gs.board[4] = NO_CARD;
            for o in &mut out {
                *o /= 44.0;
            }
            out
        }

        fn fold_value(&self, gs: &GameState, reach: &[f64]) -> Vec<f64> {
            let folder = if gs.folded & 1 != 0 { 0usize } else { 1 };
            let sign = if folder == self.br { -1.0 } else { 1.0 };
            let amount = gs.total_committed[folder] as f64;
            let opp = 1 - self.br;
            self.hands[self.br]
                .iter()
                .map(|h| {
                    let mut v = 0.0;
                    for (j, &r) in reach.iter().enumerate() {
                        if r != 0.0 && !collide(h, &self.hands[opp][j], &gs.board) {
                            v += sign * amount * r;
                        }
                    }
                    v
                })
                .collect()
        }

        fn showdown(&self, gs: &GameState, reach: &[f64]) -> Vec<f64> {
            use poker_core::lut_eval::evaluate_7_lut;
            let matched = gs.total_committed[0].min(gs.total_committed[1]) as f64;
            let b = &gs.board;
            let opp = 1 - self.br;
            self.hands[self.br]
                .iter()
                .map(|h| {
                    let hr = evaluate_7_lut(&[h[0], h[1], b[0], b[1], b[2], b[3], b[4]]);
                    let mut v = 0.0;
                    for (j, &r) in reach.iter().enumerate() {
                        let jh = self.hands[opp][j];
                        if r == 0.0 || collide(h, &jh, b) {
                            continue;
                        }
                        let jr = evaluate_7_lut(&[jh[0], jh[1], b[0], b[1], b[2], b[3], b[4]]);
                        v += r * matched
                            * if hr > jr {
                                1.0
                            } else if hr < jr {
                                -1.0
                            } else {
                                0.0
                            };
                    }
                    v
                })
                .collect()
        }
    }

    /// Two holdings collide with each other or the visible board.
    fn collide(h: &[u8; 2], j: &[u8; 2], board: &[u8; 5]) -> bool {
        let mut used = 0u64;
        for &c in board {
            if c != NO_CARD {
                used |= 1 << c;
            }
        }
        for &c in h {
            if used & (1 << c) != 0 {
                return true;
            }
            used |= 1 << c;
        }
        j.iter().any(|&c| used & (1 << c) != 0)
    }

    /// Test-local mirror of the solver's raise-cap filter (per-street reset).
    fn test_capped(gs: &GameState, raises: u32, cap: u32) -> Vec<Action> {
        let full = legal_actions(gs);
        if raises < cap {
            return full.to_vec();
        }
        let has_passive = full.iter().any(|a| matches!(a, Action::Check | Action::Call));
        full.iter()
            .copied()
            .filter(|a| !(matches!(a, Action::Raise(_)) || (matches!(a, Action::AllIn) && has_passive)))
            .collect()
    }

    /// Richer-than-duel supports on the turn board (sets, pairs, draws, air)
    /// so the river betting actually matters.
    fn turn_supports() -> ([[u8; 2]; 3], [[u8; 2]; 3]) {
        (
            [
                [make_card(12, 1), make_card(12, 2)], // A♦A♥: top set
                [make_card(11, 2), make_card(10, 2)], // K♥Q♥: top-ish pair
                [make_card(4, 0), make_card(3, 0)],   // 6♣5♣: air
            ],
            [
                [make_card(6, 0), make_card(6, 1)],   // 8♣8♦: bluff-catcher
                [make_card(12, 3), make_card(2, 0)],  // A♠4♣: two pair
                [make_card(9, 0), make_card(8, 1)],   // J♣T♦: gutshot air
            ],
        )
    }

    /// The headline gate for full-river turn resolving: solved WITH the real
    /// river betting, the strategy is near-equilibrium in the TRUE turn+river
    /// game (measured by the independent support-vector exact BR above).  The
    /// leaf-cut resolves cannot even express river play (their keys stop at
    /// the reveal → uniform river in the true game), which is exactly the gap
    /// this mode closes — their true-game exploitability must be clearly worse.
    #[test]
    fn full_river_turn_resolve_is_near_equilibrium_in_the_true_game() {
        let (s0, s1) = turn_supports();
        let beliefs = [BeliefState::from_hands(&s0), BeliefState::from_hands(&s1)];
        let root = public_root_at(turn_board(), 16, 2);
        let cap = 1;

        // 150 iterations already lands well under the bound (600 → 0.0014 bb,
        // bound 0.05 — huge slack); the cut arm only needs to be *worse*,
        // which it is by three orders of magnitude (~1.4 bb: it cannot
        // express river play at all).
        let full = solve_vectorized_full_river(&root, &beliefs, 150, cap);
        let cut = solve_vectorized_capped(&root, &beliefs, 150, cap);
        assert!(
            full.public_nodes > cut.public_nodes,
            "full-river tree must contain the river betting ({} vs {})",
            full.public_nodes,
            cut.public_nodes
        );

        let expl_full = TrueBr::exploitability(&root, [&s0, &s1], &full.strategy, cap);
        let expl_cut = TrueBr::exploitability(&root, [&s0, &s1], &cut.strategy, cap);
        println!("true-game exploitability: full-river {expl_full:.4} bb, leaf-cut {expl_cut:.4} bb");
        assert!(
            expl_full < 0.05,
            "full-river resolve should be near-optimal in the true game, got {expl_full}"
        );
        assert!(
            expl_full < expl_cut,
            "solving the real river betting must beat the leaf cut in the true game \
             ({expl_full} vs {expl_cut})"
        );
    }

    // ------------------------------------------------------------------
    // Continual re-solving: the vectorized CFV gadget.
    // ------------------------------------------------------------------

    /// The safety + no-distortion gate for the vectorized gadget (mirrors the
    /// explicit `gadget.rs`/`continual.rs` tests):
    ///
    /// 1. extracted CFVs are consistent — their reach-weighted mean equals the
    ///    profile's value for the opponent in the explicit oracle;
    /// 2. re-solving the same spot constrained by those CFVs leaves the
    ///    opponent's exact best response vs our deployed strategy no better
    ///    than it was against the bootstrap (re-entry stays safe);
    /// 3. our own deployed strategy stays near-optimal (feeding
    ///    near-equilibrium CFVs does not distort the resolve).
    #[test]
    fn gadget_resolve_is_safe_and_true_cfvs_do_not_distort() {
        let beliefs = duel_ranges();
        let root = public_root(river_board(), 20);

        let mut boot = VectorCfr::new(&root, &beliefs);
        boot.run(1_000);
        let cfvs = boot.opponent_cfvs();
        let me = root.current_player();
        let opp = 1 - me;
        let strat_a = boot.into_resolved().strategy;

        // (1) CFV consistency against the explicit oracle's profile value.
        let leaf = CheckdownLeafEval::new();
        let oracle = Subgame::new(root.clone(), &beliefs, &leaf);
        let opp_reach = if opp == 0 { &beliefs[0] } else { &beliefs[1] };
        let mut prior = [0.0f64; NUM_COMBOS];
        for (i, p) in prior.iter_mut().enumerate() {
            let [a, b] = combo_cards(i);
            *p = opp_reach.prob(a, b);
        }
        let me_reach = if me == 0 { &beliefs[0] } else { &beliefs[1] };
        let mut me_prior = [0.0f64; NUM_COMBOS];
        for (i, p) in me_prior.iter_mut().enumerate() {
            let [a, b] = combo_cards(i);
            *p = me_reach.prob(a, b);
        }
        let mass = valid_reach(&root.board, &me_prior);
        let (mut num, mut den) = (0.0, 0.0);
        for j in 0..NUM_COMBOS {
            let w = prior[j] * mass[j];
            num += w * cfvs[j];
            den += w;
        }
        let mean_cfv = num / den;
        let pv = profile_value(&oracle, &strat_a, opp);
        assert!(
            (mean_cfv - pv).abs() < 0.02,
            "reach-weighted mean CFV {mean_cfv:.4} should match the oracle profile value {pv:.4}"
        );

        // (2)+(3) Gadget re-solve constrained by the carried values.
        let mut gadget = VectorCfr::new(&root, &beliefs).with_opponent_gadget(cfvs);
        gadget.run(1_000);
        let strat_b = gadget.into_resolved().strategy;

        // Safety AND no-distortion are both measured by the opponent's exact
        // BR against OUR deployed strategy (the Step-27 lesson: full-profile
        // exploitability is spuriously high here, because the opponent's own
        // emitted betting strategy is untrained on hands the gadget would
        // Terminate — junk we never deploy).  No better than the bootstrap =
        // safe; no more than ε worse = the near-equilibrium CFVs did not
        // distort our side of the resolve.
        let br_vs_boot = best_response_value(&oracle, opp, &strat_a);
        let br_vs_gadget = best_response_value(&oracle, opp, &strat_b);
        assert!(
            br_vs_gadget <= br_vs_boot + 0.02,
            "opponent BR vs the gadget re-solve ({br_vs_gadget:.4}) must not beat \
             its value vs the bootstrap ({br_vs_boot:.4})"
        );
        assert!(
            br_vs_gadget >= br_vs_boot - 0.05,
            "gadget resolve distorted our strategy: opp BR collapsed from \
             {br_vs_boot:.4} to {br_vs_gadget:.4} (untrained lines would show here)"
        );
    }

    /// Full-river resolves must emit valid distributions on river betting
    /// nodes too (keys distinguished by each node's own board card).
    #[test]
    fn full_river_strategies_are_valid_distributions() {
        let (s0, s1) = turn_supports();
        let beliefs = [BeliefState::from_hands(&s0), BeliefState::from_hands(&s1)];
        let root = public_root_at(turn_board(), 20, 2);
        let resolved = solve_vectorized_full_river(&root, &beliefs, 50, 1);
        assert!(resolved.info_sets > 0);
        for probs in resolved.strategy.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
            assert!(probs.iter().all(|&p| p >= 0.0), "no negative probabilities");
        }
    }
}
