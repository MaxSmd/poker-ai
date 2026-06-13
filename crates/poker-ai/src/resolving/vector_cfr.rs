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
//! emits its average strategy under the *same* `info_key` (player + hand + board
//! + history), so [`exploitability`](crate::solver::best_response::exploitability)
//! scores the vectorized result inside the explicit game and the two must agree.
//!
//! **Scope.** This increment solves *complete-board* (river) subgames — exactly
//! the 1326×1326 blow-up the finding targets — with exact showdown and fold
//! terminals.  Depth-limit leaves (turn/flop cuts) carry the multi-continuation
//! leaf values of finding #1 as per-hand vectors; that vectorized leaf (a
//! `board_cfvs` average over runouts behind the opponent's continuation choice)
//! is the follow-up and is rejected here with a clear panic rather than scored
//! wrongly.

use std::collections::HashMap;

use poker_core::legal_actions;
use poker_core::state::{GameState, NO_CARD};

// `board_cfvs` indexes its reach/output by `features::combo_index`, so the whole
// solver works in that ordering (NOT `belief_state`'s, a different bijection).
use crate::abstraction::features::{board_cfvs, combo_cards, combo_index};
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
    /// A fold terminal: card-independent per-player net payoff (bb units),
    /// weighted at solve time by the blocker-corrected opponent reach.
    Fold { payoffs: [f64; 2] },
    /// A betting decision for `player`; `children[a]` is the node after legal
    /// action `a`.  `store` indexes the regret/strategy arrays; `board`/`history`
    /// reproduce the explicit info key when emitting the strategy.
    Decision {
        player: usize,
        store: usize,
        children: Vec<usize>,
        board: [u8; 5],
        history: Vec<u8>,
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
            for ai in 0..a {
                let idx = h * a + ai;
                let r = &mut self.regret[idx];
                *r = (*r + child_p[ai][h] - v_p[h]).max(0.0);
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
    big_blind: f64,
    t: u64,
}

impl VectorCfr {
    /// Build the public tree rooted at `root` (a complete-board public state)
    /// over the two belief ranges.  Solved with CFR⁺ (RM⁺ + linear averaging).
    pub fn new(root: &GameState, beliefs: &[BeliefState]) -> Self {
        assert_eq!(beliefs.len(), 2, "heads-up vectorized resolving needs two ranges");
        let board = root.board;
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

        let mut me = Self {
            kinds: Vec::new(),
            stores: Vec::new(),
            root: 0,
            reach0,
            reach1,
            board,
            big_blind,
            t: 0,
        };
        me.root = me.build(root.clone(), Vec::new());
        me
    }

    fn build(&mut self, gs: GameState, history: Vec<u8>) -> usize {
        if gs.is_terminal() {
            let active = (0..gs.num_players as usize).filter(|&i| gs.folded & (1 << i) == 0).count();
            let id = self.kinds.len();
            if active <= 1 {
                let p = gs.terminal_payoffs();
                self.kinds.push(NodeKind::Fold {
                    payoffs: [p[0] as f64 / self.big_blind, p[1] as f64 / self.big_blind],
                });
            } else {
                self.kinds.push(NodeKind::Showdown {
                    half_pot: (gs.pot as f64 / 2.0) / self.big_blind,
                });
            }
            return id;
        }
        assert!(
            !gs.board[..gs.board_cards_count()].contains(&NO_CARD),
            "vector_cfr currently supports complete-board (river) subgames; \
             depth-limit-leaf vectorization (finding #1 continuations over runouts) is the follow-up"
        );

        let player = gs.current_player();
        let acts = legal_actions(&gs);
        let mut children = Vec::with_capacity(acts.len());
        for (i, &act) in acts.iter().enumerate() {
            let mut next = gs.clone();
            next.apply_action(act);
            let mut h = history.clone();
            h.push(i as u8);
            children.push(self.build(next, h));
        }
        let store = self.stores.len();
        self.stores.push(NodeStore::new(acts.len()));
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Decision { player, store, children, board: gs.board, history });
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
            Self::cfr(&self.kinds, &mut self.stores, self.root, &reach_tr, &reach_op, traverser, self.t, &self.board);
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
    ) -> Vec<f64> {
        match &kinds[id] {
            NodeKind::Showdown { half_pot } => {
                // Traverser's value = reach-weighted showdown over the opponent.
                let mut v = [0.0; NUM_COMBOS];
                board_cfvs(*board, reach_op, *half_pot, &mut v);
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
                        let cv = Self::cfr(kinds, stores, child, &rt, reach_op, traverser, t, board);
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
                        let cv = Self::cfr(kinds, stores, child, reach_tr, &ro, traverser, t, board);
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
            let NodeKind::Decision { player, store, board, history, children } = kind else {
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
                let key = info_key(*player, combo_cards(h), board, history);
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

/// Reproduce [`Subgame::info_key`](crate::resolving::subgame): FNV-1a of
/// `player`, the (sorted) hole, the visible board, a separator, then the action
/// history.  `combo_cards` already returns `a < b`, matching the sort there.
fn info_key(player: usize, hole: [u8; 2], board: &[u8; 5], history: &[u8]) -> u64 {
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

    /// A clean heads-up river public root (check/call to the river, no extra
    /// money), holes are placeholders overwritten per hand.
    fn public_root(board: [u8; 5], stack: u32) -> GameState {
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
        while gs.street < 3 && !gs.is_terminal() {
            let acts = legal_actions(&gs);
            let act = if acts.iter().any(|&a| a == Action::Check) { Action::Check } else { Action::Call };
            gs.apply_action(act);
        }
        gs
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
