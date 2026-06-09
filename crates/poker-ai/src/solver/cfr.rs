//! Generic full-traversal Counterfactual Regret Minimization.
//!
//! This is the *correctness* solver: it traverses the entire game tree every
//! iteration (no sampling), so it is exact and deterministic and converges
//! cleanly on small games.  It is the implementation validated against Kuhn and
//! Leduc before the sampled MCCFR variants (external sampling, baselines,
//! pruning) are layered on for full NLHE — those add variance and are pointless
//! to debug until this exact core is proven correct.
//!
//! Two regret-update regimes are supported:
//!
//! * [`Variant::Vanilla`] — textbook CFR: regrets accumulate undiscounted and
//!   the average strategy weights every iteration equally.  Simplest to trust.
//! * [`Variant::Dcfr`] — Discounted CFR with the full `(α, β, γ)` schedule from
//!   [`crate::solver::dcfr`].
//!
//! Storage here is a `HashMap<info_key, Node>` rather than the production
//! Structure-of-Arrays [`crate::solver::regret_table::RegretTable`].  That is a
//! deliberate choice for the validation path: the map is obviously correct and
//! handles dynamic info-set discovery for arbitrary small games.  The SoA table
//! is for the memory-bound NLHE blueprint, where the info-set count is fixed by
//! the abstraction and known up front.

use std::collections::HashMap;

use super::dcfr::Discount;
use crate::games::Game;

/// Which regret-update regime the solver uses.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    /// Undiscounted textbook CFR.
    Vanilla,
    /// Discounted CFR with the given `(α, β, γ)` schedule.
    Dcfr(Discount),
}

/// Per-information-set accumulators.
#[derive(Clone, Debug)]
struct Node {
    /// Cumulative counterfactual regret per action.
    regret_sum: Vec<f64>,
    /// Weighted cumulative strategy per action (numerator of the average).
    strategy_sum: Vec<f64>,
}

impl Node {
    fn new(num_actions: usize) -> Self {
        Self { regret_sum: vec![0.0; num_actions], strategy_sum: vec![0.0; num_actions] }
    }

    /// Current strategy by regret matching: positive regrets normalized, or
    /// uniform when no action has positive regret.
    fn strategy(&self) -> Vec<f64> {
        let positive: Vec<f64> = self.regret_sum.iter().map(|&r| r.max(0.0)).collect();
        let total: f64 = positive.iter().sum();
        let n = self.regret_sum.len();
        if total > 0.0 {
            positive.iter().map(|&p| p / total).collect()
        } else {
            vec![1.0 / n as f64; n]
        }
    }
}

/// A trained CFR solver over a game `G`.
pub struct Cfr<G: Game> {
    game: G,
    variant: Variant,
    nodes: HashMap<u64, Node>,
    iterations: u64,
}

impl<G: Game> Cfr<G> {
    /// Create a solver for `game` using the given update regime.
    pub fn new(game: G, variant: Variant) -> Self {
        Self { game, variant, nodes: HashMap::new(), iterations: 0 }
    }

    /// Run `iters` full-tree CFR iterations.
    pub fn train(&mut self, iters: u64) {
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations;
            let root = self.game.root();
            self.walk(&root, 1.0, 1.0, 1.0, t);
        }
    }

    /// The average strategy per information set — the object that converges to a
    /// Nash equilibrium and is what gets deployed.
    pub fn average_strategy(&self) -> HashMap<u64, Vec<f64>> {
        self.nodes
            .iter()
            .map(|(&key, node)| {
                let total: f64 = node.strategy_sum.iter().sum();
                let probs = if total > 0.0 {
                    node.strategy_sum.iter().map(|&s| s / total).collect()
                } else {
                    let n = node.strategy_sum.len();
                    vec![1.0 / n as f64; n]
                };
                (key, probs)
            })
            .collect()
    }

    /// Number of discovered information sets.
    pub fn num_info_sets(&self) -> usize {
        self.nodes.len()
    }

    /// The *current* (last-iterate) regret-matching strategy per information
    /// set.  Useful for diagnosing whether the regret recursion is converging
    /// independently of the averaging scheme.
    pub fn current_strategy(&self) -> HashMap<u64, Vec<f64>> {
        self.nodes.iter().map(|(&key, node)| (key, node.strategy())).collect()
    }

    /// Recursive CFR traversal.
    ///
    /// `p0`, `p1` are the players' reach probabilities; `pc` is the chance
    /// reach.  Returns the expected utility of `state` **for player 0**.
    fn walk(&mut self, state: &G::State, p0: f64, p1: f64, pc: f64, t: u64) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.utility(state, 0);
        }
        if self.game.is_chance(state) {
            let mut value = 0.0;
            for (child, prob) in self.game.chance_outcomes(state) {
                value += prob * self.walk(&child, p0, p1, pc * prob, t);
            }
            return value;
        }

        let player = self.game.current_player(state);
        let key = self.game.info_key(state);
        let num_actions = self.game.num_actions(state);

        let strategy = self
            .nodes
            .entry(key)
            .or_insert_with(|| Node::new(num_actions))
            .strategy();

        // Recurse into each action, computing per-action utilities for player 0.
        let mut action_util = vec![0.0; num_actions];
        let mut node_util = 0.0;
        for a in 0..num_actions {
            let child = self.game.apply(state, a);
            let (cp0, cp1) = if player == 0 {
                (p0 * strategy[a], p1)
            } else {
                (p0, p1 * strategy[a])
            };
            action_util[a] = self.walk(&child, cp0, cp1, pc, t);
            node_util += strategy[a] * action_util[a];
        }

        // Counterfactual reach (opponent × chance) and the acting player's own
        // reach.  Utilities above are from player 0's perspective, so player 1's
        // regret is computed on the negated utility.
        let (cf_reach, own_reach, sign) = if player == 0 {
            (p1 * pc, p0, 1.0)
        } else {
            (p0 * pc, p1, -1.0)
        };
        let node_util_player = sign * node_util;

        self.update(key, t, cf_reach, own_reach, &strategy, |a| {
            sign * action_util[a] - node_util_player
        });

        node_util
    }

    /// Apply the regret and strategy-sum updates for one info set under the
    /// active variant.  `instantaneous(a)` is the (player-perspective)
    /// instantaneous regret of action `a`.
    fn update(
        &mut self,
        key: u64,
        t: u64,
        cf_reach: f64,
        own_reach: f64,
        strategy: &[f64],
        instantaneous: impl Fn(usize) -> f64,
    ) {
        let node = self.nodes.get_mut(&key).expect("node was inserted in walk");
        match self.variant {
            Variant::Vanilla => {
                for a in 0..node.regret_sum.len() {
                    node.regret_sum[a] += cf_reach * instantaneous(a);
                    node.strategy_sum[a] += own_reach * strategy[a];
                }
            }
            Variant::Dcfr(d) => {
                let pos = d.positive_factor(t);
                let neg = d.negative_factor(t);
                let sw = d.strategy_weight(t);
                for a in 0..node.regret_sum.len() {
                    let r = &mut node.regret_sum[a];
                    *r *= if *r > 0.0 { pos } else { neg };
                    *r += cf_reach * instantaneous(a);
                    node.strategy_sum[a] += sw * own_reach * strategy[a];
                }
            }
        }
    }
}
