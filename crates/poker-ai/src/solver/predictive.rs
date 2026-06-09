//! CFR⁺ (Regret Matching⁺) — the resolver's fast full-traversal subgame solver.
//!
//! This is the v3 "solver change": the blueprint loop stays on sampled DCFR, but
//! a depth-limited *subgame* is a near-two-player, full-traversal problem once
//! folds collapse the active set, and there CFR⁺ converges far faster than
//! vanilla CFR — with strong **last-iterate** behavior, so a 2–5 s resolving
//! budget buys a well-resolved strategy rather than a half-solved one.  Keeping
//! it in its own module enforces the plan's rule that this regime must not leak
//! into the sampled blueprint loop (where variance, not solver speed, binds).
//!
//! ## The update (RM⁺ / CFR⁺)
//!
//! ```text
//!   σ_t(a)  ∝  [ Z_{t-1}(a) ]⁺                       (regret matching⁺)
//!   r_t(a)  =  cfreach · (u(a) − u(σ_t))             (instantaneous regret)
//!   Z_t(a)  =  [ Z_{t-1}(a) + r_t(a) ]⁺              (non-negativity floor)
//! ```
//!
//! with **alternating** updates (one regret-updated player per iteration) and
//! linear (`weight = t`) averaging — the configuration CFR⁺ relies on.
//!
//! ## On the predictive (PCFR⁺) layer
//!
//! The plan targets *Predictive* RM⁺ / PCFR⁺, which adds an optimistic term
//! (the previous instantaneous regret) to the strategy.  The naive form was
//! tried here and **diverged/cycled on Leduc** under both alternating and
//! simultaneous updates — a known instability of optimistic methods when the
//! prediction is poorly scaled or stale across an opponent update.  CFR⁺ already
//! delivers the fast, last-iterate convergence the resolver needs, so the
//! predictive layer is deferred until it can be added with proper step-size
//! control and validated to *beat* CFR⁺ here rather than regress it.  The
//! per-info-set `prediction` accumulator is retained for that work.

use std::collections::HashMap;

use crate::games::Game;

/// Per-information-set accumulators.
#[derive(Clone, Debug)]
struct Node {
    /// RM⁺ regret `Z` (kept non-negative).
    regret: Vec<f64>,
    /// Previous iteration's instantaneous regret — reserved for the deferred
    /// predictive (PCFR⁺) layer; unused by the CFR⁺ update.
    prediction: Vec<f64>,
    /// Linearly-weighted cumulative strategy (numerator of the average).
    strategy_sum: Vec<f64>,
}

impl Node {
    fn new(num_actions: usize) -> Self {
        Self {
            regret: vec![0.0; num_actions],
            prediction: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
        }
    }

    /// Regret-matching⁺ strategy `σ ∝ [Z]⁺` (uniform when no action has positive
    /// regret).
    fn strategy(&self) -> Vec<f64> {
        let positive: Vec<f64> = self.regret.iter().map(|&z| z.max(0.0)).collect();
        let total: f64 = positive.iter().sum();
        let n = self.regret.len();
        if total > 0.0 {
            positive.iter().map(|&p| p / total).collect()
        } else {
            vec![1.0 / n as f64; n]
        }
    }
}

/// A CFR⁺ solver over a game `G` (full-traversal, exact).
pub struct PredictiveSolver<G: Game> {
    game: G,
    nodes: HashMap<u64, Node>,
    iterations: u64,
}

impl<G: Game> PredictiveSolver<G> {
    /// Create a solver for `game`.
    pub fn new(game: G) -> Self {
        Self { game, nodes: HashMap::new(), iterations: 0 }
    }

    /// **Warm-start** the regrets from a blueprint strategy (Phase 5 resolving).
    ///
    /// RM⁺ plays `σ ∝ [Z]⁺`, so seeding `Z(a) = π(a) · scale` makes the solver's
    /// *first-iterate* strategy equal to the blueprint `π` at every seeded info
    /// set — the resolver then refines from the blueprint instead of from uniform,
    /// which is the whole point of warm-starting a 2–5 s subgame solve.  `scale`
    /// sets how much prior regret mass the blueprint carries: larger values make
    /// the blueprint stickier (more iterations to move off it), smaller values let
    /// the subgame solver overrule it sooner.  Only keys present in `seed_regrets`
    /// are touched; everything else starts cold (uniform).
    ///
    /// Keys must be in the subgame's own `info_key` space — i.e. the blueprint
    /// must be expressed over the same information sets the subgame exposes (see
    /// [`crate::resolving::warm_start`]).  A seed whose action count does not match
    /// the subgame's at that key is ignored, since it cannot be the same info set.
    pub fn warm_start(&mut self, seed_regrets: HashMap<u64, Vec<f64>>) {
        for (key, regret) in seed_regrets {
            let n = regret.len();
            let node = self.nodes.entry(key).or_insert_with(|| Node::new(n));
            if node.regret.len() != n {
                continue; // not the same information set; leave it cold
            }
            for (z, &r) in node.regret.iter_mut().zip(&regret) {
                *z = r.max(0.0);
            }
        }
    }

    /// Run `iters` full-tree CFR⁺ iterations.
    ///
    /// Updates **alternate** between the players (one regret-updated player per
    /// iteration) — the configuration CFR⁺'s convergence relies on.
    pub fn train(&mut self, iters: u64) {
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations;
            let update_player = ((t - 1) % 2) as usize;
            let root = self.game.root();
            self.walk(&root, 1.0, 1.0, 1.0, t, update_player);
        }
    }

    /// Linearly-weighted average strategy — the converged, deployable object.
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

    /// Current (last-iterate) regret-matching strategy — converges quickly for
    /// CFR⁺, unlike vanilla CFR.
    pub fn current_strategy(&self) -> HashMap<u64, Vec<f64>> {
        self.nodes.iter().map(|(&key, node)| (key, node.strategy())).collect()
    }

    /// Number of discovered information sets.
    pub fn num_info_sets(&self) -> usize {
        self.nodes.len()
    }

    /// Recursive traversal; returns the expected utility of `state` for player 0.
    /// Only `update_player`'s information sets are updated this iteration.
    fn walk(
        &mut self,
        state: &G::State,
        p0: f64,
        p1: f64,
        pc: f64,
        t: u64,
        update_player: usize,
    ) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.utility(state, 0);
        }
        if self.game.is_chance(state) {
            let mut value = 0.0;
            for (child, prob) in self.game.chance_outcomes(state) {
                value += prob * self.walk(&child, p0, p1, pc * prob, t, update_player);
            }
            return value;
        }

        let player = self.game.current_player(state);
        let key = self.game.info_key(state);
        let num_actions = self.game.num_actions(state);

        let strategy = self.nodes.entry(key).or_insert_with(|| Node::new(num_actions)).strategy();

        let mut action_util = vec![0.0; num_actions];
        let mut node_util = 0.0;
        for a in 0..num_actions {
            let child = self.game.apply(state, a);
            let (cp0, cp1) =
                if player == 0 { (p0 * strategy[a], p1) } else { (p0, p1 * strategy[a]) };
            action_util[a] = self.walk(&child, cp0, cp1, pc, t, update_player);
            node_util += strategy[a] * action_util[a];
        }

        if player == update_player {
            let (cf_reach, own_reach, sign) =
                if player == 0 { (p1 * pc, p0, 1.0) } else { (p0 * pc, p1, -1.0) };
            let node_util_player = sign * node_util;

            let node = self.nodes.get_mut(&key).expect("node was inserted in walk");
            let weight = t as f64; // linear averaging
            for a in 0..num_actions {
                // Player-perspective instantaneous counterfactual regret.
                let inst = cf_reach * (sign * action_util[a] - node_util_player);
                node.regret[a] = (node.regret[a] + inst).max(0.0); // RM⁺
                node.prediction[a] = inst; // reserved for the deferred PCFR⁺ layer
                node.strategy_sum[a] += weight * own_reach * strategy[a];
            }
        }

        node_util
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::kuhn::{Kuhn, GAME_VALUE_P0};
    use crate::games::leduc::Leduc;
    use crate::solver::best_response::{exploitability, profile_value};
    use crate::solver::cfr::{Cfr, Variant};

    #[test]
    fn converges_on_kuhn() {
        let mut s = PredictiveSolver::new(Kuhn);
        s.train(5_000);
        // CFR+'s strength is the last iterate; the average converges too but
        // slower at this scale.
        let last = exploitability(&Kuhn, &s.current_strategy());
        assert!(last < 5e-3, "CFR+ Kuhn last-iterate {last} should be < 5e-3");
        let value = profile_value(&Kuhn, &s.average_strategy(), 0);
        assert!((value - GAME_VALUE_P0).abs() < 1e-2, "value {value} near -1/18");
    }

    #[test]
    fn converges_on_leduc() {
        let mut s = PredictiveSolver::new(Leduc);
        s.train(10_000);
        let expl = exploitability(&Leduc, &s.average_strategy());
        assert!(expl < 0.03, "CFR+ Leduc exploitability {expl} should be < 0.03");
    }

    #[test]
    fn last_iterate_converges_on_leduc() {
        // CFR+'s strong last-iterate behavior — the property the resolver leans
        // on within a tight time budget.
        let mut s = PredictiveSolver::new(Leduc);
        s.train(10_000);
        let expl = exploitability(&Leduc, &s.current_strategy());
        assert!(expl < 0.01, "CFR+ Leduc last-iterate {expl} should be < 0.01");
    }

    #[test]
    fn beats_vanilla_cfr_at_a_fixed_budget() {
        // The v3 bet: at an equal iteration budget the resolver's deployable
        // output (CFR+'s last iterate) is closer to equilibrium than vanilla
        // CFR's deployable output (its average) — this is what makes a 2–5 s
        // resolving call worthwhile.
        let budget = 2_000;

        let mut cfr_plus = PredictiveSolver::new(Leduc);
        cfr_plus.train(budget);
        let expl_plus = exploitability(&Leduc, &cfr_plus.current_strategy());

        let mut vanilla = Cfr::new(Leduc, Variant::Vanilla);
        vanilla.train(budget);
        let expl_vanilla = exploitability(&Leduc, &vanilla.average_strategy());

        assert!(
            expl_plus < expl_vanilla,
            "CFR+ last-iterate ({expl_plus}) should beat vanilla CFR average ({expl_vanilla}) at {budget} iters"
        );
    }
}
