//! External-sampling Monte Carlo CFR (MCCFR).
//!
//! Full-traversal CFR (`super::cfr`) is the correctness oracle but visits the
//! entire tree every iteration — intractable for NLHE, where a single iteration
//! cannot enumerate all opponent hands.  External sampling is the tractable
//! scheme (Lanctot et al., 2009): on each traversal one
//! player is the *traverser* whose every action is explored to update regret,
//! while **chance and the opponents are sampled** from their current
//! distributions.
//!
//! ## Why this module exists separately from `cfr`
//!
//! Sampling introduces variance — the named convergence enemy in multiplayer.
//! This is the regime where Discounted CFR's noise-suppression actually pays
//! off (it did *not* in the variance-free full-traversal setting), and it is
//! the substrate the VR-MCCFR baseline (control variate) layers onto next.
//! Keeping it apart from the validated full-traversal solver makes the variance
//! behavior easy to compare against the exact reference.
//!
//! The current strategy and average strategy are stored exactly as in `cfr`
//! (a `HashMap<info_key, node>`); only the *update rule* differs.

use std::collections::HashMap;
use std::io;
use std::path::Path;


use serde::{Deserialize, Serialize};

use super::cfr::Variant;
use super::pruning::PruningConfig;

use crate::games::Game;
use crate::util::rng::{sample_index, xorshift_next_unit};

/// EMA learning rate for the running baseline.  A single value per
/// (info set, action) is updated toward observed counterfactual values — the
/// "third f32 accumulator" the memory budget reserves for VR-MCCFR.
const BASELINE_RATE: f64 = 0.1;

/// Per-information-set accumulators (mirrors the full-traversal solver, plus the
/// VR-MCCFR baseline).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Node {
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
    /// Running baseline value per action, in the info-set **owner's**
    /// perspective.  Used as a control variate at sampled (opponent) nodes.
    baseline: Vec<f64>,
    /// Previous iteration's instantaneous regret per action — the optimistic
    /// (Farina et al.) momentum term `R += 2·rₜ − r_{t−1}`.  Zero unless
    /// optimistic updates are enabled.
    prev_inst: Vec<f64>,
    /// Consecutive iterations this action's cumulative regret has been below the
    /// RBP threshold θ (Regret-Based Pruning).  Zero unless pruning is enabled.
    consec_below: Vec<u32>,
}

impl Node {
    fn new(num_actions: usize) -> Self {
        Self {
            regret_sum: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
            baseline: vec![0.0; num_actions],
            prev_inst: vec![0.0; num_actions],
            consec_below: vec![0; num_actions],
        }
    }

    /// Regret-matching current strategy.
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

/// An external-sampling MCCFR solver over a game `G`.
pub struct Mccfr<G: Game> {
    game: G,
    variant: Variant,
    use_baseline: bool,
    /// Optimistic (predictive) regret updates: `R += 2·rₜ − r_{t−1}`.
    use_optimistic: bool,
    /// Regret-Based Pruning config plus the absolute iteration at which pruning
    /// turns on (`start_fraction × total_iters`, fixed by `with_pruning`).
    pruning: Option<(PruningConfig, u64)>,
    nodes: HashMap<u64, Node>,
    /// VR-MCCFR baseline for chance nodes, keyed by `Game::chance_key`, one
    /// running value per outcome (player-0 perspective).
    chance_baseline: HashMap<u64, Vec<f64>>,
    rng: u64,
    iterations: u64,
    /// Total node entries visited across all traversals — the metric that makes
    /// RBP's compute saving measurable.
    nodes_visited: u64,
}

impl<G: Game> Mccfr<G> {
    /// Create a solver with a fixed seed (reproducible runs).
    pub fn new(game: G, variant: Variant) -> Self {
        Self::with_seed(game, variant, 0x2545_F491_4F6C_DD1D)
    }

    /// Create a solver with an explicit RNG seed.
    pub fn with_seed(game: G, variant: Variant, seed: u64) -> Self {
        Self {
            game,
            variant,
            use_baseline: false,
            use_optimistic: false,
            pruning: None,
            nodes: HashMap::new(),
            chance_baseline: HashMap::new(),
            rng: seed | 1,
            iterations: 0,
            nodes_visited: 0,
        }
    }

    /// Enable the VR-MCCFR baseline (control variate at sampled nodes).
    pub fn with_baseline(mut self) -> Self {
        self.use_baseline = true;
        self
    }

    /// Enable optimistic (predictive) regret updates: the strategy is matched
    /// against `R + rₜ`, realized by accumulating `R += 2·rₜ − r_{t−1}` (Farina
    /// et al., 2021).  This accelerates **last-iterate** convergence; the
    /// deployed object is the γ-weighted *average*, so the practical gain to the
    /// blueprint is real but smaller than the last-iterate headline, and it
    /// carries no convergence guarantee in multiplayer.
    pub fn with_optimistic(mut self) -> Self {
        self.use_optimistic = true;
        self
    }

    /// Enable Regret-Based Pruning.  `total_iters` is the planned iteration
    /// budget; pruning turns on after `config.start_fraction` of it.  Branches
    /// whose regret stays below θ for K consecutive iterations stop being
    /// traversed, with a periodic full-refresh traversal as the safeguard (see
    /// [`super::pruning`]).
    pub fn with_pruning(mut self, config: PruningConfig, total_iters: u64) -> Self {
        let start = (config.start_fraction * total_iters as f64) as u64;
        self.pruning = Some((config, start));
        self
    }

    /// Write a **resumable checkpoint** of the full solver state to `path`.
    ///
    /// Long training runs (especially the cloud burst on possibly-preempted
    /// spot instances) cannot afford to lose progress to an interruption, so the
    /// loop checkpoints periodically.  Everything needed to continue *as if
    /// uninterrupted* is saved — every info set's regrets, average-strategy sums,
    /// baseline, optimistic and pruning accumulators, the chance baselines, the
    /// RNG state, the iteration counter, and the solver configuration — so a
    /// resumed run is **bit-identical** to one that never stopped.  Only the
    /// `game` is omitted (it is not data; the caller re-supplies it on load).
    ///
    /// The write is **atomic**: bytes go to a temporary file that is then renamed
    /// over `path`, so an interruption mid-write cannot corrupt an existing
    /// checkpoint.  The fields are serialized **by reference**, so saving does not
    /// transiently double the (memory-bound) regret table.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let view = CheckpointRef {
            variant: &self.variant,
            use_baseline: self.use_baseline,
            use_optimistic: self.use_optimistic,
            pruning: &self.pruning,
            nodes: &self.nodes,
            chance_baseline: &self.chance_baseline,
            rng: self.rng,
            iterations: self.iterations,
            nodes_visited: self.nodes_visited,
        };
        // Stream to the temp file — buffering the serialized state first
        // transiently doubles the memory-bound store (see the SoA twin).
        let path = path.as_ref();
        let tmp = path.with_extension("ckpt.tmp");
        let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        bincode::serialize_into(&mut w, &view).map_err(io::Error::other)?;
        std::io::Write::flush(&mut w)?;
        drop(w);
        std::fs::rename(&tmp, path)
    }

    /// Restore a solver from a checkpoint, continuing from exactly where it
    /// stopped.  The `game` is re-supplied (it was not serialized); the
    /// configuration (variant, baseline/optimistic flags, pruning schedule) is
    /// restored from the checkpoint so resumed training behaves identically.
    pub fn load_checkpoint(path: impl AsRef<Path>, game: G) -> io::Result<Self> {
        let r = std::io::BufReader::new(std::fs::File::open(path)?);
        let cp: CheckpointOwned = bincode::deserialize_from(r).map_err(io::Error::other)?;
        Ok(Self {
            game,
            variant: cp.variant,
            use_baseline: cp.use_baseline,
            use_optimistic: cp.use_optimistic,
            pruning: cp.pruning,
            nodes: cp.nodes,
            chance_baseline: cp.chance_baseline,
            rng: cp.rng,
            iterations: cp.iterations,
            nodes_visited: cp.nodes_visited,
        })
    }

    /// Run `iters` MCCFR iterations.  Each iteration traverses once per player so
    /// that every information set receives both regret updates (as traverser)
    /// and average-strategy updates (as opponent).
    pub fn train(&mut self, iters: u64) {
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations;
            let players = self.game.num_players();
            for traverser in 0..players {
                let root = self.game.root();
                self.traverse(&root, traverser, t);
            }
        }
    }

    /// Average strategy per information set — the deployed object.
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

    /// Current (last-iterate) strategy per information set.
    pub fn current_strategy(&self) -> HashMap<u64, Vec<f64>> {
        self.nodes.iter().map(|(&key, node)| (key, node.strategy())).collect()
    }

    pub fn num_info_sets(&self) -> usize {
        self.nodes.len()
    }

    /// Iterations completed so far — needed to know how much more training a
    /// resumed run owes, and to keep DCFR's `t`-weighting consistent.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Total node entries visited across all traversals so far.  With RBP enabled
    /// this grows more slowly than without, which is the saving made measurable.
    pub fn nodes_visited(&self) -> u64 {
        self.nodes_visited
    }

    /// xorshift64* — a small, fast, deterministic PRNG.  Avoids a `rand`
    /// dependency and keeps tests reproducible.
    fn next_unit(&mut self) -> f64 {
        xorshift_next_unit(&mut self.rng)
    }

    /// Sample an index from a probability distribution.
    fn sample(&mut self, probs: &[f64]) -> usize {
        let r = self.next_unit();
        sample_index(probs.iter().copied(), r)
    }

    /// External-sampling traversal returning the sampled counterfactual value of
    /// `state` for `traverser`.  Chance and opponents are sampled; the
    /// traverser's actions are all explored.
    fn traverse(&mut self, state: &G::State, traverser: usize, t: u64) -> f64 {
        self.nodes_visited += 1;
        if self.game.is_terminal(state) {
            return self.game.utility(state, traverser);
        }
        if self.game.is_chance(state) {
            // Non-enumerable chance (a full random deal): sample one child
            // directly.  No per-outcome control variate is possible here — the
            // outcome list, which the chance baseline indexes, does not exist —
            // so this path simply recurses through the sampled child.
            if !self.game.is_chance_enumerable(state) {
                let mut r = self.rng;
                let child = self.game.sample_chance(state, || xorshift_next_unit(&mut r));
                self.rng = r;
                return self.traverse(&child, traverser, t);
            }
            let outcomes = self.game.chance_outcomes(state);
            let idx = self.sample_outcome(&outcomes);
            let v_child = self.traverse(&outcomes[idx].0, traverser, t);
            if !self.use_baseline {
                return v_child;
            }
            // Control-variate correction for the sampled outcome.  Stored in
            // player-0 perspective; convert with the traverser's sign.  Sampling
            // probability equals the outcome probability, so the importance
            // weight is 1 and the corrected value is
            //   Σ_o p(o)·b(o) + (v₀ − b(o*)).
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let ckey = self.game.chance_key(state);
            let n = outcomes.len();
            let base = self.chance_baseline.entry(ckey).or_insert_with(|| vec![0.0; n]);
            let base_exp: f64 = (0..n).map(|i| outcomes[i].1 * base[i]).sum();
            let corrected0 = base_exp + (v0 - base[idx]);
            base[idx] += BASELINE_RATE * (v0 - base[idx]);
            return sgn * corrected0;
        }

        let player = self.game.current_player(state);
        let key = self.game.info_key(state);
        let num_actions = self.game.num_actions(state);
        let strategy = self.nodes.entry(key).or_insert_with(|| Node::new(num_actions)).strategy();

        if player == traverser {
            let consec = self.pruned_streaks(key, t);

            // Explore every (non-pruned) action; accumulate sampled regret.
            let mut util = vec![0.0; num_actions];
            let mut traversed = vec![true; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                let pruned = match (&consec, self.pruning) {
                    (Some(cb), Some((cfg, _))) => cfg.should_prune(cb[a]),
                    _ => false,
                };
                if pruned {
                    // A pruned action has deeply negative regret ⇒ ~zero
                    // regret-matching probability, so it contributes ~nothing to
                    // node_value; its subtree is left untraversed.
                    traversed[a] = false;
                    continue;
                }
                let child = self.game.apply(state, a);
                util[a] = self.traverse(&child, traverser, t);
                node_value += strategy[a] * util[a];
            }
            self.update_regret(key, t, &util, node_value, &traversed);
            self.refresh_traverser_baseline(key, traverser, &util, &traversed);
            node_value
        } else {
            // Opponent: accumulate average strategy, then sample one action.
            self.update_strategy(key, t, &strategy);
            let a = self.sample(&strategy);
            let child = self.game.apply(state, a);
            let v_child = self.traverse(&child, traverser, t);
            self.corrected_opponent_value_serial(key, &strategy, a, v_child, traverser)
        }
    }

    /// RBP gate: on a normal (non-refresh) iteration past the warm-up, the
    /// per-action below-θ streaks that decide pruning at this node; `None`
    /// disables pruning for this visit (a refresh iteration prunes nothing,
    /// re-touching every branch).  Shared by the clone and cursor traversals.
    fn pruned_streaks(&self, key: u64, t: u64) -> Option<Vec<u32>> {
        let (cfg, start) = self.pruning?;
        if t > start && !cfg.is_refresh_iteration(t) {
            self.nodes.get(&key).map(|n| n.consec_below.clone())
        } else {
            None
        }
    }

    /// Refresh the traverser's per-action baseline toward this iteration's
    /// sampled utilities, in player-0 perspective — only for traversed actions
    /// (a pruned action's util is not sampled).  Shared by the clone and cursor
    /// traversals; a no-op with the baseline off.
    fn refresh_traverser_baseline(&mut self, key: u64, traverser: usize, util: &[f64], traversed: &[bool]) {
        if !self.use_baseline {
            return;
        }
        let sgn = Self::sign(traverser);
        let node = self.nodes.get_mut(&key).expect("inserted in traverse");
        for (a, &u) in util.iter().enumerate() {
            if traversed[a] {
                node.baseline[a] += BASELINE_RATE * (sgn * u - node.baseline[a]);
            }
        }
    }

    /// VR-MCCFR control variate at a sampled opponent node, in player-0
    /// perspective (convert with the traverser's sign).  With sampling
    /// probability σ(a) the importance weight is 1, so the corrected value is
    /// `Σ_a σ(a)·b(a) + (v₀ − b(a*))`; the baseline then takes one EMA step
    /// toward the realized value.  Shared by the clone and cursor traversals;
    /// with the baseline off this is just `v_child`.
    fn corrected_opponent_value_serial(
        &mut self,
        key: u64,
        strategy: &[f64],
        a: usize,
        v_child: f64,
        traverser: usize,
    ) -> f64 {
        if !self.use_baseline {
            return v_child;
        }
        let sgn = Self::sign(traverser);
        let v0 = sgn * v_child;
        let (baseline_exp, baseline_a) = {
            let node = self.nodes.get(&key).expect("inserted in traverse");
            let exp: f64 = strategy.iter().zip(&node.baseline).map(|(s, b)| s * b).sum();
            (exp, node.baseline[a])
        };
        let corrected0 = baseline_exp + (v0 - baseline_a);
        let node = self.nodes.get_mut(&key).expect("inserted in traverse");
        node.baseline[a] += BASELINE_RATE * (v0 - node.baseline[a]);
        sgn * corrected0
    }

    /// Sign that converts a value from the traverser's perspective to player 0's
    /// (and back): `+1` for player 0, `−1` for player 1 in a 2-player zero-sum
    /// game.
    fn sign(traverser: usize) -> f64 {
        if traverser == 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn sample_outcome(&mut self, outcomes: &[(G::State, f64)]) -> usize {
        let r = self.next_unit();
        let mut acc = 0.0;
        for (i, &(_, p)) in outcomes.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        outcomes.len() - 1
    }

    /// Accumulate sampled counterfactual regret at the traverser's info set.
    /// External sampling already accounts for counterfactual reach, so the
    /// instantaneous regret is simply `util[a] - node_value` (no reach factor).
    ///
    /// `traversed[a]` is false for actions RBP skipped this iteration: their
    /// regret is frozen (no update, no discount), but their below-θ streak keeps
    /// advancing so they stay pruned.  With optimistic updates on, the increment
    /// is the predictive `2·rₜ − r_{t−1}` instead of `rₜ`.
    fn update_regret(&mut self, key: u64, t: u64, util: &[f64], node_value: f64, traversed: &[bool]) {
        let optimistic = self.use_optimistic;
        let pruning = self.pruning.map(|(cfg, _)| cfg);
        let (pos, neg) = match self.variant {
            Variant::Vanilla => (1.0, 1.0),
            Variant::Dcfr(d) => (d.positive_factor(t), d.negative_factor(t)),
        };
        let discount = matches!(self.variant, Variant::Dcfr(_));

        let node = self.nodes.get_mut(&key).expect("inserted in traverse");
        for a in 0..node.regret_sum.len() {
            if traversed[a] {
                let inst = util[a] - node_value;
                let r = &mut node.regret_sum[a];
                if discount {
                    *r *= if *r > 0.0 { pos } else { neg };
                }
                *r += if optimistic { 2.0 * inst - node.prev_inst[a] } else { inst };
                node.prev_inst[a] = inst;
            }
            // Maintain the RBP below-θ streak from the (possibly frozen) regret.
            if let Some(cfg) = pruning {
                if cfg.below_threshold(node.regret_sum[a]) {
                    node.consec_below[a] += 1;
                } else {
                    node.consec_below[a] = 0;
                }
            }
        }
    }

    /// Accumulate the average strategy at an opponent's info set.
    fn update_strategy(&mut self, key: u64, t: u64, strategy: &[f64]) {
        let weight = match self.variant {
            Variant::Vanilla => 1.0,
            Variant::Dcfr(d) => d.strategy_weight(t),
        };
        let node = self.nodes.get_mut(&key).expect("inserted in traverse");
        for (s, &p) in node.strategy_sum.iter_mut().zip(strategy) {
            *s += weight * p;
        }
    }
}

/// Borrowed view of the solver state for **saving** a checkpoint — serializes
/// the regret table by reference, so writing a checkpoint does not clone (and
/// transiently double) the memory-bound store.
#[derive(Serialize)]
struct CheckpointRef<'a> {
    variant: &'a Variant,
    use_baseline: bool,
    use_optimistic: bool,
    pruning: &'a Option<(PruningConfig, u64)>,
    nodes: &'a HashMap<u64, Node>,
    chance_baseline: &'a HashMap<u64, Vec<f64>>,
    rng: u64,
    iterations: u64,
    nodes_visited: u64,
}

/// Owned mirror of [`CheckpointRef`] for **loading**.  bincode is not
/// self-describing and `&T` serializes identically to `T`, so the two structs
/// share a byte layout as long as their fields stay in the same order.
#[derive(Deserialize)]
struct CheckpointOwned {
    variant: Variant,
    use_baseline: bool,
    use_optimistic: bool,
    pruning: Option<(PruningConfig, u64)>,
    nodes: HashMap<u64, Node>,
    chance_baseline: HashMap<u64, Vec<f64>>,
    rng: u64,
    iterations: u64,
    nodes_visited: u64,
}

mod atomic;
mod cursor;
mod parallel;
mod soa;

pub use soa::{LeanMccfr, SoaMccfr};

#[cfg(test)]
mod tests;
