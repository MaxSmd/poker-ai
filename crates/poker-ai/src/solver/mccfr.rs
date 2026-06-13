//! External-sampling Monte Carlo CFR (MCCFR).
//!
//! Full-traversal CFR (`super::cfr`) is the correctness oracle but visits the
//! entire tree every iteration — intractable for NLHE, where a single iteration
//! cannot enumerate all opponent hands.  External sampling is the tractable
//! scheme the plan specifies (Lanctot et al., 2009): on each traversal one
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

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::cfr::Variant;
use super::pruning::PruningConfig;
use super::regret_table::RegretTable;
use crate::games::{CursorGame, Game, IndexedGame};
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
    /// carries no convergence guarantee in multiplayer (see the plan's caveat).
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
        let bytes = bincode::serialize(&view).map_err(io::Error::other)?;
        let path = path.as_ref();
        let tmp = path.with_extension("ckpt.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
    }

    /// Restore a solver from a checkpoint, continuing from exactly where it
    /// stopped.  The `game` is re-supplied (it was not serialized); the
    /// configuration (variant, baseline/optimistic flags, pruning schedule) is
    /// restored from the checkpoint so resumed training behaves identically.
    pub fn load_checkpoint(path: impl AsRef<Path>, game: G) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let cp: CheckpointOwned = bincode::deserialize(&bytes).map_err(io::Error::other)?;
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
            // RBP: on a normal (non-refresh) iteration past the warm-up, skip
            // actions whose regret has been below θ for K consecutive iterations.
            // A refresh iteration prunes nothing, re-touching every branch.
            let prune_now = match self.pruning {
                Some((cfg, start)) => t > start && !cfg.is_refresh_iteration(t),
                None => false,
            };
            let consec = if prune_now {
                self.nodes.get(&key).map(|n| n.consec_below.clone())
            } else {
                None
            };

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
            // Refresh the baseline toward these values, in player-0 perspective —
            // only for traversed actions (a pruned action's util is not sampled).
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                let node = self.nodes.get_mut(&key).expect("inserted in traverse");
                for a in 0..num_actions {
                    if traversed[a] {
                        node.baseline[a] += BASELINE_RATE * (sgn * util[a] - node.baseline[a]);
                    }
                }
            }
            node_value
        } else {
            // Opponent: accumulate average strategy, then sample one action.
            self.update_strategy(key, t, &strategy);
            let a = self.sample(&strategy);
            let child = self.game.apply(state, a);
            let v_child = self.traverse(&child, traverser, t);

            if !self.use_baseline {
                return v_child;
            }
            // VR-MCCFR control variate, in player-0 perspective (convert with
            // the traverser's sign).  With sampling probability σ(a) the
            // importance weight is 1, so the corrected value is
            //   Σ_a σ(a)·b(a) + (v₀ − b(a*)).
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let (baseline_exp, baseline_a) = {
                let node = self.nodes.get(&key).expect("inserted in traverse");
                let exp: f64 = (0..num_actions).map(|i| strategy[i] * node.baseline[i]).sum();
                (exp, node.baseline[a])
            };
            let corrected0 = baseline_exp + (v0 - baseline_a);
            {
                let node = self.nodes.get_mut(&key).expect("inserted in traverse");
                node.baseline[a] += BASELINE_RATE * (v0 - node.baseline[a]);
            }
            sgn * corrected0
        }
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
        for a in 0..node.strategy_sum.len() {
            node.strategy_sum[a] += weight * strategy[a];
        }
    }

    // ── Parallel (batched) training ──────────────────────────────────────────
    //
    // The plan's cloud burst needs the CFR loop parallelized "lock-free so the
    // rented cores aren't idle on a memory-bound workload."  The shared regret
    // store makes naive parallel updates a data race, so the scheme here is
    // **mini-batch MCCFR**: a batch of independent iterations runs in parallel,
    // each reading the same start-of-batch strategy snapshot (read-only, hence
    // lock-free) and writing its regret/strategy contributions into a private
    // [`Delta`]; the deltas are then merged into the master store in fixed
    // iteration order (so the result is deterministic for a fixed seed + batch).
    //
    // The trade-off is staleness: within a batch every iteration sees the same
    // strategy, so a large batch is more parallel but converges per-iteration a
    // little slower.  Modest batches recover most of the speed-up with no
    // material convergence loss.
    //
    // The **VR-MCCFR baseline is supported** here (the plan's headline variance
    // lever, which the cloud-burst path needs): workers read the start-of-batch
    // baseline snapshot as the control variate and emit per-(info set, action)
    // target deltas that are merged as EMA steps in iteration order, so the result
    // stays deterministic.  Optimistic and pruning remain serial-only — they
    // compose poorly with batch staleness and were inert on push/fold (Step 15).

    /// Train with `total_iters` iterations, running `batch` iterations in
    /// parallel at a time (mini-batch MCCFR).  Deterministic for a fixed seed and
    /// `batch`.  Use [`train`](Self::train) for the variance-reduction features.
    pub fn train_parallel(&mut self, total_iters: u64, batch: u64)
    where
        G: Sync,
    {
        let batch = batch.max(1);
        let players = self.game.num_players();
        let mut done = 0u64;
        while done < total_iters {
            let this = batch.min(total_iters - done);
            let base = self.iterations; // iterations completed before this batch

            // Parallel: each iteration produces a private delta against the
            // current (read-only) strategy snapshot.  `collect` preserves the
            // iteration order, which the merge below relies on for determinism.
            let deltas: Vec<Delta> = (0..this)
                .into_par_iter()
                .map(|i| {
                    let t = base + i + 1;
                    let mut rng = splitmix(self.rng, t);
                    let mut delta = Delta::default();
                    for traverser in 0..players {
                        let root = self.game.root();
                        self.traverse_ro(&root, traverser, &mut rng, &mut delta, t);
                    }
                    delta
                })
                .collect();

            // Serial merge in iteration order — deterministic regardless of how
            // the threads were scheduled.
            for (i, delta) in deltas.into_iter().enumerate() {
                self.iterations += 1;
                self.apply_delta(delta, base + i as u64 + 1);
            }
            done += this;
        }
    }

    /// Read-only strategy lookup: regret-matching at a known info set, or uniform
    /// if it has not been visited yet.  Used by the parallel traversal, which
    /// must not mutate the shared store.
    fn strategy_ro(&self, key: u64, num_actions: usize) -> Vec<f64> {
        match self.nodes.get(&key) {
            Some(node) => node.strategy(),
            None => vec![1.0 / num_actions as f64; num_actions],
        }
    }

    /// Read-only **snapshot** of an info set's baseline (start-of-batch state),
    /// zeros if unseen — the control variate the parallel traversal reads.
    fn baseline_ro(&self, key: u64, num_actions: usize) -> Vec<f64> {
        match self.nodes.get(&key) {
            Some(node) => node.baseline.clone(),
            None => vec![0.0; num_actions],
        }
    }

    /// External-sampling traversal that **reads** the shared store and writes its
    /// regret / average-strategy contributions into `delta` (the parallel,
    /// lock-free counterpart of [`traverse`](Self::traverse), without the
    /// baseline/optimistic/pruning refinements).
    fn traverse_ro(
        &self,
        state: &G::State,
        traverser: usize,
        rng: &mut u64,
        delta: &mut Delta,
        t: u64,
    ) -> f64 {
        delta.nodes_visited += 1;
        if self.game.is_terminal(state) {
            return self.game.utility(state, traverser);
        }
        if self.game.is_chance(state) {
            if !self.game.is_chance_enumerable(state) {
                let child = self.game.sample_chance(state, || xorshift_next_unit(rng));
                return self.traverse_ro(&child, traverser, rng, delta, t);
            }
            let outcomes = self.game.chance_outcomes(state);
            let idx = sample_index(outcomes.iter().map(|&(_, p)| p), xorshift_next_unit(rng));
            return self.traverse_ro(&outcomes[idx].0, traverser, rng, delta, t);
        }

        let player = self.game.current_player(state);
        let key = self.game.info_key(state);
        let num_actions = self.game.num_actions(state);
        let strategy = self.strategy_ro(key, num_actions);

        if player == traverser {
            let mut util = vec![0.0; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                let child = self.game.apply(state, a);
                util[a] = self.traverse_ro(&child, traverser, rng, delta, t);
                node_value += strategy[a] * util[a];
            }
            let acc = delta.regret_inst.entry(key).or_insert_with(|| vec![0.0; num_actions]);
            for a in 0..num_actions {
                acc[a] += util[a] - node_value;
            }
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                for a in 0..num_actions {
                    record_baseline(delta, key, num_actions, a, sgn * util[a]);
                }
            }
            node_value
        } else {
            let weight = match self.variant {
                Variant::Vanilla => 1.0,
                Variant::Dcfr(d) => d.strategy_weight(t),
            };
            let acc = delta.strat.entry(key).or_insert_with(|| vec![0.0; num_actions]);
            for a in 0..num_actions {
                acc[a] += weight * strategy[a];
            }
            let a = sample_index(strategy.iter().copied(), xorshift_next_unit(rng));
            let child = self.game.apply(state, a);
            let v_child = self.traverse_ro(&child, traverser, rng, delta, t);
            self.corrected_opponent_value(key, &strategy, a, v_child, traverser, delta)
        }
    }

    /// At a sampled opponent node, apply the VR-MCCFR control variate using the
    /// snapshot baseline and record the realized target — shared by both parallel
    /// traversals.  With the baseline off this is just `v_child`.
    fn corrected_opponent_value(
        &self,
        key: u64,
        strategy: &[f64],
        a: usize,
        v_child: f64,
        traverser: usize,
        delta: &mut Delta,
    ) -> f64 {
        if !self.use_baseline {
            return v_child;
        }
        let n = strategy.len();
        let sgn = Self::sign(traverser);
        let v0 = sgn * v_child; // player-0 perspective
        let snap = self.baseline_ro(key, n);
        let baseline_exp: f64 = (0..n).map(|i| strategy[i] * snap[i]).sum();
        let corrected0 = baseline_exp + (v0 - snap[a]);
        record_baseline(delta, key, n, a, v0);
        sgn * corrected0
    }

    /// Merge one iteration's [`Delta`] into the master store, applying the same
    /// per-iteration discount as the serial regret update.
    fn apply_delta(&mut self, delta: Delta, t: u64) {
        let (pos, neg) = match self.variant {
            Variant::Vanilla => (1.0, 1.0),
            Variant::Dcfr(d) => (d.positive_factor(t), d.negative_factor(t)),
        };
        let discount = matches!(self.variant, Variant::Dcfr(_));
        self.nodes_visited += delta.nodes_visited;
        for (key, inst) in delta.regret_inst {
            let node = self.nodes.entry(key).or_insert_with(|| Node::new(inst.len()));
            for a in 0..node.regret_sum.len() {
                let r = &mut node.regret_sum[a];
                if discount {
                    *r *= if *r > 0.0 { pos } else { neg };
                }
                *r += inst[a];
            }
        }
        for (key, s) in delta.strat {
            let node = self.nodes.entry(key).or_insert_with(|| Node::new(s.len()));
            for a in 0..node.strategy_sum.len() {
                node.strategy_sum[a] += s[a];
            }
        }
        // VR-MCCFR baseline: one EMA step toward this iteration's mean target per
        // (info set, action), applied in iteration order so the result is
        // deterministic regardless of thread scheduling.
        if self.use_baseline {
            for (key, sums) in delta.baseline_sum {
                let cnt = &delta.baseline_cnt[&key];
                let node = self.nodes.entry(key).or_insert_with(|| Node::new(sums.len()));
                for a in 0..node.baseline.len() {
                    if cnt[a] > 0 {
                        let mean = sums[a] / cnt[a] as f64;
                        node.baseline[a] += BASELINE_RATE * (mean - node.baseline[a]);
                    }
                }
            }
        }
    }
}

// ── Cursor fast path ─────────────────────────────────────────────────────────
//
// The clone-based [`Game`] trait returns a freshly-allocated child on every
// `apply`.  For the real-mechanics blueprint games that wrap a
// `poker_core::GameState`, that clone drags along a pre-allocated `UndoStack`, so
// the traversal heap-allocates on *every node* — discarding the zero-allocation
// mutate-and-undo design `poker_core` was built for.  These methods are the same
// external-sampling MCCFR as above, but driven through [`CursorGame`]: a single
// `GameState` is walked in place (`apply`/`undo`), the legal-action list is
// computed once per node and held on the stack frame, and information keys are
// folded without a per-node `Vec`.  They reuse every regret/strategy/baseline
// helper unchanged and are **bit-identical** to the clone-based path for a fixed
// seed (proven by the `*_matches_clone_*` tests).
impl<G: Game + CursorGame> Mccfr<G> {
    /// Cursor-based counterpart of [`train`](Self::train): zero per-node
    /// allocation.  Bit-identical to `train` for a fixed seed on a game that
    /// implements both [`Game`] and [`CursorGame`].
    pub fn train_fast(&mut self, iters: u64) {
        let mut cursor = CursorGame::root(&self.game);
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations;
            let players = CursorGame::num_players(&self.game);
            for traverser in 0..players {
                // `cursor` starts (and is left) at the pre-deal root; the chance
                // branch deals it in place and restores it before returning.
                self.traverse_cursor(&mut cursor, traverser, t);
            }
        }
    }

    /// External-sampling traversal over a [`CursorGame`] cursor — the structural
    /// mirror of [`traverse`](Self::traverse), reaching children via
    /// `apply`/`undo` instead of cloning.  Leaves `cursor` exactly as it found it.
    fn traverse_cursor(
        &mut self,
        cursor: &mut <G as CursorGame>::Cursor,
        traverser: usize,
        t: u64,
    ) -> f64 {
        self.nodes_visited += 1;
        if CursorGame::is_terminal(&self.game, cursor) {
            return CursorGame::utility(&self.game, cursor, traverser);
        }
        if CursorGame::is_chance(&self.game, cursor) {
            // Non-enumerable chance (a full random deal): deal in place, recurse,
            // then restore the pre-deal root.  No per-outcome control variate is
            // possible (the outcome list does not exist).
            let mut r = self.rng;
            CursorGame::sample_chance(&self.game, cursor, || xorshift_next_unit(&mut r));
            self.rng = r;
            let v = self.traverse_cursor(cursor, traverser, t);
            CursorGame::undo_chance(&self.game, cursor);
            return v;
        }

        let player = CursorGame::current_player(&self.game, cursor);
        let key = CursorGame::info_key(&self.game, cursor);
        // Legal actions computed once per node and held on this stack frame.
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        let strategy = self.nodes.entry(key).or_insert_with(|| Node::new(num_actions)).strategy();

        if player == traverser {
            let prune_now = match self.pruning {
                Some((cfg, start)) => t > start && !cfg.is_refresh_iteration(t),
                None => false,
            };
            let consec = if prune_now {
                self.nodes.get(&key).map(|n| n.consec_below.clone())
            } else {
                None
            };

            let mut util = vec![0.0; num_actions];
            let mut traversed = vec![true; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                let pruned = match (&consec, self.pruning) {
                    (Some(cb), Some((cfg, _))) => cfg.should_prune(cb[a]),
                    _ => false,
                };
                if pruned {
                    traversed[a] = false;
                    continue;
                }
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse_cursor(cursor, traverser, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            self.update_regret(key, t, &util, node_value, &traversed);
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                let node = self.nodes.get_mut(&key).expect("inserted in traverse");
                for a in 0..num_actions {
                    if traversed[a] {
                        node.baseline[a] += BASELINE_RATE * (sgn * util[a] - node.baseline[a]);
                    }
                }
            }
            node_value
        } else {
            self.update_strategy(key, t, &strategy);
            let a = self.sample(&strategy);
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse_cursor(cursor, traverser, t);
            CursorGame::undo(&self.game, cursor);

            if !self.use_baseline {
                return v_child;
            }
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let (baseline_exp, baseline_a) = {
                let node = self.nodes.get(&key).expect("inserted in traverse");
                let exp: f64 = (0..num_actions).map(|i| strategy[i] * node.baseline[i]).sum();
                (exp, node.baseline[a])
            };
            let corrected0 = baseline_exp + (v0 - baseline_a);
            {
                let node = self.nodes.get_mut(&key).expect("inserted in traverse");
                node.baseline[a] += BASELINE_RATE * (v0 - node.baseline[a]);
            }
            sgn * corrected0
        }
    }

    /// Cursor-based counterpart of [`train_parallel`](Self::train_parallel):
    /// mini-batch MCCFR with each worker walking its own in-place cursor.
    /// Bit-identical to `train_parallel` for a fixed seed and `batch`.
    pub fn train_parallel_fast(&mut self, total_iters: u64, batch: u64)
    where
        G: Sync,
    {
        let batch = batch.max(1);
        let players = CursorGame::num_players(&self.game);
        let mut done = 0u64;
        while done < total_iters {
            let this = batch.min(total_iters - done);
            let base = self.iterations;

            let deltas: Vec<Delta> = (0..this)
                .into_par_iter()
                .map(|i| {
                    let t = base + i + 1;
                    let mut rng = splitmix(self.rng, t);
                    let mut delta = Delta::default();
                    let mut cursor = CursorGame::root(&self.game);
                    for traverser in 0..players {
                        self.traverse_ro_cursor(&mut cursor, traverser, &mut rng, &mut delta, t);
                    }
                    delta
                })
                .collect();

            for (i, delta) in deltas.into_iter().enumerate() {
                self.iterations += 1;
                self.apply_delta(delta, base + i as u64 + 1);
            }
            done += this;
        }
    }

    /// Read-only cursor traversal writing into `delta` — the cursor counterpart
    /// of [`traverse_ro`](Self::traverse_ro).  Leaves `cursor` as it found it.
    fn traverse_ro_cursor(
        &self,
        cursor: &mut <G as CursorGame>::Cursor,
        traverser: usize,
        rng: &mut u64,
        delta: &mut Delta,
        t: u64,
    ) -> f64 {
        delta.nodes_visited += 1;
        if CursorGame::is_terminal(&self.game, cursor) {
            return CursorGame::utility(&self.game, cursor, traverser);
        }
        if CursorGame::is_chance(&self.game, cursor) {
            CursorGame::sample_chance(&self.game, cursor, || xorshift_next_unit(rng));
            let v = self.traverse_ro_cursor(cursor, traverser, rng, delta, t);
            CursorGame::undo_chance(&self.game, cursor);
            return v;
        }

        let player = CursorGame::current_player(&self.game, cursor);
        let key = CursorGame::info_key(&self.game, cursor);
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        let strategy = self.strategy_ro(key, num_actions);

        if player == traverser {
            let mut util = vec![0.0; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse_ro_cursor(cursor, traverser, rng, delta, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            let acc = delta.regret_inst.entry(key).or_insert_with(|| vec![0.0; num_actions]);
            for a in 0..num_actions {
                acc[a] += util[a] - node_value;
            }
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                for a in 0..num_actions {
                    record_baseline(delta, key, num_actions, a, sgn * util[a]);
                }
            }
            node_value
        } else {
            let weight = match self.variant {
                Variant::Vanilla => 1.0,
                Variant::Dcfr(d) => d.strategy_weight(t),
            };
            let acc = delta.strat.entry(key).or_insert_with(|| vec![0.0; num_actions]);
            for a in 0..num_actions {
                acc[a] += weight * strategy[a];
            }
            let a = sample_index(strategy.iter().copied(), xorshift_next_unit(rng));
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse_ro_cursor(cursor, traverser, rng, delta, t);
            CursorGame::undo(&self.game, cursor);
            self.corrected_opponent_value(key, &strategy, a, v_child, traverser, delta)
        }
    }
}

/// One iteration's private regret / strategy contributions, accumulated by a
/// parallel worker and merged into the shared store afterward.  Keys are
/// `info_key`s; each `Vec` is per-action and its length is the action count
/// (used to create the master node on first merge).
#[derive(Default)]
struct Delta {
    /// Summed instantaneous regret `util[a] − node_value` at traverser nodes.
    regret_inst: HashMap<u64, Vec<f64>>,
    /// Summed `weight · σ(a)` at opponent nodes (the average-strategy numerator).
    strat: HashMap<u64, Vec<f64>>,
    /// VR-MCCFR baseline targets observed this iteration, in player-0 perspective:
    /// summed target value and visit count per (info set, action).  Merged into
    /// the running baseline as an EMA step in iteration order (deterministic).
    baseline_sum: HashMap<u64, Vec<f64>>,
    baseline_cnt: HashMap<u64, Vec<u32>>,
    /// Node entries this worker visited (folded into the global counter on merge,
    /// since the shared counter can't be touched from the parallel `&self` path).
    nodes_visited: u64,
}

/// Record one VR-MCCFR baseline target (player-0 perspective) into a worker's
/// delta — accumulated, then merged as an EMA step in iteration order.
fn record_baseline(delta: &mut Delta, key: u64, n: usize, a: usize, target: f64) {
    delta.baseline_sum.entry(key).or_insert_with(|| vec![0.0; n])[a] += target;
    delta.baseline_cnt.entry(key).or_insert_with(|| vec![0; n])[a] += 1;
}

// ── SoA (flat) blueprint solver ──────────────────────────────────────────────
//
// For an [`IndexedGame`] the info-set space is known up front, so regrets live in
// a flat `f32` [`RegretTable`] addressed by a computed index — the ~10×-smaller
// store the memory budget assumes — instead of the `HashMap<u64, Node>`.  The
// HashMap solver above is untouched (it stays the correctness reference for the
// validation games); this is a separate, parallel implementation of the same
// external-sampling DCFR + VR-MCCFR baseline, storing into the SoA table.
// Arithmetic in `f64`, stored `f32`.  Optimistic / pruning are not implemented
// here (inert on push/fold; the full blueprint can add the optional table arrays
// later).  The transient per-iteration delta reuses [`Delta`] keyed by the
// info-set index cast to `u64`.
pub struct SoaMccfr<G: IndexedGame> {
    game: G,
    variant: Variant,
    use_baseline: bool,
    table: RegretTable,
    rng: u64,
    iterations: u64,
    nodes_visited: u64,
}

impl<G: IndexedGame> SoaMccfr<G> {
    /// Create a solver with a fixed default seed.
    pub fn new(game: G, variant: Variant) -> Self {
        Self::with_seed(game, variant, 0x2545_F491_4F6C_DD1D)
    }

    /// Create a solver with an explicit RNG seed; the flat table is laid out from
    /// the game's known info-set capacity.
    pub fn with_seed(game: G, variant: Variant, seed: u64) -> Self {
        let capacity = game.info_set_capacity();
        let table = RegretTable::with_layout(capacity, |i| game.actions_at(i), false, false);
        Self { game, variant, use_baseline: false, table, rng: seed | 1, iterations: 0, nodes_visited: 0 }
    }

    /// Enable the VR-MCCFR baseline (control variate).
    pub fn with_baseline(mut self) -> Self {
        self.use_baseline = true;
        self
    }

    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    pub fn nodes_visited(&self) -> u64 {
        self.nodes_visited
    }

    /// Per-info-set storage footprint (bytes) of the flat table.
    pub fn bytes_per_info_set(&self) -> usize {
        self.table.bytes_per_info_set()
    }

    /// Average (deployable) strategy at a dense info-set index.
    pub fn average_strategy_at(&self, index: usize) -> Vec<f64> {
        let mut out = Vec::new();
        self.table.average_into(index, &mut out);
        out
    }

    /// Number of info sets in the flat table (the game's
    /// [`info_set_capacity`](crate::games::IndexedGame::info_set_capacity)).
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Whether the info set at `index` was ever reached (has strategy mass).
    pub fn is_visited(&self, index: usize) -> bool {
        self.table.is_visited(index)
    }

    /// Run `iters` external-sampling iterations (serial).
    pub fn train(&mut self, iters: u64) {
        let mut cursor = CursorGame::root(&self.game);
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations;
            let players = CursorGame::num_players(&self.game);
            for traverser in 0..players {
                self.traverse(&mut cursor, traverser, t);
            }
        }
    }

    fn sample(&mut self, probs: &[f64]) -> usize {
        sample_index(probs.iter().copied(), xorshift_next_unit(&mut self.rng))
    }

    fn traverse(&mut self, cursor: &mut G::Cursor, traverser: usize, t: u64) -> f64 {
        self.nodes_visited += 1;
        if CursorGame::is_terminal(&self.game, cursor) {
            return CursorGame::utility(&self.game, cursor, traverser);
        }
        if CursorGame::is_chance(&self.game, cursor) {
            let mut r = self.rng;
            CursorGame::sample_chance(&self.game, cursor, || xorshift_next_unit(&mut r));
            self.rng = r;
            let v = self.traverse(cursor, traverser, t);
            CursorGame::undo_chance(&self.game, cursor);
            return v;
        }

        let player = CursorGame::current_player(&self.game, cursor);
        let index = self.game.info_set_index(cursor);
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        let mut strategy = Vec::new();
        self.table.strategy_into(index, &mut strategy);

        if player == traverser {
            let mut util = vec![0.0; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse(cursor, traverser, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            self.update_regret(index, t, &util, node_value);
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                let b = self.table.baseline_mut(index);
                for a in 0..num_actions {
                    b[a] = (b[a] as f64 + BASELINE_RATE * (sgn * util[a] - b[a] as f64)) as f32;
                }
            }
            node_value
        } else {
            self.update_strategy(index, t, &strategy);
            let a = self.sample(&strategy);
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse(cursor, traverser, t);
            CursorGame::undo(&self.game, cursor);
            if !self.use_baseline {
                return v_child;
            }
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let (baseline_exp, baseline_a) = {
                let b = self.table.baseline(index);
                ((0..num_actions).map(|i| strategy[i] * b[i] as f64).sum::<f64>(), b[a] as f64)
            };
            let corrected0 = baseline_exp + (v0 - baseline_a);
            let b = self.table.baseline_mut(index);
            b[a] = (b[a] as f64 + BASELINE_RATE * (v0 - b[a] as f64)) as f32;
            sgn * corrected0
        }
    }

    fn sign(traverser: usize) -> f64 {
        if traverser == 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn update_regret(&mut self, index: usize, t: u64, util: &[f64], node_value: f64) {
        let (pos, neg) = match self.variant {
            Variant::Vanilla => (1.0, 1.0),
            Variant::Dcfr(d) => (d.positive_factor(t), d.negative_factor(t)),
        };
        let discount = matches!(self.variant, Variant::Dcfr(_));
        let regret = self.table.regret_mut(index);
        for a in 0..regret.len() {
            let mut r = regret[a] as f64;
            if discount {
                r *= if r > 0.0 { pos } else { neg };
            }
            r += util[a] - node_value;
            regret[a] = r as f32;
        }
    }

    fn update_strategy(&mut self, index: usize, t: u64, strategy: &[f64]) {
        let weight = match self.variant {
            Variant::Vanilla => 1.0,
            Variant::Dcfr(d) => d.strategy_weight(t),
        };
        let s = self.table.strategy_sum_mut(index);
        for a in 0..s.len() {
            s[a] = (s[a] as f64 + weight * strategy[a]) as f32;
        }
    }

    /// Mini-batch parallel training (mirrors [`Mccfr::train_parallel_fast`]),
    /// merging index-keyed deltas — including the baseline — in iteration order.
    pub fn train_parallel(&mut self, total_iters: u64, batch: u64)
    where
        G: Sync,
    {
        let batch = batch.max(1);
        let players = CursorGame::num_players(&self.game);
        let mut done = 0u64;
        while done < total_iters {
            let this = batch.min(total_iters - done);
            let base = self.iterations;
            let deltas: Vec<Delta> = (0..this)
                .into_par_iter()
                .map(|i| {
                    let t = base + i + 1;
                    let mut rng = splitmix(self.rng, t);
                    let mut delta = Delta::default();
                    let mut cursor = CursorGame::root(&self.game);
                    for traverser in 0..players {
                        self.traverse_ro(&mut cursor, traverser, &mut rng, &mut delta, t);
                    }
                    delta
                })
                .collect();
            for (i, delta) in deltas.into_iter().enumerate() {
                self.iterations += 1;
                self.apply_delta(delta, base + i as u64 + 1);
            }
            done += this;
        }
    }

    fn traverse_ro(
        &self,
        cursor: &mut G::Cursor,
        traverser: usize,
        rng: &mut u64,
        delta: &mut Delta,
        t: u64,
    ) -> f64 {
        delta.nodes_visited += 1;
        if CursorGame::is_terminal(&self.game, cursor) {
            return CursorGame::utility(&self.game, cursor, traverser);
        }
        if CursorGame::is_chance(&self.game, cursor) {
            CursorGame::sample_chance(&self.game, cursor, || xorshift_next_unit(rng));
            let v = self.traverse_ro(cursor, traverser, rng, delta, t);
            CursorGame::undo_chance(&self.game, cursor);
            return v;
        }

        let player = CursorGame::current_player(&self.game, cursor);
        let index = self.game.info_set_index(cursor);
        let key = index as u64;
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        let mut strategy = Vec::new();
        self.table.strategy_into(index, &mut strategy);

        if player == traverser {
            let mut util = vec![0.0; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse_ro(cursor, traverser, rng, delta, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            let acc = delta.regret_inst.entry(key).or_insert_with(|| vec![0.0; num_actions]);
            for a in 0..num_actions {
                acc[a] += util[a] - node_value;
            }
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                for a in 0..num_actions {
                    record_baseline(delta, key, num_actions, a, sgn * util[a]);
                }
            }
            node_value
        } else {
            let weight = match self.variant {
                Variant::Vanilla => 1.0,
                Variant::Dcfr(d) => d.strategy_weight(t),
            };
            let acc = delta.strat.entry(key).or_insert_with(|| vec![0.0; num_actions]);
            for a in 0..num_actions {
                acc[a] += weight * strategy[a];
            }
            let a = sample_index(strategy.iter().copied(), xorshift_next_unit(rng));
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse_ro(cursor, traverser, rng, delta, t);
            CursorGame::undo(&self.game, cursor);
            if !self.use_baseline {
                return v_child;
            }
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let b = self.table.baseline(index);
            let baseline_exp: f64 = (0..num_actions).map(|i| strategy[i] * b[i] as f64).sum();
            let corrected0 = baseline_exp + (v0 - b[a] as f64);
            record_baseline(delta, key, num_actions, a, v0);
            sgn * corrected0
        }
    }

    fn apply_delta(&mut self, delta: Delta, t: u64) {
        let (pos, neg) = match self.variant {
            Variant::Vanilla => (1.0, 1.0),
            Variant::Dcfr(d) => (d.positive_factor(t), d.negative_factor(t)),
        };
        let discount = matches!(self.variant, Variant::Dcfr(_));
        self.nodes_visited += delta.nodes_visited;
        for (key, inst) in delta.regret_inst {
            let regret = self.table.regret_mut(key as usize);
            for a in 0..regret.len() {
                let mut r = regret[a] as f64;
                if discount {
                    r *= if r > 0.0 { pos } else { neg };
                }
                r += inst[a];
                regret[a] = r as f32;
            }
        }
        for (key, s) in delta.strat {
            let ss = self.table.strategy_sum_mut(key as usize);
            for a in 0..ss.len() {
                ss[a] = (ss[a] as f64 + s[a]) as f32;
            }
        }
        if self.use_baseline {
            for (key, sums) in delta.baseline_sum {
                let cnt = &delta.baseline_cnt[&key];
                let b = self.table.baseline_mut(key as usize);
                for a in 0..b.len() {
                    if cnt[a] > 0 {
                        let mean = sums[a] / cnt[a] as f64;
                        b[a] = (b[a] as f64 + BASELINE_RATE * (mean - b[a] as f64)) as f32;
                    }
                }
            }
        }
    }

    /// Write a resumable checkpoint (the flat table plus the small scalar config).
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let view = SoaCheckpointRef {
            variant: &self.variant,
            use_baseline: self.use_baseline,
            table: &self.table,
            rng: self.rng,
            iterations: self.iterations,
            nodes_visited: self.nodes_visited,
        };
        let bytes = bincode::serialize(&view).map_err(io::Error::other)?;
        let path = path.as_ref();
        let tmp = path.with_extension("ckpt.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
    }

    /// Restore from a checkpoint, re-supplying the game.
    pub fn load_checkpoint(path: impl AsRef<Path>, game: G) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let cp: SoaCheckpointOwned = bincode::deserialize(&bytes).map_err(io::Error::other)?;
        Ok(Self {
            game,
            variant: cp.variant,
            use_baseline: cp.use_baseline,
            table: cp.table,
            rng: cp.rng,
            iterations: cp.iterations,
            nodes_visited: cp.nodes_visited,
        })
    }
}

#[derive(Serialize)]
struct SoaCheckpointRef<'a> {
    variant: &'a Variant,
    use_baseline: bool,
    table: &'a RegretTable,
    rng: u64,
    iterations: u64,
    nodes_visited: u64,
}

#[derive(Deserialize)]
struct SoaCheckpointOwned {
    variant: Variant,
    use_baseline: bool,
    table: RegretTable,
    rng: u64,
    iterations: u64,
    nodes_visited: u64,
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

/// SplitMix64-derived per-iteration RNG seed, so parallel workers get
/// independent, reproducible streams from the solver's base seed.
fn splitmix(seed: u64, iter: u64) -> u64 {
    let mut z = seed ^ iter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::blueprint::BlueprintHoldem;
    use crate::games::kuhn::{Kuhn, GAME_VALUE_P0};
    use crate::games::leduc::Leduc;
    use crate::games::push_fold::PushFoldHoldem;
    use crate::solver::best_response::{exploitability, profile_value};
    use crate::solver::dcfr::Discount;

    #[test]
    fn external_sampling_converges_on_kuhn() {
        let mut solver = Mccfr::new(Kuhn, Variant::Vanilla);
        solver.train(100_000);
        let avg = solver.average_strategy();
        let expl = exploitability(&Kuhn, &avg);
        assert!(expl < 0.01, "MCCFR exploitability {expl} should be < 0.01 after 100k iters");

        let value = profile_value(&Kuhn, &avg, 0);
        assert!(
            (value - GAME_VALUE_P0).abs() < 0.02,
            "MCCFR value {value} should be near -1/18 = {GAME_VALUE_P0}"
        );
    }

    #[test]
    fn dcfr_variant_also_converges_on_kuhn() {
        // DCFR doesn't beat vanilla on a game this small, but it must still
        // converge — the discount schedule should not break the solver.
        let mut solver = Mccfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED));
        solver.train(100_000);
        let expl = exploitability(&Kuhn, &solver.average_strategy());
        assert!(expl < 0.02, "DCFR MCCFR exploitability {expl} should be < 0.02");
    }

    #[test]
    fn discovers_all_kuhn_info_sets() {
        let mut solver = Mccfr::new(Kuhn, Variant::Vanilla);
        solver.train(5_000);
        assert_eq!(solver.num_info_sets(), 12, "should sample all 12 Kuhn info sets");
    }

    #[test]
    fn is_deterministic_for_fixed_seed() {
        let mut a = Mccfr::with_seed(Kuhn, Variant::Vanilla, 42);
        let mut b = Mccfr::with_seed(Kuhn, Variant::Vanilla, 42);
        a.train(2_000);
        b.train(2_000);
        let (sa, sb) = (a.average_strategy(), b.average_strategy());
        assert_eq!(sa.len(), sb.len());
        for (key, va) in &sa {
            let vb = &sb[key];
            for (x, y) in va.iter().zip(vb) {
                assert!((x - y).abs() < 1e-12, "same seed must give identical strategies");
            }
        }
    }

    #[test]
    fn baseline_version_still_converges() {
        // The VR-MCCFR baseline is a control variate: it must not bias the
        // result, so the baseline solver must still reach equilibrium.
        let mut solver = Mccfr::new(Kuhn, Variant::Vanilla).with_baseline();
        solver.train(100_000);
        let expl = exploitability(&Kuhn, &solver.average_strategy());
        assert!(expl < 0.01, "baseline MCCFR exploitability {expl} should be < 0.01");
    }

    #[test]
    fn baseline_is_deterministic_for_fixed_seed() {
        let mut a = Mccfr::with_seed(Kuhn, Variant::Vanilla, 7).with_baseline();
        let mut b = Mccfr::with_seed(Kuhn, Variant::Vanilla, 7).with_baseline();
        a.train(2_000);
        b.train(2_000);
        for (key, va) in &a.average_strategy() {
            for (x, y) in va.iter().zip(&b.average_strategy()[key]) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn baseline_reduces_exploitability_on_kuhn() {
        // Averaged over a fixed set of seeds (so this is deterministic, not
        // flaky), the baseline's control-variate variance reduction lowers the
        // mean exploitability at a fixed iteration budget.
        let seeds = 1..=12u64;
        let iters = 20_000;
        let mean = |with_baseline: bool| -> f64 {
            let xs: Vec<f64> = seeds
                .clone()
                .map(|s| {
                    let mut m = Mccfr::with_seed(Kuhn, Variant::Vanilla, s);
                    if with_baseline {
                        m = m.with_baseline();
                    }
                    m.train(iters);
                    exploitability(&Kuhn, &m.average_strategy())
                })
                .collect();
            xs.iter().sum::<f64>() / xs.len() as f64
        };
        let with = mean(true);
        let without = mean(false);
        assert!(
            with < without,
            "baseline mean exploitability ({with}) should beat no-baseline ({without})"
        );
    }

    #[test]
    fn optimistic_still_converges_on_kuhn() {
        // Optimistic (predictive) updates must not bias the solver: the deployed
        // average strategy still reaches equilibrium.
        let mut s = Mccfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED)).with_optimistic();
        s.train(100_000);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "optimistic MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn optimistic_is_deterministic_for_fixed_seed() {
        let run = || {
            let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 9).with_optimistic();
            s.train(2_000);
            s.average_strategy()
        };
        let (a, b) = (run(), run());
        for (key, va) in &a {
            for (x, y) in va.iter().zip(&b[key]) {
                assert!((x - y).abs() < 1e-12, "optimistic updates must be deterministic");
            }
        }
    }

    #[test]
    fn pruning_preserves_convergence_on_kuhn() {
        // RBP freezes provably-bad branches; the deployed strategy must still
        // converge (a refresh traversal re-checks frozen branches).
        let total = 100_000;
        let cfg = PruningConfig { theta: -2000.0, k: 80, start_fraction: 0.2, refresh_interval: 5_000 };
        let mut s = Mccfr::new(Kuhn, Variant::Vanilla).with_pruning(cfg, total);
        s.train(total);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "pruned MCCFR exploitability {expl} should still converge");
    }

    #[test]
    fn pruning_visits_fewer_nodes_than_plain() {
        // The point of RBP: fewer node visits at an equal iteration budget.  Same
        // seed so the only difference is pruning.
        let total = 100_000;
        let cfg = PruningConfig { theta: -2000.0, k: 80, start_fraction: 0.2, refresh_interval: 5_000 };
        let mut pruned = Mccfr::with_seed(Kuhn, Variant::Vanilla, 1).with_pruning(cfg, total);
        let mut plain = Mccfr::with_seed(Kuhn, Variant::Vanilla, 1);
        pruned.train(total);
        plain.train(total);
        assert!(
            pruned.nodes_visited() < plain.nodes_visited(),
            "RBP should visit fewer nodes: pruned={} plain={}",
            pruned.nodes_visited(),
            plain.nodes_visited()
        );
    }

    #[test]
    fn parallel_converges_on_kuhn() {
        // Mini-batch (parallel) MCCFR is the plain external-sampling estimator,
        // so the deployed average must still reach equilibrium.
        let mut s = Mccfr::new(Kuhn, Variant::Vanilla);
        s.train_parallel(100_000, 64);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "parallel MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn parallel_dcfr_converges_on_kuhn() {
        let mut s = Mccfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED));
        s.train_parallel(100_000, 64);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "parallel DCFR MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn parallel_is_deterministic_for_fixed_seed_and_batch() {
        // Workers merge in iteration order, so a fixed seed + batch gives a
        // bit-identical result no matter how the threads were scheduled.
        let run = || {
            let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 5);
            s.train_parallel(4_000, 32);
            s.average_strategy()
        };
        let (a, b) = (run(), run());
        assert_eq!(a.len(), b.len());
        for (key, va) in &a {
            for (x, y) in va.iter().zip(&b[key]) {
                assert!((x - y).abs() < 1e-12, "parallel training must be deterministic");
            }
        }
    }

    #[test]
    fn parallel_baseline_is_deterministic() {
        // The parallel baseline reads a read-only snapshot and merges target
        // deltas in iteration order, so a fixed seed + batch is reproducible.
        let run = || {
            let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 5).with_baseline();
            s.train_parallel(4_000, 32);
            s.average_strategy()
        };
        let (a, b) = (run(), run());
        assert_eq!(a.len(), b.len());
        for (key, va) in &a {
            for (x, y) in va.iter().zip(&b[key]) {
                assert!((x - y).abs() < 1e-12, "parallel baseline must be deterministic");
            }
        }
    }

    #[test]
    fn parallel_baseline_converges_on_kuhn() {
        // The control variate must not bias the parallel estimator.
        let mut s = Mccfr::new(Kuhn, Variant::Vanilla).with_baseline();
        s.train_parallel(100_000, 32);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "parallel baseline MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn parallel_baseline_reduces_variance_on_kuhn() {
        // Over a fixed seed set, the parallel baseline's control variate lowers
        // the mean exploitability at a fixed budget — the cloud-burst variance
        // lever, now available on the parallel path (mirrors the serial test).
        let seeds = 1..=12u64;
        let (iters, batch) = (20_000, 32);
        let mean = |with_baseline: bool| -> f64 {
            let xs: Vec<f64> = seeds
                .clone()
                .map(|s| {
                    let mut m = Mccfr::with_seed(Kuhn, Variant::Vanilla, s);
                    if with_baseline {
                        m = m.with_baseline();
                    }
                    m.train_parallel(iters, batch);
                    exploitability(&Kuhn, &m.average_strategy())
                })
                .collect();
            xs.iter().sum::<f64>() / xs.len() as f64
        };
        let (with, without) = (mean(true), mean(false));
        assert!(with < without, "parallel baseline mean ({with}) should beat no-baseline ({without})");
    }

    fn strategies_equal(a: &HashMap<u64, Vec<f64>>, b: &HashMap<u64, Vec<f64>>) {
        assert_eq!(a.len(), b.len(), "same info sets");
        for (key, va) in a {
            let vb = &b[key];
            for (x, y) in va.iter().zip(vb) {
                assert!((x - y).abs() < 1e-12, "checkpoint resume must be bit-identical");
            }
        }
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mccfr_ckpt_{tag}_{}.bin", std::process::id()))
    }

    #[test]
    fn resume_from_checkpoint_is_bit_identical() {
        // A run interrupted at the half-way point and resumed from a checkpoint
        // must produce exactly the same strategy as one that never stopped — the
        // proof the full resumable state (regrets, sums, baseline, RNG, iteration
        // counter) round-trips correctly.
        let mut whole = Mccfr::with_seed(Kuhn, Variant::Dcfr(Discount::RECOMMENDED), 11).with_baseline();
        whole.train(100_000);

        let mut part = Mccfr::with_seed(Kuhn, Variant::Dcfr(Discount::RECOMMENDED), 11).with_baseline();
        part.train(50_000);
        let path = temp_path("resume");
        part.save_checkpoint(&path).unwrap();
        drop(part);

        let mut resumed = Mccfr::load_checkpoint(&path, Kuhn).unwrap();
        assert_eq!(resumed.iterations(), 50_000, "iteration counter restored");
        resumed.train(50_000);
        std::fs::remove_file(&path).ok();

        strategies_equal(&whole.average_strategy(), &resumed.average_strategy());
    }

    #[test]
    fn checkpoint_restores_config_and_counters() {
        // The configuration (variant, baseline/optimistic/pruning) and counters
        // must survive the round-trip, not just the regret table.
        let cfg = PruningConfig { theta: -2000.0, k: 80, start_fraction: 0.2, refresh_interval: 5_000 };
        let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 3)
            .with_optimistic()
            .with_pruning(cfg, 100_000);
        s.train(10_000);
        let (it, nv) = (s.iterations(), s.nodes_visited());
        let path = temp_path("config");
        s.save_checkpoint(&path).unwrap();

        let resumed = Mccfr::load_checkpoint(&path, Kuhn).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(resumed.iterations(), it);
        assert_eq!(resumed.nodes_visited(), nv);
        // Continuing must stay deterministic against an uninterrupted twin.
        let mut twin = Mccfr::with_seed(Kuhn, Variant::Vanilla, 3)
            .with_optimistic()
            .with_pruning(cfg, 100_000);
        twin.train(10_000);
        let mut a = resumed;
        a.train(5_000);
        twin.train(5_000);
        strategies_equal(&a.average_strategy(), &twin.average_strategy());
    }

    #[test]
    fn save_is_atomic_no_leftover_temp() {
        // The atomic write renames a temp file into place; afterwards only the
        // checkpoint exists, never a stray `.ckpt.tmp`.
        let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 1);
        s.train(100);
        let path = temp_path("atomic");
        s.save_checkpoint(&path).unwrap();
        let tmp = path.with_extension("ckpt.tmp");
        assert!(path.exists(), "checkpoint written");
        assert!(!tmp.exists(), "no leftover temp file after atomic rename");
        std::fs::remove_file(&path).ok();
    }

    // ── Cursor fast path: bit-identical to the clone-based path ──────────────
    //
    // The whole point of the cursor path is that it changes *nothing* observable
    // — same RNG consumption, same info keys, same updates — only the allocation
    // behavior.  These tests pin that: a fixed seed must yield identical info-set
    // counts, identical strategies, and identical node-visit counts across the
    // two paths.  (Mirrors the bit-identical checkpoint-resume tests above.)

    #[test]
    fn train_fast_matches_clone_on_push_fold() {
        let make = || PushFoldHoldem::new(40, 2, 1, 0);
        let mut clone = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 7).with_baseline();
        clone.train(3_000);
        let mut fast = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 7).with_baseline();
        fast.train_fast(3_000);
        assert_eq!(clone.num_info_sets(), fast.num_info_sets());
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    #[test]
    fn train_fast_matches_clone_with_optimistic_and_rbp() {
        // Exercises the cursor traverser's prune/optimistic branches.
        let total = 5_000;
        let cfg = PruningConfig { theta: -5_000.0, k: 50, start_fraction: 0.2, refresh_interval: 1_000 };
        let make = || PushFoldHoldem::new(40, 2, 1, 0);
        let build = || {
            Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 9)
                .with_baseline()
                .with_optimistic()
                .with_pruning(cfg, total)
        };
        let mut clone = build();
        clone.train(total);
        let mut fast = build();
        fast.train_fast(total);
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    #[test]
    fn train_parallel_fast_matches_clone_parallel_on_push_fold() {
        let make = || PushFoldHoldem::new(40, 2, 1, 0);
        let mut clone = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 5);
        clone.train_parallel(4_000, 32);
        let mut fast = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 5);
        fast.train_parallel_fast(4_000, 32);
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    #[test]
    fn train_fast_matches_clone_on_blueprint() {
        // The blueprint mints fresh postflop info sets on every deal, so this
        // also checks the two paths *discover* the same information sets.
        let make = || BlueprintHoldem::new(40, 2, 1, 0);
        let mut clone = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 3);
        clone.train(1_000);
        let mut fast = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 3);
        fast.train_fast(1_000);
        assert_eq!(clone.num_info_sets(), fast.num_info_sets());
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    /// Throughput of the cursor fast path vs the clone-based path on the
    /// blueprint (deep trees ⇒ the clone-per-node undo-stack allocation hurts
    /// most).  Prints nodes/sec for both; the fast path must not be slower.  Run
    /// in release to get a meaningful number:
    ///   cargo test -p poker-ai --release -- --ignored cursor_fast_path_is_faster --nocapture
    #[test]
    #[ignore]
    fn cursor_fast_path_is_faster() {
        use std::time::Instant;
        let iters = 200_000;

        let mut clone = Mccfr::with_seed(BlueprintHoldem::new(40, 2, 1, 0), Variant::Dcfr(Discount::RECOMMENDED), 1);
        let t0 = Instant::now();
        clone.train(iters);
        let clone_s = t0.elapsed().as_secs_f64();

        let mut fast = Mccfr::with_seed(BlueprintHoldem::new(40, 2, 1, 0), Variant::Dcfr(Discount::RECOMMENDED), 1);
        let t0 = Instant::now();
        fast.train_fast(iters);
        let fast_s = t0.elapsed().as_secs_f64();

        // Same work either way (bit-identical), so nodes/sec is a fair ratio.
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        let nodes = clone.nodes_visited() as f64;
        println!(
            "blueprint {iters} iters: clone {:.2}s ({:.0} nodes/s) vs cursor {:.2}s ({:.0} nodes/s) — {:.2}x",
            clone_s, nodes / clone_s, fast_s, nodes / fast_s, clone_s / fast_s
        );
        assert!(fast_s <= clone_s * 1.05, "cursor path should not be slower (clone {clone_s:.2}s, fast {fast_s:.2}s)");
    }

    /// MCCFR converges on Leduc too — slower and noisier than full traversal, so
    /// run on demand:  cargo test -p poker-ai --release -- --ignored mccfr
    #[test]
    #[ignore]
    fn external_sampling_converges_on_leduc() {
        let mut solver = Mccfr::new(Leduc, Variant::Vanilla);
        solver.train(300_000);
        let expl = exploitability(&Leduc, &solver.average_strategy());
        assert!(expl < 0.05, "Leduc MCCFR exploitability {expl} should be < 0.05");
    }

    /// RBP θ/K sensitivity on Leduc: across a small sweep, pruning should keep
    /// exploitability low while cutting node visits.  On demand (slow):
    ///   cargo test -p poker-ai --release -- --ignored rbp_sensitivity
    #[test]
    #[ignore]
    fn rbp_sensitivity_on_leduc() {
        let total = 400_000;
        let mut plain = Mccfr::with_seed(Leduc, Variant::Dcfr(Discount::RECOMMENDED), 3);
        plain.train(total);
        let plain_expl = exploitability(&Leduc, &plain.average_strategy());
        let plain_nodes = plain.nodes_visited();
        println!("plain: expl={plain_expl:.5}, nodes={plain_nodes}");

        for &(theta, k) in &[(-50.0, 50u32), (-100.0, 100), (-300.0, 200)] {
            let cfg = PruningConfig { theta, k, start_fraction: 0.2, refresh_interval: 10_000 };
            let mut s = Mccfr::with_seed(Leduc, Variant::Dcfr(Discount::RECOMMENDED), 3)
                .with_pruning(cfg, total);
            s.train(total);
            let expl = exploitability(&Leduc, &s.average_strategy());
            let nodes = s.nodes_visited();
            println!("θ={theta} K={k}: expl={expl:.5}, nodes={nodes} ({:.1}% of plain)",
                100.0 * nodes as f64 / plain_nodes as f64);
            assert!(expl < 0.05, "θ={theta} K={k} stayed converged ({expl})");
            assert!(nodes <= plain_nodes, "pruning should not increase node visits");
        }
    }
}
