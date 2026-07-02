//! Mini-batch lock-free parallel training for the `HashMap`-backed solver, plus
//! the [`Delta`] worker-contribution machinery shared by every parallel path
//! (clone, cursor, and SoA).
//!
//! See the module comment inside the `impl` below for the batching scheme and
//! its determinism/staleness trade-off.

use std::collections::HashMap;

use rayon::prelude::*;

use super::{Mccfr, Node, BASELINE_RATE};
use crate::games::Game;
use crate::solver::cfr::Variant;
use crate::util::rng::{sample_index, xorshift_next_unit};

impl<G: Game> Mccfr<G> {
    // ── Parallel (batched) training ──────────────────────────────────────────
    //
    // The cloud burst needs the CFR loop parallelized lock-free so the
    // rented cores aren't idle on a memory-bound workload.  The shared regret
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
    // The **VR-MCCFR baseline is supported** here (the headline variance
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
    pub(super) fn strategy_ro(&self, key: u64, num_actions: usize) -> Vec<f64> {
        match self.nodes.get(&key) {
            Some(node) => node.strategy(),
            None => vec![1.0 / num_actions as f64; num_actions],
        }
    }

    /// Read-only **snapshot** of an info set's baseline (start-of-batch state),
    /// zeros if unseen — the control variate the parallel traversal reads.
    pub(super) fn baseline_ro(&self, key: u64, num_actions: usize) -> Vec<f64> {
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
            let sgn = self.use_baseline.then_some(Self::sign(traverser));
            record_traverser_delta(delta, key, &util, node_value, sgn);
            node_value
        } else {
            let weight = match self.variant {
                Variant::Vanilla => 1.0,
                Variant::Dcfr(d) => d.strategy_weight(t),
            };
            record_strategy_delta(delta, key, weight, &strategy);
            let a = sample_index(strategy.iter().copied(), xorshift_next_unit(rng));
            let child = self.game.apply(state, a);
            let v_child = self.traverse_ro(&child, traverser, rng, delta, t);
            self.corrected_opponent_value(key, &strategy, a, v_child, traverser, delta)
        }
    }

    /// At a sampled opponent node, apply the VR-MCCFR control variate using the
    /// snapshot baseline and record the realized target — shared by both parallel
    /// traversals.  With the baseline off this is just `v_child`.
    pub(super) fn corrected_opponent_value(
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
    pub(super) fn apply_delta(&mut self, delta: Delta, t: u64) {
        let (pos, neg) = match self.variant {
            Variant::Vanilla => (1.0, 1.0),
            Variant::Dcfr(d) => (d.positive_factor(t), d.negative_factor(t)),
        };
        let discount = matches!(self.variant, Variant::Dcfr(_));
        self.nodes_visited += delta.nodes_visited;
        for (key, inst) in delta.regret_inst {
            let node = self.nodes.entry(key).or_insert_with(|| Node::new(inst.len()));
            for (r, &i) in node.regret_sum.iter_mut().zip(&inst) {
                if discount {
                    *r *= if *r > 0.0 { pos } else { neg };
                }
                *r += i;
            }
        }
        for (key, s) in delta.strat {
            let node = self.nodes.entry(key).or_insert_with(|| Node::new(s.len()));
            for (sum, &v) in node.strategy_sum.iter_mut().zip(&s) {
                *sum += v;
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

/// One iteration's private regret / strategy contributions, accumulated by a
/// parallel worker and merged into the shared store afterward.  Keys are
/// `info_key`s; each `Vec` is per-action and its length is the action count
/// (used to create the master node on first merge).
#[derive(Default)]
pub(super) struct Delta {
    /// Summed instantaneous regret `util[a] − node_value` at traverser nodes.
    pub(super) regret_inst: HashMap<u64, Vec<f64>>,
    /// Summed `weight · σ(a)` at opponent nodes (the average-strategy numerator).
    pub(super) strat: HashMap<u64, Vec<f64>>,
    /// VR-MCCFR baseline targets observed this iteration, in player-0 perspective:
    /// summed target value and visit count per (info set, action).  Merged into
    /// the running baseline as an EMA step in iteration order (deterministic).
    pub(super) baseline_sum: HashMap<u64, Vec<f64>>,
    pub(super) baseline_cnt: HashMap<u64, Vec<u32>>,
    /// Node entries this worker visited (folded into the global counter on merge,
    /// since the shared counter can't be touched from the parallel `&self` path).
    pub(super) nodes_visited: u64,
}

/// Record one VR-MCCFR baseline target (player-0 perspective) into a worker's
/// delta — accumulated, then merged as an EMA step in iteration order.
pub(super) fn record_baseline(delta: &mut Delta, key: u64, n: usize, a: usize, target: f64) {
    delta.baseline_sum.entry(key).or_insert_with(|| vec![0.0; n])[a] += target;
    delta.baseline_cnt.entry(key).or_insert_with(|| vec![0; n])[a] += 1;
}

/// Record a traverser node's per-action instantaneous regrets — and, when the
/// baseline is on (`baseline_sign = Some(sign)`), its per-action baseline
/// targets — into a worker's delta.  Shared by the clone, cursor, and SoA
/// parallel traversals.
pub(super) fn record_traverser_delta(
    delta: &mut Delta,
    key: u64,
    util: &[f64],
    node_value: f64,
    baseline_sign: Option<f64>,
) {
    let acc = delta.regret_inst.entry(key).or_insert_with(|| vec![0.0; util.len()]);
    for (acc_a, &u) in acc.iter_mut().zip(util) {
        *acc_a += u - node_value;
    }
    if let Some(sgn) = baseline_sign {
        for (a, &u) in util.iter().enumerate() {
            record_baseline(delta, key, util.len(), a, sgn * u);
        }
    }
}

/// Accumulate `weight · σ(a)` at an opponent node into a worker's delta (the
/// average-strategy numerator).  Shared by the three parallel traversals.
pub(super) fn record_strategy_delta(delta: &mut Delta, key: u64, weight: f64, strategy: &[f64]) {
    let acc = delta.strat.entry(key).or_insert_with(|| vec![0.0; strategy.len()]);
    for (acc_a, &s) in acc.iter_mut().zip(strategy) {
        *acc_a += weight * s;
    }
}

/// SplitMix64-derived per-iteration RNG seed, so parallel workers get
/// independent, reproducible streams from the solver's base seed.
pub(super) fn splitmix(seed: u64, iter: u64) -> u64 {
    let mut z = seed ^ iter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}
