//! The flat structure-of-arrays blueprint solver ([`SoaMccfr`]): the ~10×
//! smaller regret store for [`IndexedGame`]s, with serial and mini-batch
//! parallel training plus atomic resumable checkpoints.

use std::io;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::parallel::{record_baseline, record_strategy_delta, record_traverser_delta, splitmix, Delta};
use super::BASELINE_RATE;
use crate::games::{CursorGame, IndexedGame};
use crate::solver::cfr::Variant;
use crate::solver::lean_table::LeanTable;
use crate::solver::regret_table::{RegretStore, RegretTable};
use crate::util::rng::{sample_index, xorshift_next_unit};

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
pub struct SoaMccfr<G: IndexedGame, S: RegretStore = RegretTable> {
    game: G,
    variant: Variant,
    use_baseline: bool,
    table: S,
    rng: u64,
    iterations: u64,
    nodes_visited: u64,
}

/// The quantized-store solver: the same serial algorithm over a [`LeanTable`]
/// (i16/u16 accumulators, half the RAM).  Pair it with
/// [`Discount::LINEAR`](crate::solver::dcfr::Discount::LINEAR) — quantized
/// regrets need Linear CFR's growing magnitudes (see `lean_table.rs`).
pub type LeanMccfr<G> = SoaMccfr<G, LeanTable>;

impl<G: IndexedGame, S: RegretStore> SoaMccfr<G, S> {
    /// Create a solver with a fixed default seed.
    pub fn new(game: G, variant: Variant) -> Self {
        Self::with_seed(game, variant, 0x2545_F491_4F6C_DD1D)
    }

    /// Create a solver with an explicit RNG seed; the flat table is laid out from
    /// the game's known info-set capacity.
    pub fn with_seed(game: G, variant: Variant, seed: u64) -> Self {
        let capacity = game.info_set_capacity();
        let table = S::build(capacity, &|i| game.actions_at(i));
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
            self.table.add_regret(index, &util, node_value, t, self.variant);
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                for (a, &u) in util.iter().enumerate() {
                    self.table.baseline_ema(index, a, sgn * u);
                }
            }
            node_value
        } else {
            self.table.add_strategy(index, &strategy, t, self.variant, &mut self.rng);
            let a = self.sample(&strategy);
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse(cursor, traverser, t);
            CursorGame::undo(&self.game, cursor);
            if !self.use_baseline {
                return v_child;
            }
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let (baseline_exp, baseline_a) = self.table.baseline_pair(index, &strategy, a);
            let corrected0 = baseline_exp + (v0 - baseline_a);
            self.table.baseline_ema(index, a, v0);
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

}

// ── f32-store-only paths ─────────────────────────────────────────────────────
//
// The mini-batch parallel merge, the lock-free atomic trainer, and the
// checkpoint format all operate on the concrete f32 arrays; the quantized
// store is serial-only until it earns those (benchmark first).
impl<G: IndexedGame> SoaMccfr<G, RegretTable> {
    /// Lock-free atomic training over `threads` OS threads — the many-core
    /// path (see [`super::atomic`] for the design and what it trades away).
    /// Workers claim iteration numbers from a shared counter and CAS directly
    /// into the flat table: no batches, no merge, near-linear scaling.
    /// **Not bit-deterministic across runs** (thread interleaving changes
    /// float update order); use [`train`](Self::train) or
    /// [`train_parallel`](Self::train_parallel) when reproducibility matters
    /// more than throughput.  Convergence is gated by
    /// `atomic_converges_like_the_serial_soa`.
    pub fn train_atomic(&mut self, iters: u64, threads: usize)
    where
        G: Sync,
    {
        self.nodes_visited += super::atomic::run_atomic(
            &self.game,
            &mut self.table,
            super::atomic::AtomicRun {
                variant: self.variant,
                use_baseline: self.use_baseline,
                seed: self.rng,
                base_iter: self.iterations,
                iters,
                threads,
            },
        );
        self.iterations += iters;
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
            for (r32, &i) in regret.iter_mut().zip(&inst) {
                let mut r = *r32 as f64;
                if discount {
                    r *= if r > 0.0 { pos } else { neg };
                }
                r += i;
                *r32 = r as f32;
            }
        }
        for (key, s) in delta.strat {
            let ss = self.table.strategy_sum_mut(key as usize);
            for (sum, &v) in ss.iter_mut().zip(&s) {
                *sum += v;
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
