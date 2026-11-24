//! The flat structure-of-arrays blueprint solver ([`SoaMccfr`]): the ~10×
//! smaller regret store for [`IndexedGame`]s, with serial and mini-batch
//! parallel training plus atomic resumable checkpoints.

use std::io;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::parallel::{record_baseline, record_strategy_delta, record_traverser_delta, Delta};
use crate::util::rng::splitmix_stream;
use super::BASELINE_RATE;
use crate::games::{CursorGame, IndexedGame, Layout, Span};
use crate::solver::variant::Variant;
use crate::solver::lean_table::LeanTable;
use crate::solver::regret_table::{sr_add, RegretStore, RegretTable};
use crate::solver::MAX_ACTIONS;
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
    /// Stochastic-rounding stream for the strategy sum, deliberately **separate
    /// from `rng`**: both stores round their average-strategy accumulator, and
    /// drawing those numbers from the sampling stream would make the set of
    /// trajectories a seed explores depend on the storage precision — which
    /// would have made the f64→f32 change impossible to A/B honestly.
    sr_rng: u64,
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
        let table = S::build(game.layout());
        Self {
            game,
            variant,
            use_baseline: false,
            table,
            rng: seed | 1,
            sr_rng: super::atomic::SR_STREAM ^ (seed | 1),
            iterations: 0,
            nodes_visited: 0,
        }
    }

    /// Enable the VR-MCCFR baseline (control variate).  The store allocates
    /// its baseline array here, so a run without control variates never
    /// carries it — a third of the table on either store.
    pub fn with_baseline(mut self) -> Self {
        self.use_baseline = true;
        self.table.enable_baseline();
        self
    }

    /// Whether the VR-MCCFR baseline is on (fixed at construction, and
    /// restored from a checkpoint on resume).
    pub fn uses_baseline(&self) -> bool {
        self.use_baseline
    }

    /// The regret/strategy schedule (fixed at construction, and restored from
    /// a checkpoint on resume).
    pub fn variant(&self) -> Variant {
        self.variant
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
        self.table.average_into(self.table.span(index), &mut out);
        out
    }

    /// The flat table's slot layout — one [`Span`] per info set, in index
    /// order.
    pub fn layout(&self) -> &Layout {
        self.table.layout()
    }

    /// The average strategy of the info set at `at` written into `out`, or
    /// `false` (and `out` untouched) if training never reached it.  For
    /// exports that walk [`layout`](Self::layout) in order: no per-info-set
    /// lookup or allocation.
    pub fn average_strategy_into(&self, at: Span, out: &mut Vec<f64>) -> bool {
        if !self.table.is_visited(at) {
            return false;
        }
        self.table.average_into(at, out);
        true
    }

    /// Number of info sets in the flat table (the game's
    /// [`info_set_capacity`](crate::games::IndexedGame::info_set_capacity)).
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Whether the info set at `index` was ever reached (has strategy mass).
    pub fn is_visited(&self, index: usize) -> bool {
        self.table.is_visited(self.table.span(index))
    }

    /// The slots of the info set at `cursor`: from the game when it can say
    /// (no lookup at all), else from the table's layout.  Once per node.
    #[inline]
    fn span_at(&self, cursor: &G::Cursor, index: usize) -> Span {
        let at = self.game.slot_span(cursor, index).unwrap_or_else(|| self.table.span(index));
        debug_assert_eq!(at, self.table.span(index), "the game's slot span disagrees with the table layout");
        at
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
        let at = self.span_at(cursor, index);
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        // Stack scratch, not `Vec`s: this runs once per decision node.
        let mut strategy = [0.0f64; MAX_ACTIONS];
        let n = self.table.strategy_into(at, &mut strategy);
        debug_assert_eq!(n, num_actions, "table fan-out must match the engine's");
        let strategy = &strategy[..n];

        if player == traverser {
            let mut util = [0.0f64; MAX_ACTIONS];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse(cursor, traverser, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            let util = &util[..num_actions];
            self.table.add_regret(at, util, node_value, t, self.variant);
            if self.use_baseline {
                let sgn = Self::sign(traverser);
                for (a, &u) in util.iter().enumerate() {
                    self.table.baseline_ema(at, a, sgn * u);
                }
            }
            node_value
        } else {
            self.table.add_strategy(at, strategy, t, self.variant, &mut self.sr_rng);
            let a = self.sample(strategy);
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse(cursor, traverser, t);
            CursorGame::undo(&self.game, cursor);
            if !self.use_baseline {
                return v_child;
            }
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let (baseline_exp, baseline_a) = self.table.baseline_pair(at, strategy, a);
            let corrected0 = baseline_exp + (v0 - baseline_a);
            self.table.baseline_ema(at, a, v0);
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
    /// path (see `super::atomic` for the design and what it trades away).
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

    /// Mini-batch parallel training (mirrors [`train_parallel_fast`](super::Mccfr::train_parallel_fast)),
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
                    let mut rng = splitmix_stream(self.rng, t);
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
        let at = self.span_at(cursor, index);
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        // Stack scratch, not `Vec`s: this runs once per decision node.
        let mut strategy = [0.0f64; MAX_ACTIONS];
        let n = self.table.strategy_into(at, &mut strategy);
        debug_assert_eq!(n, num_actions, "table fan-out must match the engine's");
        let strategy = &strategy[..n];

        if player == traverser {
            let mut util = [0.0f64; MAX_ACTIONS];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse_ro(cursor, traverser, rng, delta, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            let sgn = self.use_baseline.then_some(Self::sign(traverser));
            record_traverser_delta(delta, key, &util[..num_actions], node_value, sgn);
            node_value
        } else {
            let weight = match self.variant {
                Variant::Vanilla => 1.0,
                Variant::Dcfr(d) => d.strategy_weight(t),
            };
            record_strategy_delta(delta, key, weight, strategy);
            let a = sample_index(strategy.iter().copied(), xorshift_next_unit(rng));
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse_ro(cursor, traverser, rng, delta, t);
            CursorGame::undo(&self.game, cursor);
            if !self.use_baseline {
                return v_child;
            }
            let sgn = Self::sign(traverser);
            let v0 = sgn * v_child;
            let b = self.table.baseline(at);
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
            let regret = self.table.regret_mut(self.table.span(key as usize));
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
            let sr = &mut self.sr_rng;
            let ss = self.table.strategy_sum_mut(self.table.span(key as usize));
            for (sum, &v) in ss.iter_mut().zip(&s) {
                sr_add(sum, v, sr);
            }
        }
        if self.use_baseline {
            for (key, sums) in delta.baseline_sum {
                let cnt = &delta.baseline_cnt[&key];
                let b = self.table.baseline_mut(self.table.span(key as usize));
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
            magic: SOA_CKPT_MAGIC,
            variant: &self.variant,
            use_baseline: self.use_baseline,
            table: &self.table,
            rng: self.rng,
            sr_rng: self.sr_rng,
            iterations: self.iterations,
            nodes_visited: self.nodes_visited,
        };
        // Stream straight to the temp file: buffering the serialized table in
        // RAM first would transiently double the footprint (a 15 GB table cost
        // ~30 GB extra at 200 bb — observed on the first server run).
        let path = path.as_ref();
        let tmp = path.with_extension("ckpt.tmp");
        let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        bincode::serialize_into(&mut w, &view).map_err(io::Error::other)?;
        std::io::Write::flush(&mut w)?;
        drop(w);
        std::fs::rename(&tmp, path)
    }

    /// Restore from a checkpoint, re-supplying the game.  Streams from disk
    /// (no whole-file byte buffer — same rationale as the save path).
    ///
    /// Accepts **every** checkpoint layout this solver has written (see the
    /// `legacy` module), so a run interrupted under an older build resumes
    /// under this one, converted in place.
    ///
    /// Fails if the checkpoint's table is not laid out the way `game` lays
    /// one out — a different raise cap, stack depth or bucket count.  Such a
    /// table has the wrong info sets in its slots; training on would not
    /// crash, it would silently mix them.
    pub fn load_checkpoint(path: impl AsRef<Path>, game: G) -> io::Result<Self> {
        let path = path.as_ref();
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        // `peek_magic` consumes the marker, so what follows is exactly the
        // body — which is why the read structs omit the `magic` field the
        // write struct carries.
        let cp = match peek_magic(&mut r)? {
            SOA_CKPT_MAGIC => {
                let cp: SoaCheckpointBody = bincode::deserialize_from(r).map_err(io::Error::other)?;
                cp.table.validate().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                cp
            }
            legacy::V2_MAGIC => legacy::load_v2(r)?,
            _ => legacy::load_v1(std::io::BufReader::new(std::fs::File::open(path)?))?,
        };
        if *cp.table.layout() != game.layout() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} holds a table of {} info sets / {} slots, laid out for a different game than this one \
                     ({} info sets) — it was trained with another raise cap, stack depth or bucket count",
                    path.display(),
                    cp.table.capacity(),
                    cp.table.total_slots(),
                    game.info_set_capacity(),
                ),
            ));
        }
        Ok(Self {
            game,
            variant: cp.variant,
            use_baseline: cp.use_baseline,
            table: cp.table,
            rng: cp.rng,
            sr_rng: cp.sr_rng,
            iterations: cp.iterations,
            nodes_visited: cp.nodes_visited,
        })
    }
}

/// Marker at the head of a current-layout SoA checkpoint.  The first
/// checkpoints had none and start with bincode's enum tag for `Variant` — a
/// `u32` of 0 or 1 — so every generation is unambiguously distinguishable and
/// an interrupted server run can be migrated instead of orphaned.
const SOA_CKPT_MAGIC: u64 = 0x3341_4F53_524B_4F50; // "POKRSOA3", little-endian

/// Read the leading `u64` of a checkpoint.  A legacy file without a marker is
/// reopened by the caller (checkpoints are large and streamed, so reopening
/// beats buffering).
fn peek_magic(r: &mut impl std::io::Read) -> io::Result<u64> {
    let mut head = [0u8; 8];
    match r.read_exact(&mut head) {
        Ok(()) => Ok(u64::from_le_bytes(head)),
        // A file too short to hold the marker is certainly not one.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(0),
        Err(e) => Err(e),
    }
}

#[derive(Serialize)]
struct SoaCheckpointRef<'a> {
    magic: u64,
    variant: &'a Variant,
    use_baseline: bool,
    table: &'a RegretTable,
    rng: u64,
    sr_rng: u64,
    iterations: u64,
    nodes_visited: u64,
}

/// The checkpoint after its marker: bincode writes fields back to back with no
/// framing, so this is byte-for-byte the tail of [`SoaCheckpointRef`].
#[derive(Deserialize)]
struct SoaCheckpointBody {
    variant: Variant,
    use_baseline: bool,
    table: RegretTable,
    rng: u64,
    sr_rng: u64,
    iterations: u64,
    nodes_visited: u64,
}

/// Reading the checkpoint layouts earlier builds wrote.
///
/// bincode is not self-describing, so an old file cannot simply be handed to
/// the current `Deserialize` — the fields differ and everything after the
/// first difference would be misread.  These mirrors reproduce the old layouts
/// exactly and convert on load:
///
/// * **v2** (`POKRSOA2`): the table carried an action count and a slot offset
///   *per info set* (5 bytes each — 3 GB at 595 M info sets), plus two
///   accumulator arrays for features this solver never implemented, always
///   empty.  The action counts become the run-length [`Layout`]; the rest is
///   dropped.
/// * **v1** (no marker): as v2, but the strategy sums were `f64`.  They are
///   narrowed to `f32` — a one-time round-to-nearest of an already-accumulated
///   total, which preserves the within-info-set ratios that are the only thing
///   the average depends on — and the rounding stream is seeded from the run's
///   own RNG so the resumed run stays deterministic.
mod legacy {
    use super::*;

    pub(super) const V2_MAGIC: u64 = 0x3241_4F53_524B_4F50; // "POKRSOA2"

    #[derive(Deserialize)]
    struct TableV2 {
        regret: Vec<f32>,
        strategy_sum: Vec<f32>,
        baseline: Vec<f32>,
        _prev_inst: Vec<f32>,
        _consec_below: Vec<u32>,
        num_actions: Vec<u8>,
        _offsets: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct CheckpointV2 {
        variant: Variant,
        use_baseline: bool,
        table: TableV2,
        rng: u64,
        sr_rng: u64,
        iterations: u64,
        nodes_visited: u64,
    }

    #[derive(Deserialize)]
    struct TableV1 {
        regret: Vec<f32>,
        strategy_sum: Vec<f64>,
        baseline: Vec<f32>,
        _prev_inst: Vec<f32>,
        _consec_below: Vec<u32>,
        num_actions: Vec<u8>,
        _offsets: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct CheckpointV1 {
        variant: Variant,
        use_baseline: bool,
        table: TableV1,
        rng: u64,
        iterations: u64,
        nodes_visited: u64,
    }

    /// The per-info-set action counts as a run-length layout.  The offsets the
    /// old table stored alongside were their running sum, which the layout
    /// recomputes (in `u64`, where the old `u32` could wrap).
    fn layout_of(num_actions: &[u8]) -> Layout {
        Layout::from_runs(num_actions.iter().map(|&n| (1, n as usize)))
    }

    /// The body of a v2 checkpoint, after its marker.
    pub(super) fn load_v2(r: impl std::io::Read) -> io::Result<SoaCheckpointBody> {
        let cp: CheckpointV2 = bincode::deserialize_from(r).map_err(io::Error::other)?;
        let t = cp.table;
        let layout = layout_of(&t.num_actions);
        Ok(SoaCheckpointBody {
            variant: cp.variant,
            use_baseline: cp.use_baseline,
            table: RegretTable::from_parts(t.regret, t.strategy_sum, t.baseline, layout)?,
            rng: cp.rng,
            sr_rng: cp.sr_rng,
            iterations: cp.iterations,
            nodes_visited: cp.nodes_visited,
        })
    }

    /// A whole v1 checkpoint (it has no marker to skip).
    pub(super) fn load_v1(r: impl std::io::Read) -> io::Result<SoaCheckpointBody> {
        let cp: CheckpointV1 = bincode::deserialize_from(r).map_err(io::Error::other)?;
        let t = cp.table;
        let layout = layout_of(&t.num_actions);
        let strategy_sum = t.strategy_sum.into_iter().map(|x| x as f32).collect();
        Ok(SoaCheckpointBody {
            variant: cp.variant,
            use_baseline: cp.use_baseline,
            table: RegretTable::from_parts(t.regret, strategy_sum, t.baseline, layout)?,
            rng: cp.rng,
            sr_rng: super::super::atomic::SR_STREAM ^ cp.rng,
            iterations: cp.iterations,
            nodes_visited: cp.nodes_visited,
        })
    }

    /// Serialize a solver's state in an **old** layout — test-only, so the
    /// migrations are gated against bytes actually shaped like the old
    /// checkpoints rather than against a hand-written assumption about them.
    /// `v1` selects the marker-less `f64`-sum form.
    #[cfg(test)]
    pub(super) fn write_old<G: IndexedGame>(
        w: &mut impl std::io::Write,
        s: &SoaMccfr<G, RegretTable>,
        v1: bool,
    ) -> io::Result<()> {
        #[derive(Serialize)]
        struct Table<'a, S> {
            regret: &'a [f32],
            strategy_sum: S,
            baseline: &'a [f32],
            prev_inst: &'a [f32],
            consec_below: &'a [u32],
            num_actions: Vec<u8>,
            offsets: Vec<u32>,
        }
        let (regret, strategy_sum, baseline, layout) = s.table.parts_for_test();
        let num_actions = layout.expanded_widths();
        let offsets = layout.spans().map(|at| at.base as u32).collect();
        let ser = |e| io::Error::other(e);
        if v1 {
            let table = Table {
                regret,
                strategy_sum: strategy_sum.iter().map(|&x| x as f64).collect::<Vec<f64>>(),
                baseline,
                prev_inst: &[],
                consec_below: &[],
                num_actions,
                offsets,
            };
            bincode::serialize_into(w, &(s.variant, s.use_baseline, table, s.rng, s.iterations, s.nodes_visited))
                .map_err(ser)
        } else {
            let table =
                Table { regret, strategy_sum, baseline, prev_inst: &[], consec_below: &[], num_actions, offsets };
            bincode::serialize_into(
                w,
                &(V2_MAGIC, s.variant, s.use_baseline, table, s.rng, s.sr_rng, s.iterations, s.nodes_visited),
            )
            .map_err(ser)
        }
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::games::push_fold::PushFoldHoldem;
    use crate::solver::dcfr::Discount;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("soa_ckpt_{tag}_{}.bin", std::process::id()))
    }

    fn trained() -> SoaMccfr<PushFoldHoldem> {
        let game = PushFoldHoldem::new(40, 2, 1, 0);
        let mut s = SoaMccfr::with_seed(game, Variant::Dcfr(Discount::RECOMMENDED), 4).with_baseline();
        s.train(20_000);
        s
    }

    fn write_old(s: &SoaMccfr<PushFoldHoldem>, path: &Path, v1: bool) {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        legacy::write_old(&mut w, s, v1).unwrap();
        std::io::Write::flush(&mut w).unwrap();
    }

    #[test]
    fn current_checkpoint_round_trips_and_resumes() {
        let s = trained();
        let path = temp_path("v3");
        s.save_checkpoint(&path).unwrap();
        let loaded =
            SoaMccfr::load_checkpoint(&path, PushFoldHoldem::new(40, 2, 1, 0)).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.iterations(), s.iterations());
        assert_eq!(loaded.nodes_visited(), s.nodes_visited());
        assert_eq!(loaded.average_strategy_at(0), s.average_strategy_at(0));
    }

    #[test]
    fn a_checkpoint_with_per_info_set_arrays_still_loads() {
        // The layout the server's checkpoints are in today: an action count
        // and an offset for every info set.  Loading one must give back the
        // identical table — same bits in every slot — and train on from it
        // exactly as the solver that wrote it would have.
        let mut s = trained();
        let path = temp_path("v2");
        write_old(&s, &path, false);
        let mut loaded = SoaMccfr::load_checkpoint(&path, PushFoldHoldem::new(40, 2, 1, 0)).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.iterations(), s.iterations(), "counters survive");
        assert_eq!(loaded.nodes_visited(), s.nodes_visited());
        assert!(loaded.uses_baseline());
        assert_eq!(loaded.table.parts_for_test(), s.table.parts_for_test(), "the table is bit-identical");
        s.train(2_000);
        loaded.train(2_000);
        assert_eq!(loaded.table.parts_for_test(), s.table.parts_for_test(), "and resumes bit-identically");
    }

    #[test]
    fn a_pre_f32_checkpoint_still_loads() {
        // The server's interrupted runs must survive the storage change: a file
        // written in the oldest layout (f64 strategy sums, no magic marker) has to
        // load, carry its counters, and keep its average strategy.
        let s = trained();
        let path = temp_path("v1");
        write_old(&s, &path, true);
        let loaded =
            SoaMccfr::load_checkpoint(&path, PushFoldHoldem::new(40, 2, 1, 0)).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.iterations(), s.iterations(), "counters survive");
        assert_eq!(loaded.nodes_visited(), s.nodes_visited());
        assert_eq!(loaded.bytes_per_info_set(), 24, "converted to the f32 layout");
        // The sums went f64 -> f32, so the deployed strategy is preserved to
        // f32 precision rather than bit-for-bit.
        for (a, b) in loaded.average_strategy_at(0).iter().zip(s.average_strategy_at(0)) {
            assert!((a - b).abs() < 1e-6, "strategy preserved: {a} vs {b}");
        }
        // …and it can keep training from there.
        let mut resumed = loaded;
        resumed.train(1_000);
        assert_eq!(resumed.iterations(), s.iterations() + 1_000);
    }

    /// A game laid out differently from the one a checkpoint was trained on —
    /// here by claiming a third action everywhere, standing in for a changed
    /// raise cap.
    struct Wider(PushFoldHoldem);

    impl CursorGame for Wider {
        type Cursor = <PushFoldHoldem as CursorGame>::Cursor;
        type Action = <PushFoldHoldem as CursorGame>::Action;
        type Actions = <PushFoldHoldem as CursorGame>::Actions;
        fn num_players(&self) -> usize {
            self.0.num_players()
        }
        fn root(&self) -> Self::Cursor {
            self.0.root()
        }
        fn is_terminal(&self, c: &Self::Cursor) -> bool {
            self.0.is_terminal(c)
        }
        fn is_chance(&self, c: &Self::Cursor) -> bool {
            self.0.is_chance(c)
        }
        fn utility(&self, c: &Self::Cursor, player: usize) -> f64 {
            self.0.utility(c, player)
        }
        fn current_player(&self, c: &Self::Cursor) -> usize {
            self.0.current_player(c)
        }
        fn legal(&self, c: &Self::Cursor) -> Self::Actions {
            self.0.legal(c)
        }
        fn info_key(&self, c: &Self::Cursor) -> u64 {
            CursorGame::info_key(&self.0, c)
        }
        fn apply(&self, c: &mut Self::Cursor, a: usize, action: Self::Action) {
            self.0.apply(c, a, action)
        }
        fn undo(&self, c: &mut Self::Cursor) {
            self.0.undo(c)
        }
        fn sample_chance(&self, c: &mut Self::Cursor, next_unit: impl FnMut() -> f64) {
            self.0.sample_chance(c, next_unit)
        }
        fn undo_chance(&self, c: &mut Self::Cursor) {
            self.0.undo_chance(c)
        }
    }

    impl IndexedGame for Wider {
        fn info_set_capacity(&self) -> usize {
            self.0.info_set_capacity()
        }
        fn info_set_index(&self, c: &Self::Cursor) -> usize {
            self.0.info_set_index(c)
        }
        fn actions_at(&self, _index: usize) -> usize {
            3
        }
    }

    #[test]
    fn a_checkpoint_from_a_different_game_is_refused() {
        // Resuming a table under another betting tree used to go unnoticed:
        // the info-set count can even match while the slots mean something
        // else.  It has to be an error that names the cause.
        let s = trained();
        let path = temp_path("foreign");
        s.save_checkpoint(&path).unwrap();
        let err = match SoaMccfr::load_checkpoint(&path, Wider(PushFoldHoldem::new(40, 2, 1, 0))) {
            Err(e) => e,
            Ok(_) => panic!("a table laid out for another game must not load"),
        };
        std::fs::remove_file(&path).ok();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("different game"), "{err}");
    }
}
