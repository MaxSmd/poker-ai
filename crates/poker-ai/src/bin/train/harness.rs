//! The chunked training loop shared by every trainer configuration.
//!
//! One driver — advance, checkpoint, progress line, `@wandb` metrics, with the
//! expensive evaluation gated by `--expl-every` — behind a small trait, plus
//! one adapter per solver family.  Cadence and checkpoint policy change here
//! and nowhere else.

use std::path::Path;
use std::time::Instant;

use poker_ai::games::{CursorGame, Game, IndexedGame};
use poker_ai::solver::mccfr::{Mccfr, SoaMccfr};

use crate::{emit_metric, Cadence};


/// What the shared loop needs from a trainer.  Implemented once for the
/// HashMap/cursor solver ([`CursorTrainer`]) and once for the flat SoA solver
/// ([`SoaTrainer`]), each generic over the game — so all four CLI paths
/// (HashMap/SoA × push-fold/blueprint) drive the same loop.
pub trait TrainerOps {
    fn iterations(&self) -> u64;
    fn nodes_visited(&self) -> u64;
    /// Discovered (HashMap) or allocated (SoA) info sets — the always-on
    /// health signal (a HashMap count that fails to plateau means the card
    /// abstraction is missing coverage).
    fn info_sets(&self) -> usize;
    fn advance(&mut self, step: u64);
    fn save_checkpoint(&self, path: &Path);
    /// Expensive periodic evaluation in bb/hand.  Called only on
    /// `--expl-every`-gated chunks; `None` (the default) skips the report.
    fn exploitability(&self) -> Option<f64> {
        None
    }
}

/// Train to `iters` in `cad.chunk`-sized chunks: advance → checkpoint →
/// progress line → `@wandb` metrics, with the expensive exploitability eval
/// gated to every `cad.expl_every`-th chunk (plus the last).  This is the
/// **single point of change** for cadence, checkpointing, and reporting —
/// every training path runs exactly this loop.
pub fn run_chunked(t: &mut dyn TrainerOps, iters: u64, cad: &Cadence, ckpt: &Path) {
    let start = Instant::now();
    let mut chunk_idx: u64 = 0;
    while t.iterations() < iters {
        let step = cad.chunk.min(iters - t.iterations());
        t.advance(step);
        t.save_checkpoint(ckpt);

        let is_last = t.iterations() >= iters;
        let expl = (chunk_idx.is_multiple_of(cad.expl_every) || is_last)
            .then(|| t.exploitability())
            .flatten();
        println!(
            "  {:>4}%  {} info sets   {}{} nodes   ({:.1}s)  [checkpointed]",
            t.iterations() * 100 / iters,
            t.info_sets(),
            expl.map(|e| format!("exploitability {:>6.1} mbb/g   ", e * 1000.0))
                .unwrap_or_default(),
            t.nodes_visited(),
            start.elapsed().as_secs_f64()
        );

        let mut fields = vec![
            ("iteration", t.iterations().to_string()),
            ("pct", (t.iterations() * 100 / iters).to_string()),
            ("info_sets", t.info_sets().to_string()),
            ("nodes", t.nodes_visited().to_string()),
            ("elapsed_s", format!("{:.3}", start.elapsed().as_secs_f64())),
        ];
        if let Some(e) = expl {
            fields.push(("exploitability_mbb", format!("{:.4}", e * 1000.0)));
        }
        emit_metric("wandb", &fields);
        chunk_idx += 1;
    }
}

/// The optional gated exploitability evaluator a [`CursorTrainer`] may carry.
pub type ExplFn<G> = Box<dyn Fn(&Mccfr<G>) -> f64>;

/// [`TrainerOps`] for the HashMap solver on the cursor fast path.  `expl` is
/// the optional gated evaluator (push/fold supplies one; the blueprint has no
/// affordable in-loop estimator — see [`crate::blueprint::run_blueprint`]).
pub struct CursorTrainer<G: Game + CursorGame> {
    pub solver: Mccfr<G>,
    pub parallel_batch: Option<u64>,
    pub expl: Option<ExplFn<G>>,
}

impl<G: Game + CursorGame + Sync> TrainerOps for CursorTrainer<G> {
    fn iterations(&self) -> u64 {
        self.solver.iterations()
    }
    fn nodes_visited(&self) -> u64 {
        self.solver.nodes_visited()
    }
    fn info_sets(&self) -> usize {
        self.solver.num_info_sets()
    }
    fn advance(&mut self, step: u64) {
        match self.parallel_batch {
            Some(batch) => self.solver.train_parallel_fast(step, batch),
            None => self.solver.train_fast(step),
        }
    }
    fn save_checkpoint(&self, path: &Path) {
        self.solver.save_checkpoint(path).expect("write checkpoint");
    }
    fn exploitability(&self) -> Option<f64> {
        self.expl.as_ref().map(|f| f(&self.solver))
    }
}

/// How the SoA solver advances — serial, deterministic mini-batch, or
/// lock-free atomic (`--atomic` takes precedence over `--parallel`).
#[derive(Clone, Copy)]
pub enum SoaMode {
    Serial,
    Parallel(u64),
    Atomic(usize),
}

impl SoaMode {
    pub fn from_flags(atomic_threads: Option<usize>, parallel_batch: Option<u64>) -> Self {
        match (atomic_threads, parallel_batch) {
            (Some(th), _) => Self::Atomic(th),
            (None, Some(b)) => Self::Parallel(b),
            (None, None) => Self::Serial,
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Atomic(th) => format!("atomic(threads={th})"),
            Self::Parallel(b) => format!("parallel(batch={b})"),
            Self::Serial => "serial".to_string(),
        }
    }
}

/// [`TrainerOps`] for the flat SoA solver.
pub struct SoaTrainer<G: IndexedGame> {
    pub solver: SoaMccfr<G>,
    pub mode: SoaMode,
}

impl<G: IndexedGame + Sync> TrainerOps for SoaTrainer<G> {
    fn iterations(&self) -> u64 {
        self.solver.iterations()
    }
    fn nodes_visited(&self) -> u64 {
        self.solver.nodes_visited()
    }
    fn info_sets(&self) -> usize {
        self.solver.capacity()
    }
    fn advance(&mut self, step: u64) {
        match self.mode {
            SoaMode::Atomic(th) => self.solver.train_atomic(step, th),
            SoaMode::Parallel(b) => self.solver.train_parallel(step, b),
            SoaMode::Serial => self.solver.train(step),
        }
    }
    fn save_checkpoint(&self, path: &Path) {
        self.solver.save_checkpoint(path).expect("write SoA checkpoint");
    }
}
