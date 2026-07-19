//! Blueprint training entrypoint (Phase 1.5 / Phase 3).
//!
//! Trains the first *converging* heads-up blueprint — a push/fold NLHE strategy
//! over the real `poker-core` engine — with external-sampling DCFR and writes
//! the average strategy to `data/blueprint_pushfold.bin`.
//!
//! Push/fold is the right first target: it has no postflop, so it converges
//! without the cloud-scale card abstraction (see
//! [`poker_ai::games::push_fold`]).  The full-game blueprint
//! ([`poker_ai::games::blueprint`]) reuses this exact training loop once a
//! complete postflop abstraction is built; only the `Game` changes.
//!
//! Usage:
//!   train [iters] [stack_bb] [seed] [flags]
//!     iters     MCCFR iterations           (default 1_000_000)
//!     stack_bb  starting stack, big blinds (default 20)
//!     seed      RNG seed                   (default 1)
//!   flags (compose the Phase 3 algorithm stack onto the DCFR+baseline base):
//!     --optimistic       predictive regret updates (R += 2rₜ − r_{t−1})
//!     --rbp              Regret-Based Pruning
//!     --parallel[=BATCH] mini-batch parallel MCCFR (plain external sampling)
//!     --atomic[=THREADS] (with --soa) lock-free atomic training — near-linear
//!                        core scaling, NOT bit-deterministic (default: all cores)
//!     --chunk=N          iterations per progress line + checkpoint (default ~1%)
//!     --expl-every=N     run the exploitability eval only every Nth chunk (def 10)
//!     --data=DIR         artifact directory for maps/checkpoints/blueprints
//!                        (default `data/` — point it at scratch space on
//!                        quota-limited boxes)
//!
//!   train compare [iters] [stack_bb] [seed]
//!     Trains the base config and each Phase 3 feature in turn and prints a
//!     recorded before/after table (final exploitability, wall-time, node
//!     visits) — the evidence the features pay off on the real trainer.
//!
//!   train blueprint [iters] [stack_bb] [seed] [flags]
//!     Trains the full abstracted heads-up NLHE blueprint ([`BlueprintHoldem`]),
//!     loading the `cluster`-built card abstraction from data/ and capping the
//!     betting abstraction at `--cap=N` raises/street (default 1 — the memory
//!     feasibility lever).  See [`run_blueprint`] for the full flag list.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::abstraction::canonical::preflop_index;
use poker_ai::evaluation::exploitability::push_fold_exploitability;
use poker_ai::games::blueprint::BlueprintHoldem;
use poker_ai::games::push_fold::PushFoldHoldem;
use poker_ai::games::{CursorGame, Game, IndexedGame};
use poker_ai::solver::cfr::Variant;
use poker_ai::solver::dcfr::Discount;
use poker_ai::solver::mccfr::{Mccfr, SoaMccfr};
use poker_ai::solver::pruning::PruningConfig;

const BIG_BLIND: u32 = 2;
const SMALL_BLIND: u32 = 1;

/// Which Phase 3 refinements to compose onto the DCFR + baseline base.
#[derive(Clone, Copy, Default)]
struct RunConfig {
    optimistic: bool,
    rbp: bool,
    /// `Some(batch)` ⇒ parallel mini-batch MCCFR (the parallel path is plain
    /// external sampling — no baseline/optimistic/pruning).
    parallel_batch: Option<u64>,
}

/// RBP threshold tuned to push/fold's regret scale (payoffs are ±stack chips).
fn pushfold_pruning() -> PruningConfig {
    PruningConfig { theta: -5_000.0, k: 100, start_fraction: 0.2, refresh_interval: 10_000 }
}

/// Emit one machine-readable JSON metrics line for an external experiment
/// tracker — the `scripts/train_wandb.py` Weights & Biases logger parses these.
/// A **no-op** unless `POKER_AI_METRICS` is set in the environment, so plain
/// `train` runs are byte-identical to before (the wrapper sets the var).
///
/// `tag` is the line prefix (`wandb-config` once at startup, `wandb` per
/// checkpoint); each `value` must already be a valid JSON literal (numbers bare,
/// strings quoted — use `format!("{s:?}")` for a `String`).
fn emit_metric(tag: &str, fields: &[(&str, String)]) {
    if std::env::var_os("POKER_AI_METRICS").is_none() {
        return;
    }
    let body =
        fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(",");
    println!("@{tag} {{{body}}}");
}

/// Progress + evaluation cadence for the training loops.
///
/// The trainer reports once per `chunk` iterations (and checkpoints there, so an
/// interruption costs ≤ one chunk).  The default is ~1% of the run — frequent
/// progress lines instead of the old `iters/10` (which left a 300 M-iter run
/// silent for tens of minutes before its first line).  The push/fold
/// exploitability eval (a 2 M-deal MC best response) adds up across ~100
/// progress lines, so it runs only every `expl_every`-th chunk (plus always on
/// the final one).
struct Cadence {
    chunk: u64,
    expl_every: u64,
}

/// Parse `--chunk=N` (iterations per progress line; default ~1% of `iters`) and
/// `--expl-every=N` (run the exploitability eval every Nth chunk; default 10).
fn parse_cadence(args: &[String], iters: u64) -> Cadence {
    let chunk = args
        .iter()
        .find_map(|a| a.strip_prefix("--chunk="))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or((iters / 100).max(1))
        .max(1);
    let expl_every = args
        .iter()
        .find_map(|a| a.strip_prefix("--expl-every="))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10)
        .max(1);
    Cadence { chunk, expl_every }
}

/// Artifact directory (`--data=DIR`, default `data/`): where bucket maps are
/// loaded from and checkpoints/blueprints are written.  Overridable so a
/// quota-limited box can point the bulk artifacts at scratch space without a
/// symlink.
fn data_dir(args: &[String]) -> PathBuf {
    args.iter()
        .find_map(|a| a.strip_prefix("--data="))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data"))
}

// ---------------------------------------------------------------------------
// The chunked training loop, shared by every trainer configuration.
// ---------------------------------------------------------------------------

/// What the shared loop needs from a trainer.  Implemented once for the
/// HashMap/cursor solver ([`CursorTrainer`]) and once for the flat SoA solver
/// ([`SoaTrainer`]), each generic over the game — so all four CLI paths
/// (HashMap/SoA × push-fold/blueprint) drive the same loop.
trait TrainerOps {
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
fn run_chunked(t: &mut dyn TrainerOps, iters: u64, cad: &Cadence, ckpt: &Path) {
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
type ExplFn<G> = Box<dyn Fn(&Mccfr<G>) -> f64>;

/// [`TrainerOps`] for the HashMap solver on the cursor fast path.  `expl` is
/// the optional gated evaluator (push/fold supplies one; the blueprint has no
/// affordable in-loop estimator — see [`run_blueprint`]).
struct CursorTrainer<G: Game + CursorGame> {
    solver: Mccfr<G>,
    parallel_batch: Option<u64>,
    expl: Option<ExplFn<G>>,
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
enum SoaMode {
    Serial,
    Parallel(u64),
    Atomic(usize),
}

impl SoaMode {
    fn from_flags(atomic_threads: Option<usize>, parallel_batch: Option<u64>) -> Self {
        match (atomic_threads, parallel_batch) {
            (Some(th), _) => Self::Atomic(th),
            (None, Some(b)) => Self::Parallel(b),
            (None, None) => Self::Serial,
        }
    }

    fn label(self) -> String {
        match self {
            Self::Atomic(th) => format!("atomic(threads={th})"),
            Self::Parallel(b) => format!("parallel(batch={b})"),
            Self::Serial => "serial".to_string(),
        }
    }
}

/// [`TrainerOps`] for the flat SoA solver.
struct SoaTrainer<G: IndexedGame> {
    solver: SoaMccfr<G>,
    mode: SoaMode,
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

/// Build a (fresh, untrained) solver with `cfg` applied.
fn build_solver(stack: u32, seed: u64, iters: u64, cfg: RunConfig) -> Mccfr<PushFoldHoldem> {
    let game = PushFoldHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0);
    let mut solver = Mccfr::with_seed(game, Variant::Dcfr(Discount::RECOMMENDED), seed);
    // The parallel path can't use the serial-only refinements, so only enable
    // the baseline / optimistic / RBP stack on the serial path.
    if cfg.parallel_batch.is_none() {
        solver = solver.with_baseline();
        if cfg.optimistic {
            solver = solver.with_optimistic();
        }
        if cfg.rbp {
            solver = solver.with_pruning(pushfold_pruning(), iters);
        }
    }
    solver
}

/// Run `solver` for the chosen number of `iters` in one shot; return the average
/// strategy, wall-time, and node visits (used by the comparison harness).
fn train_with(
    stack: u32,
    seed: u64,
    iters: u64,
    cfg: RunConfig,
) -> (HashMap<u64, Vec<f64>>, std::time::Duration, u64) {
    let mut solver = build_solver(stack, seed, iters, cfg);
    let start = Instant::now();
    // Cursor fast path: zero per-node allocation, bit-identical to train/
    // train_parallel for a fixed seed (PushFoldHoldem implements CursorGame).
    match cfg.parallel_batch {
        Some(batch) => solver.train_parallel_fast(iters, batch),
        None => solver.train_fast(iters),
    }
    (solver.average_strategy(), start.elapsed(), solver.nodes_visited())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("compare") {
        run_comparison(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("blueprint") {
        run_blueprint(&args);
        return;
    }
    if args.iter().any(|a| a == "--soa") {
        run_soa(&args);
        return;
    }

    // Positional args are the numeric ones; flags start with `--`.
    let nums: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    let iters: u64 = nums.first().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let stack_bb: u32 = nums.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = nums.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let stack = stack_bb * BIG_BLIND;

    let parallel_batch = args.iter().find_map(|a| {
        a.strip_prefix("--parallel").map(|rest| rest.strip_prefix('=').and_then(|b| b.parse().ok()).unwrap_or(256))
    });
    let cfg = RunConfig {
        optimistic: args.iter().any(|a| a == "--optimistic"),
        rbp: args.iter().any(|a| a == "--rbp"),
        parallel_batch,
    };
    let resume = args.iter().any(|a| a == "--resume");

    let mut features = Vec::new();
    if cfg.optimistic {
        features.push("optimistic".to_string());
    }
    if cfg.rbp {
        features.push("rbp".to_string());
    }
    if let Some(b) = cfg.parallel_batch {
        features.push(format!("parallel(batch={b})"));
    }
    let feat = if features.is_empty() { "DCFR+baseline".into() } else { features.join("+") };

    let dir = data_dir(&args);
    std::fs::create_dir_all(&dir).expect("create data directory");
    let ckpt_path = dir.join("blueprint_pushfold.ckpt");

    // Build fresh, or resume the full solver state from a checkpoint so an
    // interrupted run continues exactly where it stopped (the config — variant,
    // baseline/optimistic/pruning — is restored from the checkpoint).
    let solver = if resume && ckpt_path.exists() {
        let game = PushFoldHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0);
        let s = Mccfr::load_checkpoint(&ckpt_path, game).expect("load checkpoint");
        println!(
            "Resuming from {} at iteration {} ({} info sets)",
            ckpt_path.display(),
            s.iterations(),
            s.num_info_sets()
        );
        s
    } else {
        println!(
            "Training heads-up push/fold blueprint: {iters} iters, {stack_bb}bb stacks, seed {seed} [{feat}]"
        );
        build_solver(stack, seed, iters, cfg)
    };

    emit_metric(
        "wandb-config",
        &[
            ("mode", "\"pushfold\"".into()),
            ("iters", iters.to_string()),
            ("stack_bb", stack_bb.to_string()),
            ("seed", seed.to_string()),
            ("resume", resume.to_string()),
            ("features", format!("{feat:?}")),
        ],
    );

    // Exploitability is the validation number, but the 2 M-deal MC best
    // response adds up across ~100 progress lines — hence the expl_every gate
    // inside the shared loop.
    let eval_game = PushFoldHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0);
    let mut trainer = CursorTrainer {
        solver,
        parallel_batch: cfg.parallel_batch,
        expl: Some(Box::new(move |s: &Mccfr<PushFoldHoldem>| {
            push_fold_exploitability(&eval_game, &s.average_strategy(), 2_000_000, 7)
        })),
    };
    run_chunked(&mut trainer, iters, &parse_cadence(&args, iters), &ckpt_path);

    // Persist the average strategy as f32 (deploy-ready; halves the footprint).
    let avg: HashMap<u64, Vec<f32>> = trainer
        .solver
        .average_strategy()
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|x| x as f32).collect()))
        .collect();
    let path = dir.join("blueprint_pushfold.bin");
    let bytes = bincode::serialize(&avg).expect("serialize strategy");
    std::fs::write(&path, &bytes).expect("write strategy");

    println!("Saved {} info sets, {} bytes -> {}", avg.len(), bytes.len(), path.display());

    print_shove_chart(stack, &avg);
}

/// Train the base config and each Phase 3 refinement in turn, printing a recorded
/// before/after table.  This is the evidence that composing optimistic updates,
/// RBP, and parallelization onto the real trainer actually pays off (or, on a
/// tree as small as push/fold, where it does and does not move the needle).
fn run_comparison(args: &[String]) {
    let iters: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let stack_bb: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let stack = stack_bb * BIG_BLIND;
    let eval_game = PushFoldHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0);
    let expl_deals = 2_000_000;

    println!("Phase 3 feature comparison: {iters} iters, {stack_bb}bb, seed {seed}");
    println!("(exploitability = exact-style MC best response of the average strategy)\n");
    println!("{:<28}{:>14}{:>16}{:>10}", "config", "expl (mbb/g)", "node visits", "time (s)");
    println!("{}", "-".repeat(68));

    let configs: [(&str, RunConfig); 5] = [
        ("DCFR + baseline (base)", RunConfig::default()),
        ("+ optimistic", RunConfig { optimistic: true, ..Default::default() }),
        ("+ RBP", RunConfig { rbp: true, ..Default::default() }),
        ("+ optimistic + RBP", RunConfig { optimistic: true, rbp: true, ..Default::default() }),
        ("parallel (batch=256, plain)", RunConfig { parallel_batch: Some(256), ..Default::default() }),
    ];

    for (label, cfg) in configs {
        let (avg, elapsed, nodes) = train_with(stack, seed, iters, cfg);
        let expl = push_fold_exploitability(&eval_game, &avg, expl_deals, 7);
        println!(
            "{:<28}{:>14.1}{:>16}{:>10.2}",
            label,
            expl * 1000.0,
            nodes,
            elapsed.as_secs_f64()
        );
    }
}

/// Train push/fold on the flat **SoA** [`RegretTable`] store (the ~10×-smaller
/// blueprint storage), via `--soa`.  Uses DCFR + the VR-MCCFR baseline; `--parallel`
/// uses the SoA parallel path (which keeps the baseline).
fn run_soa(args: &[String]) {
    let nums: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    let iters: u64 = nums.first().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let stack_bb: u32 = nums.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = nums.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let stack = stack_bb * BIG_BLIND;
    let parallel_batch = args.iter().find_map(|a| {
        a.strip_prefix("--parallel").map(|r| r.strip_prefix('=').and_then(|b| b.parse().ok()).unwrap_or(256))
    });
    let atomic_threads: Option<usize> = args.iter().find_map(|a| {
        a.strip_prefix("--atomic").map(|rest| {
            rest.strip_prefix('=').and_then(|n| n.parse().ok()).unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
            })
        })
    });

    let mode = SoaMode::from_flags(atomic_threads, parallel_batch);
    println!(
        "Training push/fold via flat SoA RegretTable: {iters} iters, {stack_bb}bb, seed {seed} [{}]",
        mode.label()
    );
    emit_metric(
        "wandb-config",
        &[
            ("mode", "\"pushfold-soa\"".into()),
            ("iters", iters.to_string()),
            ("stack_bb", stack_bb.to_string()),
            ("seed", seed.to_string()),
            ("features", format!("{:?}", mode.label())),
        ],
    );
    let solver = SoaMccfr::with_seed(
        PushFoldHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0),
        Variant::Dcfr(Discount::RECOMMENDED),
        seed,
    )
    .with_baseline();
    println!("Flat table: {} bytes/info set (vs ~350 for the HashMap Node)", solver.bytes_per_info_set());

    let dir = data_dir(args);
    std::fs::create_dir_all(&dir).expect("create data directory");
    let ckpt = dir.join("blueprint_pushfold_soa.ckpt");
    let mut trainer = SoaTrainer { solver, mode };
    run_chunked(&mut trainer, iters, &parse_cadence(args, iters), &ckpt);

    // SB opening shove = info set (sequence 0, preflop class) = preflop_index.
    print_chart(stack, |c0, c1| {
        trainer.solver.average_strategy_at(preflop_index(&[c0, c1]) as usize)[1] as f32
    });
}

/// Load the abstracted heads-up [`BlueprintHoldem`] for a real full-game training
/// run: equal `stack` chips, raise abstraction capped at `cap` raises/street, and
/// whatever per-street `{flop,turn,river}_buckets.bin` maps exist in `dir`
/// attached (a missing map leaves that street unabstracted — correct, but its
/// info sets won't plateau, which is the signal that the abstraction is needed).
fn load_blueprint_game(dir: &Path, stack: u32, cap: u32, verbose: bool) -> BlueprintHoldem {
    let mut game = BlueprintHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0).with_raise_cap(cap);
    for (street, name) in [(0usize, "flop"), (1, "turn"), (2, "river")] {
        let path = dir.join(format!("{name}_buckets.bin"));
        match BucketMap::load(&path) {
            Ok(map) => {
                if verbose {
                    println!("  {name}: {} buckets loaded from {}", map.num_buckets(), path.display());
                }
                game = game.with_street_bucket(street, map);
            }
            Err(_) if verbose => {
                println!("  {name}: no abstraction at {} — street stays unabstracted", path.display());
            }
            Err(_) => {}
        }
    }
    game
}

/// Train the **full heads-up NLHE blueprint** ([`BlueprintHoldem`]) — the real
/// abstracted game, the cloud-burst training target — with external-sampling
/// DCFR + the VR-MCCFR baseline, checkpointing each chunk (so a preempted spot
/// instance resumes with `--resume`).  Card abstraction is loaded from the
/// `cluster`-built `data/{flop,turn,river}_buckets.bin`; the betting abstraction
/// is capped at `--cap` raises/street (the feasibility lever — cap 1 fits a
/// 64 GB box, see `memory_estimate`).
///
/// Usage:
///   train blueprint [iters] [stack_bb] [seed] [flags]
///     --cap=N            raises per street (default 1)
///     --soa              flat SoA regret store (~10× smaller; needs full-coverage
///                        maps + finite cap — the cap-2/128 GB path, finding #4)
///     --optimistic       predictive regret updates (serial path only; not --soa)
///     --parallel[=BATCH] mini-batch parallel MCCFR (keeps the baseline)
///     --resume           continue from blueprint_holdem[_soa].ckpt
///     --chunk=N          iterations per progress line + checkpoint (default ~1%)
///     --data=DIR         artifact directory (default `data/`)
///
/// Exploitability is NOT measured in-loop: the sampled best-response bound needs
/// infeasibly many samples on this tree (it read *negative* at practical
/// counts).  Measure the trained artifact with `play expl` (the vectorized
/// abstract-game BR) as a milestone metric instead.
fn run_blueprint(args: &[String]) {
    if args.iter().any(|a| a == "--soa") {
        run_blueprint_soa(args);
        return;
    }
    let nums: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
    let iters: u64 = nums.first().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let stack_bb: u32 = nums.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = nums.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let stack = stack_bb * BIG_BLIND;

    let flag = |name: &str| args.iter().any(|a| a == name);
    let cap: u32 =
        args.iter().find_map(|a| a.strip_prefix("--cap=")).and_then(|s| s.parse().ok()).unwrap_or(1);
    let parallel_batch = args.iter().find_map(|a| {
        a.strip_prefix("--parallel")
            .map(|rest| rest.strip_prefix('=').and_then(|b| b.parse().ok()).unwrap_or(256))
    });
    let optimistic = flag("--optimistic");
    let resume = flag("--resume");

    let mut features = vec![format!("cap={cap}")];
    if optimistic {
        features.push("optimistic".into());
    }
    if let Some(b) = parallel_batch {
        features.push(format!("parallel(batch={b})"));
    }
    let feat = features.join("+");

    let dir = data_dir(args);
    std::fs::create_dir_all(&dir).expect("create data directory");
    let ckpt_path = dir.join("blueprint_holdem.ckpt");

    println!(
        "Training heads-up NLHE blueprint: {iters} iters, {stack_bb}bb stacks, seed {seed} [DCFR+baseline+{feat}]"
    );
    let solver = if resume && ckpt_path.exists() {
        let game = load_blueprint_game(&dir, stack, cap, true);
        let s = Mccfr::load_checkpoint(&ckpt_path, game).expect("load checkpoint");
        println!("Resuming from {} at iteration {}", ckpt_path.display(), s.iterations());
        s
    } else {
        let game = load_blueprint_game(&dir, stack, cap, true);
        let mut s = Mccfr::with_seed(game, Variant::Dcfr(Discount::RECOMMENDED), seed).with_baseline();
        // Optimistic is serial-only (composes poorly with batch staleness — Step 15).
        if parallel_batch.is_none() && optimistic {
            s = s.with_optimistic();
        }
        s
    };

    emit_metric(
        "wandb-config",
        &[
            ("mode", "\"blueprint\"".into()),
            ("iters", iters.to_string()),
            ("stack_bb", stack_bb.to_string()),
            ("seed", seed.to_string()),
            ("raise_cap", cap.to_string()),
            ("resume", resume.to_string()),
            ("features", format!("{feat:?}")),
        ],
    );

    let mut trainer = CursorTrainer { solver, parallel_batch, expl: None };
    run_chunked(&mut trainer, iters, &parse_cadence(args, iters), &ckpt_path);

    // Persist the deployable average strategy as f32 (halves the footprint).
    let avg: HashMap<u64, Vec<f32>> = trainer
        .solver
        .average_strategy()
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|x| x as f32).collect()))
        .collect();
    let path = dir.join("blueprint_holdem.bin");
    let bytes = bincode::serialize(&avg).expect("serialize strategy");
    std::fs::write(&path, &bytes).expect("write strategy");
    println!("Saved {} info sets, {} bytes -> {}", avg.len(), bytes.len(), path.display());
}

/// Train the heads-up NLHE blueprint with the **flat SoA regret store** instead
/// of the `HashMap` — the ~10×-smaller table (finding #4) that lets the cap-2
/// abstraction fit a 128 GB box.  Needs a finite `--cap` and full-coverage
/// `data/{flop,turn,river}_buckets.bin` (the dense index has one slot per
/// `(sequence, bucket)`); the game's [`BlueprintHoldem::with_indexing`] enforces
/// both.  DCFR + the VR-MCCFR baseline; `--parallel` keeps the baseline.
/// `--optimistic` is not available on this path (the SoA solver carries no
/// momentum accumulator — it stayed off to keep the store minimal).
fn run_blueprint_soa(args: &[String]) {
    use poker_ai::solver::mccfr::SoaMccfr;

    let nums: Vec<&String> = args[2..].iter().filter(|a| !a.starts_with("--")).collect();
    let iters: u64 = nums.first().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let stack_bb: u32 = nums.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let seed: u64 = nums.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let stack = stack_bb * BIG_BLIND;

    let cap: u32 =
        args.iter().find_map(|a| a.strip_prefix("--cap=")).and_then(|s| s.parse().ok()).unwrap_or(1);
    let parallel_batch = args.iter().find_map(|a| {
        a.strip_prefix("--parallel")
            .map(|rest| rest.strip_prefix('=').and_then(|b| b.parse().ok()).unwrap_or(256))
    });
    // Lock-free atomic training (`--atomic[=THREADS]`): the many-core path —
    // near-linear scaling, NOT bit-deterministic (see SoaMccfr::train_atomic).
    // Takes precedence over --parallel.
    let atomic_threads: Option<usize> = args.iter().find_map(|a| {
        a.strip_prefix("--atomic").map(|rest| {
            rest.strip_prefix('=').and_then(|n| n.parse().ok()).unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
            })
        })
    });
    let resume = args.iter().any(|a| a == "--resume");

    let mode = SoaMode::from_flags(atomic_threads, parallel_batch);
    let feat = format!("cap={cap}+soa+{}", mode.label());

    let dir = data_dir(args);
    std::fs::create_dir_all(&dir).expect("create data directory");
    let ckpt_path = dir.join("blueprint_holdem_soa.ckpt");

    println!(
        "Training heads-up NLHE blueprint (SoA store): {iters} iters, {stack_bb}bb stacks, seed {seed} [DCFR+baseline+{feat}]"
    );
    let solver = if resume && ckpt_path.exists() {
        let game = load_blueprint_game(&dir, stack, cap, true).with_indexing();
        let s = SoaMccfr::load_checkpoint(&ckpt_path, game).expect("load SoA checkpoint");
        println!("Resuming from {} at iteration {}", ckpt_path.display(), s.iterations());
        s
    } else {
        let game = load_blueprint_game(&dir, stack, cap, true).with_indexing();
        SoaMccfr::with_seed(game, Variant::Dcfr(Discount::RECOMMENDED), seed).with_baseline()
    };
    println!(
        "Flat table: {} info sets, {} bytes/info set (vs ~350 for the HashMap Node)",
        solver.capacity(),
        solver.bytes_per_info_set()
    );

    emit_metric(
        "wandb-config",
        &[
            ("mode", "\"blueprint-soa\"".into()),
            ("iters", iters.to_string()),
            ("stack_bb", stack_bb.to_string()),
            ("seed", seed.to_string()),
            ("raise_cap", cap.to_string()),
            ("info_sets", solver.capacity().to_string()),
            ("resume", resume.to_string()),
            ("features", format!("{feat:?}")),
        ],
    );

    let mut trainer = SoaTrainer { solver, mode };
    run_chunked(&mut trainer, iters, &parse_cadence(args, iters), &ckpt_path);
    let solver = trainer.solver;

    // Export the deployable average strategy in the SAME HashMap<u64, Vec<f32>>
    // format the HashMap path writes (keys reconstructed via info_key_at), so the
    // artifact is interchangeable; only visited info sets are emitted.
    let game = load_blueprint_game(&dir, stack, cap, false).with_indexing();
    let mut avg: HashMap<u64, Vec<f32>> = HashMap::new();
    for i in 0..solver.capacity() {
        if solver.is_visited(i) {
            let probs = solver.average_strategy_at(i).into_iter().map(|x| x as f32).collect();
            avg.insert(game.info_key_at(i), probs);
        }
    }
    let path = dir.join("blueprint_holdem.bin");
    let bytes = bincode::serialize(&avg).expect("serialize strategy");
    std::fs::write(&path, &bytes).expect("write strategy");
    println!("Saved {} info sets, {} bytes -> {}", avg.len(), bytes.len(), path.display());
}

/// Render the SB opening shove range as a 13×13 grid (upper triangle = suited)
/// from a HashMap-keyed average strategy.
fn print_shove_chart(stack: u32, avg: &HashMap<u64, Vec<f32>>) {
    // The SB opening info key for a concrete two-card hand (player 0, empty
    // history), via the same helper the solver keys on.
    print_chart(stack, |c0, c1| {
        let key = PushFoldHoldem::preflop_key(0, &[c0, c1], &[]);
        avg.get(&key).map(|p| p[1]).unwrap_or(0.0)
    });
}

/// Render the SB opening shove range as a 13×13 grid given a `shove(c0, c1)`
/// probability lookup.  A quick eyeball check that the blueprint looks like a
/// real push/fold chart: monotone, premiums always shoving, trash folding.
fn print_chart(stack: u32, shove: impl Fn(u8, u8) -> f32) {
    use poker_core::make_card;
    const R: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

    println!("\nSB opening shove % at {}bb (upper triangle suited):", stack / BIG_BLIND);
    print!("    ");
    for &c in R.iter().rev() {
        print!(" {c} ");
    }
    println!();
    for (ri, &rr) in R.iter().enumerate().rev() {
        print!("  {rr} ");
        for ci in (0..R.len()).rev() {
            let (hi, lo) = (ri.max(ci), ri.min(ci));
            let suited = ci > ri; // upper triangle
            let (c0, c1) = if suited {
                (make_card(hi as u8, 0), make_card(lo as u8, 0))
            } else {
                (make_card(hi as u8, 0), make_card(lo as u8, 1))
            };
            let p = shove(c0, c1);
            let g = if p > 0.8 {
                '#'
            } else if p > 0.4 {
                '+'
            } else if p > 0.05 {
                '.'
            } else {
                ' '
            };
            print!(" {g} ");
        }
        println!();
    }
    println!("(# >80%   + >40%   . >5%)");
}
