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

mod blueprint;
mod harness;
mod pushfold;

use std::path::PathBuf;

use poker_ai::solver::pruning::PruningConfig;
use poker_ai::util::cli::validate_flags;

pub const BIG_BLIND: u32 = 2;
pub const SMALL_BLIND: u32 = 1;

/// Which Phase 3 refinements to compose onto the DCFR + baseline base.
#[derive(Clone, Copy, Default)]
pub struct RunConfig {
    pub optimistic: bool,
    pub rbp: bool,
    /// `Some(batch)` ⇒ parallel mini-batch MCCFR (the parallel path is plain
    /// external sampling — no baseline/optimistic/pruning).
    pub parallel_batch: Option<u64>,
}

/// RBP threshold tuned to push/fold's regret scale (payoffs are ±stack chips).
pub fn pushfold_pruning() -> PruningConfig {
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
pub fn emit_metric(tag: &str, fields: &[(&str, String)]) {
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
pub struct Cadence {
    pub chunk: u64,
    pub expl_every: u64,
}

/// Parse `--chunk=N` (iterations per progress line; default ~1% of `iters`) and
/// `--expl-every=N` (run the exploitability eval every Nth chunk; default 10).
pub fn parse_cadence(args: &[String], iters: u64) -> Cadence {
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
pub fn data_dir(args: &[String]) -> PathBuf {
    args.iter()
        .find_map(|a| a.strip_prefix("--data="))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data"))
}

/// Cadence/output flags every mode understands.
const COMMON_FLAGS: [&str; 3] = ["chunk", "expl-every", "data"];

/// Validate `args` against `allowed` plus [`COMMON_FLAGS`].  A training run
/// costs hours, so an unrecognised flag must fail now rather than silently
/// train the wrong configuration.
fn check_flags(args: &[String], skip: usize, allowed: &[&str], positionals: usize) {
    let mut all: Vec<&str> = allowed.to_vec();
    all.extend_from_slice(&COMMON_FLAGS);
    validate_flags(args, skip, &all, positionals);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Positionals for every mode are [iters] [stack_bb] [seed].
    match args.get(1).map(|s| s.as_str()) {
        Some("compare") => {
            check_flags(&args, 2, &[], 3);
            pushfold::run_comparison(&args)
        }
        Some("blueprint") => {
            let allowed = ["cap", "soa", "atomic", "parallel", "optimistic", "resume"];
            check_flags(&args, 2, &allowed, 3);
            blueprint::run_blueprint(&args)
        }
        _ if args.iter().any(|a| a == "--soa") => {
            check_flags(&args, 1, &["soa", "atomic", "parallel"], 3);
            pushfold::run_soa(&args)
        }
        _ => {
            check_flags(&args, 1, &["optimistic", "rbp", "parallel", "resume"], 3);
            pushfold::run(&args)
        }
    }
}
