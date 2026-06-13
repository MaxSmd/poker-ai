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
//!
//!   train compare [iters] [stack_bb] [seed]
//!     Trains the base config and each Phase 3 feature in turn and prints a
//!     recorded before/after table (final exploitability, wall-time, node
//!     visits) — the evidence the features pay off on the real trainer.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use poker_ai::abstraction::canonical::preflop_index;
use poker_ai::evaluation::exploitability::push_fold_exploitability;
use poker_ai::games::push_fold::PushFoldHoldem;
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

/// Advance `solver` by `step` iterations using the configured training path.
fn train_step(solver: &mut Mccfr<PushFoldHoldem>, step: u64, cfg: RunConfig) {
    match cfg.parallel_batch {
        Some(batch) => solver.train_parallel_fast(step, batch),
        None => solver.train_fast(step),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("compare") {
        run_comparison(&args);
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

    let dir = Path::new("data");
    std::fs::create_dir_all(dir).expect("create data/ directory");
    let ckpt_path = dir.join("blueprint_pushfold.ckpt");

    // Build fresh, or resume the full solver state from a checkpoint so an
    // interrupted run continues exactly where it stopped (the config — variant,
    // baseline/optimistic/pruning — is restored from the checkpoint).
    let mut solver = if resume && ckpt_path.exists() {
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

    let eval_game = PushFoldHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0);
    let expl_deals = 2_000_000;

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

    // Train in chunks, checkpointing after each so an interruption costs at most
    // one chunk of work.  Resume picks up from `solver.iterations()`.
    let start = Instant::now();
    let chunk = (iters / 10).max(1);
    while solver.iterations() < iters {
        let step = chunk.min(iters - solver.iterations());
        train_step(&mut solver, step, cfg);
        solver.save_checkpoint(&ckpt_path).expect("write checkpoint");
        let expl = push_fold_exploitability(&eval_game, &solver.average_strategy(), expl_deals, 7);
        println!(
            "  {:>4}%  {} info sets   exploitability {:>6.1} mbb/g   {} nodes   ({:.1}s)  [checkpointed]",
            solver.iterations() * 100 / iters,
            solver.num_info_sets(),
            expl * 1000.0,
            solver.nodes_visited(),
            start.elapsed().as_secs_f64()
        );
        emit_metric(
            "wandb",
            &[
                ("iteration", solver.iterations().to_string()),
                ("pct", (solver.iterations() * 100 / iters).to_string()),
                ("info_sets", solver.num_info_sets().to_string()),
                ("exploitability_mbb", format!("{:.4}", expl * 1000.0)),
                ("nodes", solver.nodes_visited().to_string()),
                ("elapsed_s", format!("{:.3}", start.elapsed().as_secs_f64())),
            ],
        );
    }

    // Persist the average strategy as f32 (deploy-ready; halves the footprint).
    let avg: HashMap<u64, Vec<f32>> = solver
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

    let mode = parallel_batch.map_or("serial".to_string(), |b| format!("parallel(batch={b})"));
    println!("Training push/fold via flat SoA RegretTable: {iters} iters, {stack_bb}bb, seed {seed} [{mode}]");
    emit_metric(
        "wandb-config",
        &[
            ("mode", "\"pushfold-soa\"".into()),
            ("iters", iters.to_string()),
            ("stack_bb", stack_bb.to_string()),
            ("seed", seed.to_string()),
            ("features", format!("{mode:?}")),
        ],
    );
    let mut solver = SoaMccfr::with_seed(
        PushFoldHoldem::new(stack, BIG_BLIND, SMALL_BLIND, 0),
        Variant::Dcfr(Discount::RECOMMENDED),
        seed,
    )
    .with_baseline();
    println!("Flat table: {} bytes/info set (vs ~350 for the HashMap Node)", solver.bytes_per_info_set());

    let dir = Path::new("data");
    std::fs::create_dir_all(dir).expect("create data/");
    let ckpt = dir.join("blueprint_pushfold_soa.ckpt");
    let start = Instant::now();
    let chunk = (iters / 10).max(1);
    while solver.iterations() < iters {
        let step = chunk.min(iters - solver.iterations());
        match parallel_batch {
            Some(b) => solver.train_parallel(step, b),
            None => solver.train(step),
        }
        solver.save_checkpoint(&ckpt).expect("write SoA checkpoint");
        println!(
            "  {:>4}%  {} nodes  ({:.1}s)  [checkpointed]",
            solver.iterations() * 100 / iters,
            solver.nodes_visited(),
            start.elapsed().as_secs_f64()
        );
        emit_metric(
            "wandb",
            &[
                ("iteration", solver.iterations().to_string()),
                ("pct", (solver.iterations() * 100 / iters).to_string()),
                ("nodes", solver.nodes_visited().to_string()),
                ("elapsed_s", format!("{:.3}", start.elapsed().as_secs_f64())),
            ],
        );
    }

    // SB opening shove = info set (sequence 0, preflop class) = preflop_index.
    print_chart(stack, |c0, c1| solver.average_strategy_at(preflop_index(&[c0, c1]) as usize)[1] as f32);
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
