//! The push/fold trainer: the fast-converging validation target.
//!
//! No postflop, so it converges on a laptop in seconds and has a real
//! exploitability number to gate against — which is why it, not the blueprint,
//! carries the in-loop evaluation and the before/after feature comparison.

use std::collections::HashMap;
use std::time::Instant;

use poker_ai::abstraction::canonical::preflop_index;
use poker_ai::evaluation::exploitability::push_fold_exploitability;
use poker_ai::games::push_fold::PushFoldHoldem;
use poker_ai::solver::variant::Variant;
use poker_ai::solver::dcfr::Discount;
use poker_ai::solver::mccfr::{Mccfr, SoaMccfr};

use crate::harness::{run_chunked, CursorTrainer, SoaMode, SoaTrainer};
use crate::{data_dir, emit_metric, parse_cadence, pushfold_pruning, RunConfig, BIG_BLIND, SMALL_BLIND};

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

/// The default mode: train push/fold and print the shove chart.  Subcommand
/// dispatch lives in `main`.
pub fn run(args: &[String]) {
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

    let dir = data_dir(args);
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
    run_chunked(&mut trainer, iters, &parse_cadence(args, iters), &ckpt_path);

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
pub fn run_comparison(args: &[String]) {
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
pub fn run_soa(args: &[String]) {
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

