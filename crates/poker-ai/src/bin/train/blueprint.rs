//! The full heads-up NLHE blueprint trainer — the production run.
//!
//! Loads the `cluster`-built card abstraction, caps the betting abstraction,
//! and trains on either the HashMap or the flat SoA store (the latter with the
//! lock-free atomic path for many-core boxes).  Checkpoints every chunk so a
//! multi-hour run survives interruption.

use std::path::Path;

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::games::blueprint::BlueprintHoldem;
use poker_ai::play::policy::write_policy;
use poker_ai::solver::variant::Variant;
use poker_ai::solver::dcfr::Discount;
use poker_ai::solver::mccfr::Mccfr;

use crate::harness::{run_chunked, CursorTrainer, SoaMode, SoaTrainer};
use crate::{data_dir, emit_metric, parse_cadence, BIG_BLIND, SMALL_BLIND};

/// Build the blueprint game at `stack` chips and `cap` raises/street, with every
/// per-street `{flop,turn,river}_buckets.bin` map found in `dir`
/// attached (a missing map leaves that street unabstracted — correct, but its
/// info sets won't plateau, which is the signal that the abstraction is needed).
pub fn load_blueprint_game(dir: &Path, stack: u32, cap: u32, verbose: bool) -> BlueprintHoldem {
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

/// Train the **full heads-up NLHE blueprint**
/// ([`BlueprintHoldem`]) — the real
/// abstracted game, the cloud-burst training target — with external-sampling
/// DCFR + the VR-MCCFR baseline, checkpointing each chunk (so a preempted spot
/// instance resumes with `--resume`).  Card abstraction is loaded from the
/// `cluster`-built `data/{flop,turn,river}_buckets.bin`; the betting abstraction
/// is capped at `--cap` raises/street (the feasibility lever — cap 1 fits a
/// 64 GB box, see `memory_estimate`).
///
/// ```text
/// train blueprint [iters] [stack_bb] [seed] [flags]
///   --cap=N            raises per street (default 1)
///   --soa              flat SoA regret store (~10× smaller; needs full-coverage
///                      maps + finite cap)
///   --optimistic       predictive regret updates (serial path only; not --soa)
///   --parallel[=BATCH] mini-batch parallel MCCFR (keeps the baseline)
///   --resume           continue from blueprint_holdem[_soa].ckpt
///   --chunk=N          iterations per progress line + checkpoint (default ~1%)
///   --data=DIR         artifact directory (default `data/`)
/// ```
///
/// Exploitability is NOT measured in-loop: the sampled best-response bound needs
/// infeasibly many samples on this tree (it read *negative* at practical
/// counts).  Measure the trained artifact with `play expl` (the vectorized
/// abstract-game BR) as a milestone metric instead.
pub fn run_blueprint(args: &[String]) {
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
        // Optimistic is serial-only: it composes poorly with batch staleness.
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

    // Persist the deployable average strategy as f32 (halves the footprint),
    // streamed rather than buffered — see `play::policy::write_policy`.
    let path = dir.join("blueprint_holdem.bin");
    let avg = trainer.solver.average_strategy();
    let n = write_policy(
        &path,
        avg.into_iter().map(|(k, v)| (k, v.into_iter().map(|x| x as f32).collect())),
    )
    .expect("write strategy");
    println!("Saved {n} info sets -> {}", path.display());
}

/// Train the heads-up NLHE blueprint with the **flat SoA regret store** instead
/// of the `HashMap` — the ~10×-smaller table that lets the cap-2
/// abstraction fit a 128 GB box.  Needs a finite `--cap` and full-coverage
/// `data/{flop,turn,river}_buckets.bin` (the dense index has one slot per
/// `(sequence, bucket)`); the game's [`BlueprintHoldem::with_indexing`] enforces
/// both.  DCFR + the VR-MCCFR baseline; `--parallel` keeps the baseline.
/// `--optimistic` is not available on this path (the SoA solver carries no
/// momentum accumulator — it stayed off to keep the store minimal).
pub fn run_blueprint_soa(args: &[String]) {
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
    //
    // Streamed, never materialized: this table has hundreds of millions of
    // visited info sets and is still resident here, so building the `HashMap`
    // and then a whole-file `Vec<u8>` was tens of GB of peak at the one moment
    // in the run where dying costs the most (`play::policy::write_policy`).
    let game = load_blueprint_game(&dir, stack, cap, false).with_indexing();
    let path = dir.join("blueprint_holdem.bin");
    let visited = (0..solver.capacity()).filter(|&i| solver.is_visited(i)).map(|i| {
        let probs = solver.average_strategy_at(i).into_iter().map(|x| x as f32).collect();
        (game.info_key_at(i), probs)
    });
    let n = write_policy(&path, visited).expect("write strategy");
    println!("Saved {n} info sets -> {}", path.display());
}

