//! `play expl` — abstract-game exploitability of the blueprint.

use poker_ai::evaluation::vector_br::{all_flops, best_response_value, sample_flops};
use poker_ai::util::cli::{data_dir, flag};

use super::{cap_flag, load_game, load_policy, validate};

/// Abstract-game exploitability of the blueprint via the vectorized best
/// response (`evaluation::vector_br`): exact betting/turn/river/ranges,
/// Monte-Carlo over `--flops` sampled flops (or every flop with
/// `--flops=all`).  Runtime scales linearly in flops and parallelizes over
/// them; the river-heavy cap-3 tree costs on the order of CPU-hours per
/// hundred flops — a milestone metric, not a per-checkpoint one.
///
///   play expl [--flops=N|all --seed=S --data=DIR --stack-bb=N --cap=N|P,F,T,R --policy=PATH]
pub(super) fn run(args: &[String]) {
    validate(args, &["data", "stack-bb", "cap", "policy", "flops", "seed", "board-samples"], 0);
    let dir = data_dir(args);
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap = cap_flag(args);
    let seed: u64 = flag(args, "seed").unwrap_or(1);
    let flops = match flag::<String>(args, "flops").as_deref() {
        Some("all") => all_flops(),
        Some(n) => sample_flops(n.parse().unwrap_or(16), seed),
        None => sample_flops(16, seed),
    };
    // Turn/river runout samples per reveal. 0 = exact enumeration, which does
    // NOT finish on a real deep-stacked blueprint — guard against it there.
    let board_samples: usize = flag(args, "board-samples").unwrap_or(2);
    if board_samples == 0 && stack_bb > 6 {
        eprintln!(
            "refusing --board-samples=0 (exact turn/river enumeration) at {stack_bb}bb: \
             the deep betting tree × 48 × 44 runouts does not finish. Use a small \
             positive value (2–4) for a real blueprint; 0 is only for tiny test games."
        );
        std::process::exit(2);
    }

    println!("Loading abstraction from {} ({stack_bb}bb, cap-{cap})", dir.display());
    let game = load_game(&dir, stack_bb, cap);
    let policy = load_policy(args, &dir);
    let runout = if board_samples == 0 {
        "exact enumeration".to_string()
    } else {
        format!("{board_samples} turn/river samples per reveal")
    };
    println!(
        "  {} info sets; {} flops × {} (seed {seed}) — per-flop progress on stderr",
        policy.len(),
        flops.len(),
        runout
    );

    let t0 = std::time::Instant::now();
    let mut br_bb = [0.0f64; 2];
    for (seat, out) in br_bb.iter_mut().enumerate() {
        let t = std::time::Instant::now();
        eprintln!("  seat {seat} best response starting ...");
        *out = best_response_value(&game, &policy, seat, &flops, board_samples, seed);
        println!(
            "  BR as seat {seat} ({}) = {:+.4} bb/hand   [{:.0}s]",
            if seat == 0 { "SB/button" } else { "BB" },
            *out,
            t.elapsed().as_secs_f64()
        );
    }
    let expl_mbb = (br_bb[0] + br_bb[1]) / 2.0 * 1000.0;
    println!(
        "\nAbstract-game exploitability: {expl_mbb:.1} mbb/hand  \
         (BR sum {:.4} bb; {} flops; {:.0}s total)",
        br_bb[0] + br_bb[1],
        flops.len(),
        t0.elapsed().as_secs_f64()
    );
    println!(
        "@wandb {{\"expl_mbb\":{expl_mbb:.2},\"br_sb_bb\":{:.4},\"br_bb_bb\":{:.4},\"flops\":{}}}",
        br_bb[0],
        br_bb[1],
        flops.len()
    );
}
