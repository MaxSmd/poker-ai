//! `play lbr` — local best response against the live agent.

use poker_ai::util::cli::{data_dir, flag};

use super::{build_bot, hands_arg, validate, BOT_FLAGS};

/// Exploitability lower bound of the **live agent** by local best response
/// (`evaluation::lbr`).  Unlike `expl`, this exercises the whole playing stack —
/// blueprint lookups, range tracking, purification and per-decision re-solving —
/// because LBR plays real hands against it rather than best-responding to a
/// strategy file.
///
/// ```text
/// play lbr [hands] [--lbr-seed=S] [bot flags]
/// ```
pub(super) fn run(args: &[String]) {
    validate(args, &[BOT_FLAGS, &["lbr-seed"]].concat(), 1);
    let hands = hands_arg(args);
    let dir = data_dir(args);
    let mut bot = build_bot(args, &dir);
    let lbr_seed: u64 = flag(args, "lbr-seed").unwrap_or(1);

    println!("\nLocal best response: {hands} hands, seats alternating, seed {lbr_seed}");
    println!("LBR's win rate is a LOWER BOUND on the agent's exploitability.\n");
    let t0 = std::time::Instant::now();
    let out = poker_ai::evaluation::lbr::run_lbr(&mut bot, hands, lbr_seed, |o| {
        println!(
            "  {:>6} hands   LBR {:>8.1} ± {:.1} bb/100   ({:.0} mbb/hand)   luck-adj {:>8.1} ± {:.1} bb/100",
            o.hands,
            o.bb100(),
            o.ci95(),
            o.mbb_per_hand(),
            o.adj_bb100(),
            o.adj_ci95()
        );
        println!(
            "@wandb {{\"lbr_hand\":{},\"lbr_bb100\":{:.2},\"lbr_bb100_ci\":{:.2},\"lbr_mbb\":{:.1},\"lbr_adj_bb100\":{:.2},\"lbr_adj_bb100_ci\":{:.2}}}",
            o.hands,
            o.bb100(),
            o.ci95(),
            o.mbb_per_hand(),
            o.adj_bb100(),
            o.adj_ci95()
        );
    });
    if out.mbb_per_hand() <= 0.0 {
        println!(
            "\nLBR LOST: {:+.1} ± {:.1} bb/100 over {} hands ({} errors, {:.0}s).",
            out.bb100(),
            out.ci95(),
            out.hands,
            out.errors,
            t0.elapsed().as_secs_f64()
        );
        println!(
            "  Exploitability lower bound certified: 0.0 mbb/hand — a responder that loses \n               proves nothing about the agent.  The win rate above is still a valid head-to-head \n               against a fixed benchmark opponent, which is what makes the ablation arms comparable."
        );
        println!("LBR luck-adjusted: {:+.1} ± {:.1} bb/100", out.adj_bb100(), out.adj_ci95());
        println!("Blueprint lookups: {}", bot.lookup_counts());
        println!("Abstraction tracking: {}", bot.tracking_counts());
        return;
    }
    println!(
        "\nLBR exploitability lower bound: {:.1} mbb/hand  ({:+.1} ± {:.1} bb/100 over {} hands, {} errors, {:.0}s)",
        out.bound_mbb(),
        out.bb100(),
        out.ci95(),
        out.hands,
        out.errors,
        t0.elapsed().as_secs_f64()
    );
    println!("LBR luck-adjusted: {:+.1} ± {:.1} bb/100", out.adj_bb100(), out.adj_ci95());
    println!("Blueprint lookups: {}", bot.lookup_counts());
    println!("Abstraction tracking: {}", bot.tracking_counts());
}
