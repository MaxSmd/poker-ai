//! Play the trained bot against **Slumbot** (slumbot.com, heads-up NLHE,
//! 200 bb, blinds 50/100), or measure it locally.
//!
//! Subcommands: `slumbot` (play a match), `chart` (print a strategy chart),
//! `expl` (abstract-game exploitability of the **blueprint**), `lbr`
//! (exploitability lower bound of the **live agent** by local best response —
//! the only measurement here that exercises re-solving), `replay` (what the
//! agent would do in logged spots — see `replay.rs`).
//!
//! ```text
//! play slumbot [hands] [flags]
//! play lbr     [hands] [--lbr-seed=S] [bot flags]
//! play replay  --spots=SPOTS.jsonl --out=TRACE.jsonl [bot flags]
//!
//!   hands              number of hands to play (default 1000)
//!   --data=DIR         artifact directory (default `data`): needs
//!                      blueprint_holdem.bin + {flop,turn,river}_buckets.bin
//!                      from the SAME training run
//!   --policy=PATH      blueprint strategy file, overriding
//!                      DATA/blueprint_holdem.bin (the path training writes
//!                      to, so a preserved run needs this)
//!   --stack-bb=N       blueprint stack depth in bb (default 200 — Slumbot's)
//!   --cap=N            blueprint raise cap (default 3 — must match training);
//!                      --cap=P,F,T,R for a per-street-capped blueprint
//!   --no-resolve       blueprint-only (skip the vectorized re-solve entirely,
//!                      including for spots the abstraction lost track of)
//!   --no-resolve-turn  blueprint on the turn (re-solving is ON by default)
//!   --no-resolve-flop  blueprint on the flop (re-solving is ON by default)
//!   --runout-sample=N  board completions evaluated per iteration at a
//!                      depth-cut leaf, 0 = exact (default 64).  What makes
//!                      turn/flop re-solving affordable: a flop leaf has 1176
//!                      completions, and exact cost 24.6 s/decision
//!   --turn-full-river  turn resolves solve the real river betting instead of
//!                      cutting at the reveal.  Exact, and 798 s/decision —
//!                      an offline reference, not a playable mode
//!   --no-continual     solve every decision unconstrained: no CFV gadget and
//!                      no values carried between a hand's resolves
//!   --iters=N          CFR⁺ iterations per river resolve (default 1500)
//!   --turn-iters=N     CFR⁺ iterations per turn/flop resolve (default 500)
//!   --river-cap=N      raise cap inside a resolve, every street (default 3)
//!   --continuations=L  comma-separated turn/flop leaf pot scales, first 0.0
//!                      (default 0.0,0.75,1.5,3.0; a single 0.0 = check-down)
//!   --purify=X         drop action probabilities below X (default 0.1)
//!   --opp-range-floor=X  floor on the likelihood of the opponent's postflop
//!                      actions in its range update (default 0 = plain
//!                      Bayes; 1 = postflop actions do not narrow its range)
//!   --seed=N           sampling seed (default 1)
//!   --log-hands=PATH   write full per-hand histories (final action string,
//!                      both hands, position, board, winnings) as JSONL,
//!                      truncating PATH — the post-mortem feed for
//!                      scripts/analyze_slumbot.py
//!   --token=T          reuse a session token (also persisted to
//!                      DATA/slumbot_token.txt automatically)
//!   --username=U --password=P   log in a registered account instead
//! ```
//!
//! Prints a running bb/100 with a 95% confidence interval and `@wandb` metric
//! lines (compatible with scripts/train_wandb.py), and appends one line per
//! hand to DATA/slumbot_results.csv.

mod chart;
mod expl;
mod lbr;
mod replay;
mod slumbot;

use std::path::{Path, PathBuf};

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::games::blueprint::{BlueprintHoldem, RaiseCaps};
use poker_ai::play::{Bot, BotConfig, CompactPolicy};
use poker_ai::util::cli::{flag, strict_flag, validate_flags};

const ABSTRACT_BB: u32 = 2;
const ABSTRACT_SB: u32 = 1;

/// The flags that configure the playing agent ([`build_bot`]) — shared by
/// `slumbot` and `lbr`, which each add their own.
const BOT_FLAGS: &[&str] = &[
    "data", "policy", "stack-bb", "cap", "no-resolve", "no-resolve-turn", "no-resolve-flop",
    "turn-full-river", "runout-sample", "no-continual", "iters", "turn-iters", "river-cap",
    "continuations", "purify", "opp-range-floor", "seed",
];

/// `--cap=N` or `--cap=P,F,T,R` (default 3).  Unlike [`flag`], a value that
/// does not parse exits: a silently defaulted cap loads a differently shaped
/// game, and the only symptom is a collapsed lookup hit rate.
fn cap_flag(args: &[String]) -> RaiseCaps {
    strict_flag(args, "cap").unwrap_or(RaiseCaps::uniform(3))
}

/// Reject unknown flags and stray positionals (shared with the other binaries),
/// plus the one value-bearing flag that would silently misbehave: a bare
/// `--log-hands out.jsonl` drops the path on the floor and logs elsewhere.
fn validate(args: &[String], allowed: &[&str], positionals: usize) {
    validate_flags(args, 2, allowed, positionals);
    if args.iter().any(|a| a == "--log-hands") {
        eprintln!("--log-hands needs a path: --log-hands=data/hands.jsonl");
        std::process::exit(2);
    }
}

/// The optional `[hands]` positional after the subcommand (default 1000).
fn hands_arg(args: &[String]) -> u64 {
    args.get(2).filter(|a| !a.starts_with("--")).and_then(|s| s.parse().ok()).unwrap_or(1000)
}

/// Path to the blueprint strategy: `--policy=PATH`, else `DIR/blueprint_holdem.bin`.
fn policy_path(args: &[String], dir: &Path) -> PathBuf {
    flag::<String>(args, "policy")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("blueprint_holdem.bin"))
}

/// Load the blueprint strategy named by [`policy_path`], exiting if it cannot
/// be read.
fn load_policy(args: &[String], dir: &Path) -> CompactPolicy {
    let path = policy_path(args, dir);
    println!("Loading blueprint strategy {} ...", path.display());
    CompactPolicy::load(&path).unwrap_or_else(|e| {
        eprintln!("cannot load {}: {e}", path.display());
        std::process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("slumbot") => slumbot::run(&args),
        Some("chart") => chart::run(&args),
        Some("expl") => expl::run(&args),
        Some("lbr") => lbr::run(&args),
        Some("replay") => replay::run(&args),
        _ => {
            eprintln!(
                "usage: play slumbot [hands] | chart | expl | lbr [hands] | replay --spots=PATH --out=PATH  [flags]\n\
                 see the header of src/bin/play/main.rs"
            );
            std::process::exit(2);
        }
    }
}

/// Load the abstract game exactly as it was trained (same stack, cap, and
/// bucket maps) — key compatibility with `data/blueprint_holdem.bin` depends
/// on all three.
fn load_game(dir: &Path, stack_bb: u32, cap: RaiseCaps) -> BlueprintHoldem {
    let mut game =
        BlueprintHoldem::new(stack_bb * ABSTRACT_BB, ABSTRACT_BB, ABSTRACT_SB, 0).with_raise_caps(cap);
    for (street, name) in ["flop_buckets.bin", "turn_buckets.bin", "river_buckets.bin"]
        .iter()
        .enumerate()
    {
        let path = dir.join(name);
        match BucketMap::load(&path) {
            Ok(map) => {
                println!("  {}: {} buckets loaded from {}", ["flop", "turn", "river"][street], map.num_buckets(), path.display());
                game = game.with_street_bucket(street, map);
            }
            Err(e) => {
                eprintln!(
                    "WARNING: no bucket map at {} ({e}) — keys will not match a bucketed blueprint",
                    path.display()
                );
            }
        }
    }
    game
}

/// Build the playing agent from the shared flags — the setup `slumbot` and
/// `lbr` both need, kept in one place so the two commands cannot drift into
/// configuring different bots and producing incomparable numbers.
fn build_bot(args: &[String], dir: &Path) -> Bot {
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap = cap_flag(args);
    let continuations: Vec<f64> = flag::<String>(args, "continuations")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<f64>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0.0, 0.75, 1.5, 3.0]);
    let no_resolve = args.iter().any(|a| a == "--no-resolve");
    let cfg = BotConfig {
        resolve_river: !no_resolve,
        resolve_turn: !no_resolve && !args.iter().any(|a| a == "--no-resolve-turn"),
        resolve_flop: !no_resolve && !args.iter().any(|a| a == "--no-resolve-flop"),
        river_iters: flag(args, "iters").unwrap_or(1_500),
        turn_iters: flag(args, "turn-iters").unwrap_or(500),
        river_cap: flag(args, "river-cap").unwrap_or(3),
        continuations,
        turn_full_river: args.iter().any(|a| a == "--turn-full-river"),
        runout_sample: flag(args, "runout-sample").unwrap_or(64),
        continual: !args.iter().any(|a| a == "--no-continual"),
        resolve_desynced: !no_resolve,
        purify: flag(args, "purify").unwrap_or(0.1),
        opp_range_floor: strict_flag(args, "opp-range-floor").unwrap_or(0.0),
        seed: flag(args, "seed").unwrap_or(1),
    };

    println!("Loading abstraction from {} ({stack_bb}bb, cap-{cap})", dir.display());
    let game = load_game(dir, stack_bb, cap);
    let policy = load_policy(args, dir);
    let resolve_state = |on: bool, iters: u64| {
        if on {
            format!("on ({iters} iters, cap {})", cfg.river_cap)
        } else {
            "off".into()
        }
    };
    println!(
        "  {} info sets; resolve — river: {}, turn: {}, flop: {}",
        policy.len(),
        resolve_state(cfg.resolve_river, cfg.river_iters),
        resolve_state(cfg.resolve_turn, cfg.turn_iters),
        resolve_state(cfg.resolve_flop, cfg.turn_iters),
    );
    if cfg.resolve_turn || cfg.resolve_flop {
        println!("  continuations (K={}): {:?}", cfg.continuations.len(), cfg.continuations);
    }
    Bot::new(game, policy, cfg)
}
