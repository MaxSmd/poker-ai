//! Play the trained bot against **Slumbot** (slumbot.com, heads-up NLHE,
//! 200 bb, blinds 50/100).
//!
//!   play slumbot [hands] [flags]
//!
//!     hands              number of hands to play (default 1000)
//!     --data=DIR         artifact directory (default `data`): needs
//!                        blueprint_holdem.bin + {flop,turn,river}_buckets.bin
//!                        from the SAME training run
//!     --policy=PATH      blueprint strategy file, overriding
//!                        DATA/blueprint_holdem.bin (the path training writes
//!                        to, so a preserved run needs this)
//!     --stack-bb=N       blueprint stack depth in bb (default 200 — Slumbot's)
//!     --cap=N            blueprint raise cap (default 3 — must match training)
//!     --no-resolve       blueprint-only river (skip the vectorized re-solve)
//!     --resolve-turn     also re-solve the turn (runout leaves — slower)
//!     --resolve-flop     also re-solve the flop (two-card runout — much slower;
//!                        for small-sample testing)
//!     --iters=N          CFR⁺ iterations per river resolve (default 1500)
//!     --turn-iters=N     CFR⁺ iterations per turn/flop resolve (default 500)
//!     --river-cap=N      raise cap inside a resolve, every street (default 3)
//!     --continuations=L  comma-separated turn/flop leaf pot scales, first 0.0
//!                        (default 0.0,0.75,1.5,3.0; a single 0.0 = check-down)
//!     --purify=X         drop action probabilities below X (default 0.1)
//!     --seed=N           sampling seed (default 1)
//!     --log-hands=PATH   write full per-hand histories (final action string,
//!                        both hands, position, board, winnings) as JSONL,
//!                        truncating PATH — the post-mortem feed for
//!                        scripts/analyze_slumbot.py
//!     --token=T          reuse a session token (also persisted to
//!                        DATA/slumbot_token.txt automatically)
//!     --username=U --password=P   log in a registered account instead
//!
//! Prints a running bb/100 with a 95% confidence interval and `@wandb` metric
//! lines (compatible with scripts/train_wandb.py), and appends one line per
//! hand to DATA/slumbot_results.csv.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::games::blueprint::BlueprintHoldem;
use poker_ai::games::Game;
use poker_ai::evaluation::vector_br::{all_flops, best_response_value, sample_flops};
use poker_ai::play::cards::parse_cards;
use poker_ai::play::luck::luck_adjustment;
use poker_ai::play::protocol::{parse_action, BIG_BLIND};
use poker_ai::play::slumbot::SlumbotClient;
use poker_ai::play::{Bot, BotConfig, CompactPolicy};
use poker_core::action::Action;
use poker_core::make_card;
use poker_core::state::NO_CARD;

const ABSTRACT_BB: u32 = 2;
const ABSTRACT_SB: u32 = 1;

fn flag<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().find_map(|a| a.strip_prefix(&format!("--{name}="))).and_then(|v| v.parse().ok())
}

/// Reject unknown flags and stray positionals.  Every flag here takes its value
/// with `=`, so a bare `--log-hands out.jsonl` would otherwise drop the path on
/// the floor and log somewhere else entirely.
fn validate(args: &[String], allowed: &[&str], positionals: usize) {
    let mut seen_positional = 0;
    for a in &args[2..] {
        let Some(body) = a.strip_prefix("--") else {
            seen_positional += 1;
            if seen_positional > positionals {
                eprintln!("unexpected argument `{a}` (flags take their value as --name=value)");
                std::process::exit(2);
            }
            continue;
        };
        let name = body.split('=').next().unwrap_or("");
        if !allowed.contains(&name) {
            eprintln!("unknown flag `--{name}`; expected one of: {}", allowed.join(", "));
            std::process::exit(2);
        }
        if name == "log-hands" && !body.contains('=') {
            eprintln!("--log-hands needs a path: --log-hands=data/hands.jsonl");
            std::process::exit(2);
        }
    }
}

/// Path to the blueprint strategy: `--policy=PATH`, else `DIR/blueprint_holdem.bin`.
fn policy_path(args: &[String], dir: &Path) -> PathBuf {
    flag::<String>(args, "policy")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("blueprint_holdem.bin"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("slumbot") => run_slumbot(&args),
        Some("chart") => run_chart(&args),
        Some("expl") => run_expl(&args),
        _ => {
            eprintln!(
                "usage: play slumbot [hands] [flags]  |  play chart [flags]  |  play expl [flags]\n\
                 see the header of src/bin/play.rs"
            );
            std::process::exit(2);
        }
    }
}

/// Abstract-game exploitability of the blueprint via the vectorized best
/// response (`evaluation::vector_br`): exact betting/turn/river/ranges,
/// Monte-Carlo over `--flops` sampled flops (or every flop with
/// `--flops=all`).  Runtime scales linearly in flops and parallelizes over
/// them; the river-heavy cap-3 tree costs on the order of CPU-hours per
/// hundred flops — a milestone metric, not a per-checkpoint one.
///
///   play expl [--flops=N|all --seed=S --data=DIR --stack-bb=N --cap=N --policy=PATH]
fn run_expl(args: &[String]) {
    validate(args, &["data", "stack-bb", "cap", "policy", "flops", "seed", "board-samples"], 0);
    let dir = PathBuf::from(flag::<String>(args, "data").unwrap_or_else(|| "data".into()));
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap: u32 = flag(args, "cap").unwrap_or(3);
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
    let policy_path = policy_path(args, &dir);
    println!("Loading blueprint strategy {} ...", policy_path.display());
    let policy = CompactPolicy::load(&policy_path).unwrap_or_else(|e| {
        eprintln!("cannot load {}: {e}", policy_path.display());
        std::process::exit(1);
    });
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

/// Rank letters for the preflop chart, display index 0=A … 12=2.
const RANK_LETTERS: [char; 13] = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];

/// Engine rank for a display index (0=A → 12, 12=2 → 0).
fn engine_rank(display: usize) -> u8 {
    (12 - display) as u8
}

/// Concrete hole cards for chart cell `(row, col)` in the standard layout:
/// upper triangle (row<col) = suited, lower (row>col) = offsuit, diagonal =
/// pair.  Rows/cols are display indices (0=A … 12=2).
fn cell_hole(row: usize, col: usize) -> [u8; 2] {
    if row == col {
        let r = engine_rank(row);
        [make_card(r, 0), make_card(r, 1)]
    } else if row < col {
        // suited: high rank = row, low = col, same suit
        [make_card(engine_rank(row), 0), make_card(engine_rank(col), 0)]
    } else {
        // offsuit: high rank = col, low = row, different suits
        [make_card(engine_rank(col), 0), make_card(engine_rank(row), 1)]
    }
}

/// Two filler cards not colliding with `hero` — the opponent's placeholder
/// (never affects the acting player's own info key).
fn filler(hero: [u8; 2]) -> [u8; 2] {
    let mut out = [NO_CARD; 2];
    let mut n = 0;
    for c in 0u8..52 {
        if c != hero[0] && c != hero[1] {
            out[n] = c;
            n += 1;
            if n == 2 {
                break;
            }
        }
    }
    out
}

/// Aggregate an action distribution into (fold, passive=check/call,
/// aggressive=raise/all-in) probabilities.
fn classify(menu: &[Action], probs: &[f64]) -> (f64, f64, f64) {
    let (mut fold, mut passive, mut aggro) = (0.0, 0.0, 0.0);
    for (a, &p) in menu.iter().zip(probs) {
        match a {
            Action::Fold => fold += p,
            Action::Check | Action::Call => passive += p,
            Action::Raise(_) | Action::AllIn => aggro += p,
        }
    }
    (fold, passive, aggro)
}

/// Number of concrete combos a chart cell stands for: pair 6, suited 4,
/// offsuit 12 (sums to 1326 over the 169 cells).
fn combo_weight(row: usize, col: usize) -> f64 {
    if row == col {
        6.0
    } else if row < col {
        4.0
    } else {
        12.0
    }
}

/// Short label for a preflop root action; raise levels shown in big blinds.
/// `Call` reads "limp" because this is only used at the SB's opening node.
fn action_label(a: &Action) -> String {
    match a {
        Action::Fold => "fold".into(),
        Action::Check => "check".into(),
        Action::Call => "limp".into(),
        Action::Raise(n) => format!("{}bb", *n as f64 / ABSTRACT_BB as f64),
        Action::AllIn => "all-in".into(),
    }
}

/// Print a 13×13 percentage grid (values already in 0..=100), `?` where the
/// blueprint never stored the info set (uniform fallback).
fn print_grid(title: &str, vals: &[[f64; 13]; 13], missing: &[[bool; 13]; 13]) {
    println!("\n{title}  (rows/cols A→2; upper=suited, lower=offsuit, diag=pairs)");
    print!("     ");
    for c in RANK_LETTERS {
        print!("{c:>4}");
    }
    println!();
    for r in 0..13 {
        print!("  {} ", RANK_LETTERS[r]);
        for c in 0..13 {
            if missing[r][c] {
                print!("   ?");
            } else {
                print!("{:>4.0}", vals[r][c]);
            }
        }
        println!();
    }
}

/// Dump the blueprint's preflop strategy — the SB open chart and the BB
/// response to the smallest SB open — so a preflop leak (over-folding the BB,
/// too-tight opens) is visible at a glance.
///
///   play chart [--data=DIR --stack-bb=N --cap=N]
fn run_chart(args: &[String]) {
    validate(args, &["data", "stack-bb", "cap", "policy"], 0);
    let dir = PathBuf::from(flag::<String>(args, "data").unwrap_or_else(|| "data".into()));
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap: u32 = flag(args, "cap").unwrap_or(3);

    println!("Loading abstraction from {} ({stack_bb}bb, cap-{cap})", dir.display());
    let game = load_game(&dir, stack_bb, cap);
    let policy_path = policy_path(args, &dir);
    println!("Loading blueprint strategy {} ...", policy_path.display());
    let policy = CompactPolicy::load(&policy_path).unwrap_or_else(|e| {
        eprintln!("cannot load {}: {e}", policy_path.display());
        std::process::exit(1);
    });
    println!("  {} info sets", policy.len());

    let mut sb_open = [[0.0f64; 13]; 13]; // P(raise/all-in) opening the button
    let mut sb_fold = [[0.0f64; 13]; 13]; // P(fold) — the SB min-fold
    let mut sb_small = [[0.0f64; 13]; 13]; // P(smallest open) — the node the BB charts condition on
    let mut bb_fold = [[0.0f64; 13]; 13]; // P(fold) facing the smallest SB open
    let mut bb_3bet = [[0.0f64; 13]; 13]; // P(raise/all-in) — the BB 3-bet
    let mut sb_missing = [[false; 13]; 13];
    let mut bb_missing = [[false; 13]; 13];
    // Combo-weighted mean probability per SB root action (pair=6, suited=4,
    // offsuit=12 combos): how the blueprint's SB splits its play across
    // limp / each open size — invisible in the aggregated `sb_open` grid, and
    // exactly what determines the range the BB response was trained against.
    let mut sb_mix: Vec<f64> = Vec::new();
    let mut sb_mix_labels: Vec<String> = Vec::new();
    let mut sb_mix_weight = 0.0f64;

    for row in 0..13 {
        for col in 0..13 {
            let hero = cell_hole(row, col);
            let opp = filler(hero);

            // --- SB open: seat 0 (button) to act, empty history. ---
            let sb_state = game.play_state([hero, opp], [NO_CARD; 5]);
            let menu = game.actions(&sb_state);
            let key = game.info_key(&sb_state);
            sb_missing[row][col] = policy.get(key).is_none();
            let probs = policy.probs_or_uniform(key, menu.len());
            let (fold, _passive, aggro) = classify(&menu, &probs);
            sb_open[row][col] = 100.0 * aggro;
            sb_fold[row][col] = 100.0 * fold;
            if let Some(i) = menu.iter().position(|a| matches!(a, Action::Raise(_))) {
                sb_small[row][col] = 100.0 * probs[i];
            }
            if sb_mix_labels.is_empty() {
                sb_mix_labels = menu.iter().map(action_label).collect();
                sb_mix = vec![0.0; menu.len()];
            }
            let w = combo_weight(row, col);
            for (m, &p) in sb_mix.iter_mut().zip(probs.iter()) {
                *m += w * p;
            }
            sb_mix_weight += w;

            // --- BB defense vs the smallest SB open. ---
            // Put the hero hand in the BB (seat 1); a filler SB (seat 0) makes
            // the smallest raise, then the BB is to act.  The BB's key depends
            // only on its own hand + the action history, so the SB filler cards
            // are irrelevant.
            let sb_filler = filler(hero);
            let opener = game.play_state([sb_filler, hero], [NO_CARD; 5]);
            let open_menu = game.actions(&opener);
            if let Some(raise_idx) = open_menu.iter().position(|a| matches!(a, Action::Raise(_))) {
                let bb_state = game.apply(&opener, raise_idx);
                let bmenu = game.actions(&bb_state);
                let bkey = game.info_key(&bb_state);
                bb_missing[row][col] = policy.get(bkey).is_none();
                let bprobs = policy.probs_or_uniform(bkey, bmenu.len());
                let (bfold, _bpassive, baggro) = classify(&bmenu, &bprobs);
                bb_fold[row][col] = 100.0 * bfold;
                bb_3bet[row][col] = 100.0 * baggro;
            } else {
                bb_missing[row][col] = true;
            }
        }
    }

    print_grid("SB open — P(raise/all-in) %", &sb_open, &sb_missing);
    print_grid("SB fold % (limp-or-fold: high = over-folding the button)", &sb_fold, &sb_missing);
    let small_label = sb_mix_labels
        .iter()
        .find(|l| l.ends_with("bb"))
        .cloned()
        .unwrap_or_else(|| "smallest open".into());
    print_grid(
        &format!(
            "SB smallest open ({small_label}) — P %  (the range the BB response below was trained against)"
        ),
        &sb_small,
        &sb_missing,
    );
    print!("\nSB action mix (combo-weighted over all 1326 hands): ");
    for (label, m) in sb_mix_labels.iter().zip(&sb_mix) {
        print!(" {label} {:.1}%", 100.0 * m / sb_mix_weight);
    }
    println!();
    print_grid("BB vs smallest SB open — P(fold) %", &bb_fold, &bb_missing);
    print_grid("BB vs smallest SB open — P(3-bet) %", &bb_3bet, &bb_missing);
    println!(
        "\nSanity: strong hands should show high SB-open / high BB-3bet and low fold; \
         trash the reverse. `?` = info set the blueprint never visited."
    );
}

/// Load the abstract game exactly as it was trained (same stack, cap, and
/// bucket maps) — key compatibility with `data/blueprint_holdem.bin` depends
/// on all three.
fn load_game(dir: &Path, stack_bb: u32, cap: u32) -> BlueprintHoldem {
    let mut game =
        BlueprintHoldem::new(stack_bb * ABSTRACT_BB, ABSTRACT_BB, ABSTRACT_SB, 0).with_raise_cap(cap);
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

fn run_slumbot(args: &[String]) {
    validate(
        args,
        &[
            "data", "policy", "stack-bb", "cap", "no-resolve", "resolve-turn", "resolve-flop",
            "turn-checkdown", "no-continual", "iters", "turn-iters", "river-cap",
            "continuations", "purify", "seed", "log-hands", "token", "username", "password",
        ],
        1,
    );
    let hands: u64 = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let dir = PathBuf::from(flag::<String>(args, "data").unwrap_or_else(|| "data".into()));
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap: u32 = flag(args, "cap").unwrap_or(3);
    // Turn/flop continuation scales (finding #1): comma-separated, first should
    // be 0.0.  A single value (e.g. `--continuations=0.0`) is a plain check-down.
    let continuations: Vec<f64> = flag::<String>(args, "continuations")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<f64>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0.0, 0.75, 1.5, 3.0]);
    let cfg = BotConfig {
        resolve_river: !args.iter().any(|a| a == "--no-resolve"),
        resolve_turn: args.iter().any(|a| a == "--resolve-turn"),
        resolve_flop: args.iter().any(|a| a == "--resolve-flop"),
        river_iters: flag(args, "iters").unwrap_or(1_500),
        turn_iters: flag(args, "turn-iters").unwrap_or(500),
        river_cap: flag(args, "river-cap").unwrap_or(3),
        continuations,
        // Turn resolves solve the real river betting by default; the
        // K-continuation check-down cut stays available for A/Bs and speed.
        turn_full_river: !args.iter().any(|a| a == "--turn-checkdown"),
        // Continual re-solving (CFV-gadget carry) on by default; --no-continual
        // gives independent per-decision resolves for A/Bs.
        continual: !args.iter().any(|a| a == "--no-continual"),
        purify: flag(args, "purify").unwrap_or(0.1),
        seed: flag(args, "seed").unwrap_or(1),
    };

    println!("Loading abstraction from {} ({stack_bb}bb, cap-{cap})", dir.display());
    let game = load_game(&dir, stack_bb, cap);
    let policy_path = policy_path(args, &dir);
    println!("Loading blueprint strategy {} ...", policy_path.display());
    let policy = CompactPolicy::load(&policy_path).unwrap_or_else(|e| {
        eprintln!("cannot load {}: {e}", policy_path.display());
        std::process::exit(1);
    });
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
    let mut bot = Bot::new(game, policy, cfg);

    // Session token: flag > persisted file > fresh (server mints one).
    let token_path = dir.join("slumbot_token.txt");
    let client = SlumbotClient::new();
    let mut token: Option<String> = flag::<String>(args, "token")
        .or_else(|| std::fs::read_to_string(&token_path).ok().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty());
    if let (Some(u), Some(p)) = (flag::<String>(args, "username"), flag::<String>(args, "password")) {
        match client.login(&u, &p) {
            Ok(t) => token = Some(t),
            Err(e) => {
                eprintln!("login failed: {e}");
                std::process::exit(1);
            }
        }
    }

    let csv_path = dir.join("slumbot_results.csv");
    let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(&csv_path).ok();
    // Truncate: appending across runs silently concatenates matches, which has
    // twice produced a log the analyzer could not attribute to a single run.
    let mut hands_log = flag::<String>(args, "log-hands").map(|p| {
        std::fs::File::create(&p).unwrap_or_else(|e| {
            eprintln!("cannot open {p}: {e}");
            std::process::exit(1);
        })
    });

    let mut played: u64 = 0;
    let mut errors: u64 = 0;
    let mut net_bb: f64 = 0.0;
    let mut sumsq_bb: f64 = 0.0;
    let mut adj_net_bb: f64 = 0.0;
    let mut adj_sumsq_bb: f64 = 0.0;
    println!("Playing {hands} hands against Slumbot ...");
    while played < hands {
        match play_one_hand(&client, &mut bot, &mut token) {
            Ok(mut rec) => {
                rec.luck = hand_luck(&rec);
                let winnings = rec.winnings;
                played += 1;
                let bb = winnings as f64 / BIG_BLIND as f64;
                net_bb += bb;
                sumsq_bb += bb * bb;
                let adj = (winnings as f64 - rec.luck) / BIG_BLIND as f64;
                adj_net_bb += adj;
                adj_sumsq_bb += adj * adj;
                if let Some(f) = csv.as_mut() {
                    let _ = writeln!(f, "{played},{winnings}");
                }
                if let Some(f) = hands_log.as_mut() {
                    let _ = writeln!(f, "{}", rec.to_json(played));
                }
                if let Some(t) = &token {
                    let _ = std::fs::write(&token_path, t);
                }
                if played.is_multiple_of(100) || played == hands {
                    let n = played as f64;
                    let mean = net_bb / n;
                    let var = (sumsq_bb / n - mean * mean).max(0.0);
                    let ci = 1.96 * (var / n).sqrt() * 100.0;
                    let adj_mean = adj_net_bb / n;
                    let adj_var = (adj_sumsq_bb / n - adj_mean * adj_mean).max(0.0);
                    let adj_ci = 1.96 * (adj_var / n).sqrt() * 100.0;
                    println!(
                        "  {played:>6} hands   net {net_bb:>9.1} bb   {:>8.1} ± {ci:.1} bb/100   luck-adj {:>8.1} ± {adj_ci:.1} bb/100",
                        mean * 100.0,
                        adj_mean * 100.0
                    );
                    println!(
                        "@wandb {{\"hand\":{played},\"net_bb\":{net_bb:.2},\"bb100\":{:.2},\"bb100_ci\":{ci:.2},\"adj_bb100\":{:.2},\"adj_bb100_ci\":{adj_ci:.2}}}",
                        mean * 100.0,
                        adj_mean * 100.0
                    );
                }
            }
            Err(e) => {
                errors += 1;
                eprintln!("hand error ({errors} so far): {e}");
                if e.contains("token") || errors.is_multiple_of(5) {
                    token = None; // let the server mint a fresh session
                }
                if errors > 50 && errors > played {
                    eprintln!("too many errors; giving up");
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs(3));
            }
        }
    }

    let mean = if played > 0 { net_bb / played as f64 } else { 0.0 };
    let var = if played > 0 { (sumsq_bb / played as f64 - mean * mean).max(0.0) } else { 0.0 };
    let ci = if played > 0 { 1.96 * (var / played as f64).sqrt() * 100.0 } else { 0.0 };
    let adj_mean = if played > 0 { adj_net_bb / played as f64 } else { 0.0 };
    let adj_var =
        if played > 0 { (adj_sumsq_bb / played as f64 - adj_mean * adj_mean).max(0.0) } else { 0.0 };
    let adj_ci = if played > 0 { 1.96 * (adj_var / played as f64).sqrt() * 100.0 } else { 0.0 };
    println!(
        "\nDone: {played} hands, net {net_bb:.1} bb, {:.1} ± {ci:.1} bb/100 (95% CI), luck-adjusted {:.1} ± {adj_ci:.1} bb/100, {errors} errors",
        mean * 100.0,
        adj_mean * 100.0
    );
    println!("Per-hand log: {}", csv_path.display());
}

/// The AIVAT-style luck correction for a finished hand, in chips (see
/// `play::luck`).  Zero (always unbiased) whenever the opponent's cards are
/// missing or anything fails to parse — a skipped correction never biases the
/// adjusted estimate, it only forgoes variance reduction on that hand.
fn hand_luck(rec: &HandRecord) -> f64 {
    let (Ok(ours), Ok(theirs), Ok(board)) = (
        parse_cards(&rec.hole_cards),
        parse_cards(&rec.bot_hole_cards),
        parse_cards(&rec.board),
    ) else {
        return 0.0;
    };
    if ours.len() != 2 || theirs.len() != 2 {
        return 0.0;
    }
    luck_adjustment([ours[0], ours[1]], [theirs[0], theirs[1]], &board, &rec.action).unwrap_or(0.0)
}

/// The outcome of one played hand — winnings plus the fields a post-mortem
/// needs (all from the final server response).
struct HandRecord {
    winnings: i64,
    client_pos: u8,
    hole_cards: Vec<String>,
    bot_hole_cards: Vec<String>,
    board: Vec<String>,
    action: String,
    /// AIVAT-style chance correction in chips (`play::luck`); 0 when not
    /// computable.  Adjusted winnings = `winnings - luck`.
    luck: f64,
}

impl HandRecord {
    /// One JSONL line, built with `serde_json` so every field is escaped
    /// correctly (hand-rolled JSON is how subtle quoting bugs sneak in).
    /// `client_pos` follows Slumbot (0 = BB, 1 = SB); `action` is the full
    /// final action string; `reached_street` is derived so the analyzer needn't
    /// re-parse (0=preflop … 3=river; how far the hand got).
    fn to_json(&self, index: u64) -> String {
        let reached = parse_action(&self.action).map(|p| p.street).unwrap_or(0);
        let obj = serde_json::json!({
            "i": index,
            "pos": self.client_pos,
            "winnings": self.winnings,
            "luck": self.luck,
            "reached_street": reached,
            "hole": self.hole_cards,
            "bot_hole": self.bot_hole_cards,
            "board": self.board,
            "action": self.action,
        });
        obj.to_string()
    }
}

/// Play a single hand start to finish; returns the full record for logging.
fn play_one_hand(
    client: &SlumbotClient,
    bot: &mut Bot,
    token: &mut Option<String>,
) -> Result<HandRecord, String> {
    let mut r = client.new_hand(token.as_deref())?;
    if let Some(t) = r.token.take() {
        *token = Some(t);
    }

    let client_pos = r.client_pos.ok_or("new_hand response missing client_pos")?;
    let hole_strs = r.hole_cards.clone().ok_or("new_hand response missing hole_cards")?;
    let hole = parse_cards(&hole_strs)?;
    if hole.len() != 2 {
        return Err(format!("expected 2 hole cards, got {hole_strs:?}"));
    }
    let mut hs = bot.start_hand(client_pos, [hole[0], hole[1]]);

    loop {
        if let Some(w) = r.winnings {
            return Ok(HandRecord {
                winnings: w,
                client_pos,
                hole_cards: hole_strs,
                bot_hole_cards: r.bot_hole_cards.clone().unwrap_or_default(),
                board: r.board.clone().unwrap_or_default(),
                action: r.action.clone().unwrap_or_default(),
                luck: 0.0,
            });
        }
        let action = r.action.clone().ok_or("response missing action")?;
        let board = parse_cards(r.board.as_deref().unwrap_or(&[]))?;
        let parsed = parse_action(&action)?;
        if parsed.next_pos != client_pos as i8 {
            return Err(format!(
                "server awaits nobody/us mismatch (next_pos {}, we are {client_pos}): {action:?}",
                parsed.next_pos
            ));
        }
        let incr = bot.act(&mut hs, &action, &board)?;
        let t = token.clone().ok_or("no session token")?;
        r = client.act(&t, &incr)?;
        if let Some(t) = r.token.take() {
            *token = Some(t);
        }
    }
}
