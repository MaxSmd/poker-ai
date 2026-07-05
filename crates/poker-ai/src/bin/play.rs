//! Play the trained bot against **Slumbot** (slumbot.com, heads-up NLHE,
//! 200 bb, blinds 50/100).
//!
//!   play slumbot [hands] [flags]
//!
//!     hands              number of hands to play (default 1000)
//!     --data=DIR         artifact directory (default `data`): needs
//!                        blueprint_holdem.bin + {flop,turn,river}_buckets.bin
//!                        from the SAME training run
//!     --stack-bb=N       blueprint stack depth in bb (default 200 — Slumbot's)
//!     --cap=N            blueprint raise cap (default 3 — must match training)
//!     --no-resolve       blueprint-only river (skip the vectorized re-solve)
//!     --iters=N          CFR⁺ iterations per river resolve (default 1500)
//!     --river-cap=N      raise cap inside a river resolve (default 3)
//!     --purify=X         drop action probabilities below X (default 0.1)
//!     --seed=N           sampling seed (default 1)
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
use poker_ai::play::cards::parse_cards;
use poker_ai::play::protocol::{parse_action, BIG_BLIND};
use poker_ai::play::slumbot::SlumbotClient;
use poker_ai::play::{Bot, BotConfig, CompactPolicy};

const ABSTRACT_BB: u32 = 2;
const ABSTRACT_SB: u32 = 1;

fn flag<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().find_map(|a| a.strip_prefix(&format!("--{name}="))).and_then(|v| v.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("slumbot") => run_slumbot(&args),
        _ => {
            eprintln!("usage: play slumbot [hands] [flags] — see the header of src/bin/play.rs");
            std::process::exit(2);
        }
    }
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
    let hands: u64 = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let dir = PathBuf::from(flag::<String>(args, "data").unwrap_or_else(|| "data".into()));
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap: u32 = flag(args, "cap").unwrap_or(3);
    let cfg = BotConfig {
        resolve_river: !args.iter().any(|a| a == "--no-resolve"),
        river_iters: flag(args, "iters").unwrap_or(1_500),
        river_cap: flag(args, "river-cap").unwrap_or(3),
        purify: flag(args, "purify").unwrap_or(0.1),
        seed: flag(args, "seed").unwrap_or(1),
    };

    println!("Loading abstraction from {} ({stack_bb}bb, cap-{cap})", dir.display());
    let game = load_game(&dir, stack_bb, cap);
    let policy_path = dir.join("blueprint_holdem.bin");
    println!("Loading blueprint strategy {} ...", policy_path.display());
    let policy = CompactPolicy::load(&policy_path).unwrap_or_else(|e| {
        eprintln!("cannot load {}: {e}", policy_path.display());
        std::process::exit(1);
    });
    println!(
        "  {} info sets; river resolve: {}",
        policy.len(),
        if cfg.resolve_river {
            format!("on ({} iters, cap {})", cfg.river_iters, cfg.river_cap)
        } else {
            "off".into()
        }
    );
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

    let mut played: u64 = 0;
    let mut errors: u64 = 0;
    let mut net_bb: f64 = 0.0;
    let mut sumsq_bb: f64 = 0.0;
    println!("Playing {hands} hands against Slumbot ...");
    while played < hands {
        match play_one_hand(&client, &mut bot, &mut token) {
            Ok(winnings) => {
                played += 1;
                let bb = winnings as f64 / BIG_BLIND as f64;
                net_bb += bb;
                sumsq_bb += bb * bb;
                if let Some(f) = csv.as_mut() {
                    let _ = writeln!(f, "{played},{winnings}");
                }
                if let Some(t) = &token {
                    let _ = std::fs::write(&token_path, t);
                }
                if played.is_multiple_of(100) || played == hands {
                    let mean = net_bb / played as f64;
                    let var = (sumsq_bb / played as f64 - mean * mean).max(0.0);
                    let ci = 1.96 * (var / played as f64).sqrt() * 100.0;
                    println!(
                        "  {played:>6} hands   net {net_bb:>9.1} bb   {:>8.1} ± {ci:.1} bb/100",
                        mean * 100.0
                    );
                    println!(
                        "@wandb {{\"hand\":{played},\"net_bb\":{net_bb:.2},\"bb100\":{:.2},\"bb100_ci\":{ci:.2}}}",
                        mean * 100.0
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
    println!(
        "\nDone: {played} hands, net {net_bb:.1} bb, {:.1} ± {ci:.1} bb/100 (95% CI), {errors} errors",
        mean * 100.0
    );
    println!("Per-hand log: {}", csv_path.display());
}

/// Play a single hand start to finish; returns our winnings in chips.
fn play_one_hand(client: &SlumbotClient, bot: &mut Bot, token: &mut Option<String>) -> Result<i64, String> {
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
            return Ok(w);
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
