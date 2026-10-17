//! `play replay` — what the agent would do in logged spots.
//!
//! Win rates need tens of thousands of hands to tell two configurations
//! apart; the agent's action distribution in one specific spot needs only the
//! spots themselves.  This replays logged Slumbot hands up to a chosen
//! decision of ours and records the distribution the agent would draw from
//! there, under whatever bot flags it is given.
//!
//! Every earlier decision of ours is replayed with the move actually logged
//! ([`Bot::act_forced`]), so the agent's ranges, carried values and abstract
//! mirror follow the real hand; Slumbot's actions are fed as observed
//! events, exactly as in a live match.
//!
//! ```text
//! play replay --spots=SPOTS.jsonl --out=TRACE.jsonl [bot flags]
//! ```
//!
//! `SPOTS.jsonl` comes from `scripts/replay_spots.py select`: one logged hand
//! per line plus `target`, the offset in its action string of the decision to
//! measure.  Each output line repeats the spot's `i`, `hole`, `board` and the
//! logged move, and adds the agent's probability of folding, calling,
//! checking and raising there, and which path decided (resolve / blueprint /
//! desynced).  `scripts/replay_spots.py summarize` turns traces into the
//! comparison table.

use std::io::{BufRead, BufReader, Write};

use poker_ai::play::cards::parse_cards;
use poker_ai::play::protocol::parse_action;
use poker_ai::play::tracker::RealMove;
use poker_ai::play::{Bot, DecisionPath};
use poker_ai::util::cli::{data_dir, flag};

use super::{build_bot, validate, BOT_FLAGS};

/// One logged hand and the decision in it to measure.
#[derive(serde::Deserialize)]
struct Spot {
    i: u64,
    pos: u8,
    hole: Vec<String>,
    board: Vec<String>,
    action: String,
    target: usize,
}

pub(super) fn run(args: &[String]) {
    validate(args, &[BOT_FLAGS, &["spots", "out"]].concat(), 0);
    let (Some(spots_path), Some(out_path)) = (flag::<String>(args, "spots"), flag::<String>(args, "out")) else {
        eprintln!("usage: play replay --spots=SPOTS.jsonl --out=TRACE.jsonl [bot flags]");
        std::process::exit(2);
    };
    let spots_path = spots_path.as_str();
    let spots: Vec<Spot> = BufReader::new(std::fs::File::open(spots_path).unwrap_or_else(|e| {
        eprintln!("cannot open {spots_path}: {e}");
        std::process::exit(1);
    }))
    .lines()
    .map_while(Result::ok)
    .filter(|l| !l.trim().is_empty())
    .map(|l| serde_json::from_str(&l).unwrap_or_else(|e| panic!("bad spot line: {e}")))
    .collect();

    let mut bot = build_bot(args, &data_dir(args));
    let mut out = std::fs::File::create(&out_path).unwrap_or_else(|e| {
        eprintln!("cannot create {out_path}: {e}");
        std::process::exit(1);
    });
    println!("\nReplaying {} spots -> {out_path}", spots.len());
    let t0 = std::time::Instant::now();
    let mut errors = 0u64;
    for (n, spot) in spots.iter().enumerate() {
        match replay_one(&mut bot, spot) {
            Ok(line) => {
                writeln!(out, "{line}").expect("write trace");
            }
            Err(e) => {
                errors += 1;
                eprintln!("  spot {} (hand {}): {e}", n, spot.i);
            }
        }
        if (n + 1) % 50 == 0 || n + 1 == spots.len() {
            println!(
                "  {:>5}/{} spots   {:.0}s   {} errors",
                n + 1,
                spots.len(),
                t0.elapsed().as_secs_f64(),
                errors
            );
        }
    }
    println!(
        "Done: {} spots, {errors} errors, {} forced moves unmatched",
        spots.len(),
        bot.forced_mismatches()
    );
    println!("Abstraction tracking: {}", bot.tracking_counts());
}

/// Cards visible on `street` (0 preflop … 3 river).
fn visible(board: &[u8], street: u8) -> Result<Vec<u8>, String> {
    let n = [0, 3, 4, 5][street.min(3) as usize];
    board.get(..n).map(<[u8]>::to_vec).ok_or_else(|| format!("log board has {} cards, street {street} needs {n}", board.len()))
}

/// Start offsets of each move token (`k c f b<N>`) in a Slumbot action string.
fn token_starts(action: &str) -> Vec<usize> {
    let b = action.as_bytes();
    (0..b.len()).filter(|&i| matches!(b[i], b'k' | b'c' | b'f' | b'b')).collect()
}

/// The move token starting at offset `at`.
fn token_at(action: &str, at: usize) -> &str {
    let rest = &action[at..];
    let len = match rest.strip_prefix('b') {
        Some(digits) => 1 + digits.bytes().take_while(u8::is_ascii_digit).count(),
        None => 1,
    };
    &rest[..len]
}

fn replay_one(bot: &mut Bot, spot: &Spot) -> Result<String, String> {
    let hole = parse_cards(&spot.hole)?;
    if hole.len() != 2 {
        return Err(format!("expected 2 hole cards, got {:?}", spot.hole));
    }
    let board = parse_cards(&spot.board)?;
    let mut hs = bot.start_hand(spot.pos, [hole[0], hole[1]]);

    for at in token_starts(&spot.action) {
        if at > spot.target {
            break;
        }
        let prefix = &spot.action[..at];
        let parsed = parse_action(prefix)?;
        if parsed.next_pos != spot.pos as i8 {
            continue; // Slumbot's move: fed as an event by the next act
        }
        let logged = token_at(&spot.action, at);
        let seen = visible(&board, parsed.street)?;
        bot.act_forced(&mut hs, prefix, &seen, logged)?;
        if at == spot.target {
            let t = bot.last_trace().ok_or("decision left no trace")?;
            let path = match t.path {
                DecisionPath::Blueprint => "blueprint",
                DecisionPath::Resolve => "resolve",
                DecisionPath::Desynced => "desynced",
            };
            return Ok(serde_json::json!({
                "i": spot.i,
                "hole": spot.hole,
                "board": spot.board[..seen.len()],
                "logged": logged,
                "path": path,
                "fold": t.prob(|m| m == RealMove::Fold),
                "call": t.prob(|m| m == RealMove::Call),
                "check": t.prob(|m| m == RealMove::Check),
                "raise": t.prob(|m| matches!(m, RealMove::BetTo(_))),
            })
            .to_string());
        }
    }
    Err(format!("target offset {} is not one of our decisions in {:?}", spot.target, spot.action))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_split_on_moves_not_digits_or_streets() {
        let a = "b200c/kb100c/kk/";
        assert_eq!(token_starts(a), vec![0, 4, 6, 7, 11, 13, 14]);
        assert_eq!(token_at(a, 0), "b200");
        assert_eq!(token_at(a, 7), "b100");
        assert_eq!(token_at(a, 11), "c");
    }

    /// A real logged turn spot (hand 8 of the 10k log): we limp the small
    /// blind, the flop checks through, the big blind leads the turn and we
    /// folded.  The replay must force our two earlier moves, reach the turn
    /// decision through the resolver, and report a distribution over the
    /// moves legal facing a bet.
    #[test]
    fn replays_a_logged_turn_spot() {
        use poker_ai::games::blueprint::BlueprintHoldem;
        use poker_ai::play::{BotConfig, CompactPolicy};
        let cfg = BotConfig {
            resolve_river: false,
            resolve_flop: false,
            turn_iters: 20,
            river_cap: 1,
            continuations: vec![0.0],
            runout_sample: 8,
            purify: 0.0,
            ..Default::default()
        };
        let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
        let mut bot = Bot::new(game, CompactPolicy::from_entries(vec![]), cfg);
        let s = |v: &[&str]| v.iter().map(|c| c.to_string()).collect::<Vec<_>>();
        let spot = Spot {
            i: 8,
            pos: 1,
            hole: s(&["Ks", "6h"]),
            board: s(&["9h", "7c", "6d", "2s"]),
            action: "ck/kk/b100f".into(),
            target: 10,
        };
        let line = replay_one(&mut bot, &spot).expect("replays");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["path"], "resolve");
        assert_eq!(v["logged"], "f");
        assert_eq!(v["board"].as_array().unwrap().len(), 4);
        let p = |k: &str| v[k].as_f64().unwrap();
        assert!((p("fold") + p("call") + p("raise") - 1.0).abs() < 1e-9, "{line}");
        assert_eq!(p("check"), 0.0, "no check facing a bet");
        assert_eq!(bot.tracking_counts().decisions, 3, "limp, flop check, turn");
        assert_eq!(bot.forced_mismatches(), 0);

        // A target that is Slumbot's move is an error, not a silent skip.
        let wrong = Spot { target: 6, ..spot };
        assert!(replay_one(&mut bot, &wrong).is_err());
    }
}
