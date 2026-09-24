//! `play slumbot` — a match against Slumbot over its HTTP API.

use std::io::Write as _;

use poker_ai::play::cards::parse_cards;
use poker_ai::play::luck::luck_adjustment;
use poker_ai::play::protocol::{parse_action, BIG_BLIND};
use poker_ai::play::slumbot::SlumbotClient;
use poker_ai::play::Bot;
use poker_ai::util::cli::{data_dir, flag};

use super::{build_bot, hands_arg, validate, BOT_FLAGS};

pub(super) fn run(args: &[String]) {
    validate(args, &[BOT_FLAGS, &["log-hands", "token", "username", "password"]].concat(), 1);
    let hands = hands_arg(args);
    let dir = data_dir(args);
    let mut bot = build_bot(args, &dir);

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
                let adj = (winnings as f64 - rec.luck - rec.action_luck) / BIG_BLIND as f64;
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
                    // Blueprint hit rate on every progress line: the uniform
                    // fallback is silent, and a bot running on uniform
                    // fallbacks plays legal poker, logs no errors, and loses.
                    // Worth seeing at hand 100, not hand 2000.
                    let lookups = bot.lookup_counts();
                    println!("         {lookups}");
                    println!(
                        "@wandb {{\"hand\":{played},\"net_bb\":{net_bb:.2},\"bb100\":{:.2},\"bb100_ci\":{ci:.2},\"adj_bb100\":{:.2},\"adj_bb100_ci\":{adj_ci:.2},\"bp_hit_rate\":{:.4}}}",
                        mean * 100.0,
                        adj_mean * 100.0,
                        lookups.hit_rate()
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
    let lookups = bot.lookup_counts();
    println!("Blueprint lookups: {lookups}");
    println!("Abstraction tracking: {}", bot.tracking_counts());
    // A low hit rate invalidates the match rather than reporting it: the bot was
    // playing uniform-random in the abstract game and feeding the range tracker
    // likelihoods the blueprint never produced.  Loud, because the failure mode
    // is that everything else looks normal.
    if lookups.hit_rate() < 0.9 && lookups.total() > 0 {
        eprintln!(
            "WARNING: only {:.1}% of blueprint lookups hit. This match does not measure the \
             blueprint — check that --cap and --stack-bb match the training run, and that \
             --policy points at the artifact those settings produced.",
            100.0 * lookups.hit_rate()
        );
    }
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
    /// computable.
    luck: f64,
    /// AIVAT action-luck term for our own re-solved decisions, in chips
    /// (`HandState::action_luck_chips`).  Adjusted winnings =
    /// `winnings - luck - action_luck`.
    action_luck: f64,
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
            "action_luck": self.action_luck,
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
                action_luck: hs.action_luck_chips(),
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
