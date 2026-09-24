//! `play chart` — the blueprint's preflop strategy as 13×13 grids.

use poker_ai::games::Game;
use poker_ai::util::cli::{data_dir, flag};
use poker_core::action::Action;
use poker_core::make_card;
use poker_core::state::NO_CARD;

use super::{cap_flag, load_game, load_policy, validate, ABSTRACT_BB};

/// Dump the blueprint's preflop strategy — the SB open chart and the BB
/// response to the smallest SB open — so a preflop leak (over-folding the BB,
/// too-tight opens) is visible at a glance.
///
///   play chart [--data=DIR --stack-bb=N --cap=N]
pub(super) fn run(args: &[String]) {
    validate(args, &["data", "stack-bb", "cap", "policy"], 0);
    let dir = data_dir(args);
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap = cap_flag(args);

    println!("Loading abstraction from {} ({stack_bb}bb, cap-{cap})", dir.display());
    let game = load_game(&dir, stack_bb, cap);
    let policy = load_policy(args, &dir);
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
