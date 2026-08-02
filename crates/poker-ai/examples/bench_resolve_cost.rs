//! Per-decision cost of each play-time resolve mode at deployment depth.
//!
//! A resolve's cost depends only on the PUBLIC tree (stack, pot, raise cap) —
//! not on the blueprint — so this runs anywhere.  Reports public decision
//! nodes and wall-clock per resolve, which is what decides whether a mode is
//! usable in a live match (Slumbot allows ~a few seconds per decision).
//!
//!   cargo run --release --example bench_resolve_cost

use std::time::Instant;

use poker_ai::resolving::belief_state::BeliefState;
use poker_ai::resolving::vector_cfr::VectorCfr;
use poker_core::action::Action;
use poker_core::legal_actions;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

/// Slumbot's structure: 50/100 blinds, 200 bb stacks.
const BB: u32 = 100;
const SB: u32 = 50;
const STACK: u32 = 20_000;

/// A public state reached by checking/calling to `target_street` — a deep,
/// realistic spot (small pot, ~full stacks behind = the largest tree).
fn public_root_at(board: [u8; 5], target_street: u8) -> GameState {
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    let mut used = 0u64;
    for &c in &board {
        if c != NO_CARD {
            used |= 1 << c;
        }
    }
    let mut spare = (0u8..52).filter(|&c| used & (1 << c) == 0);
    holes[0] = [spare.next().unwrap(), spare.next().unwrap()];
    holes[1] = [spare.next().unwrap(), spare.next().unwrap()];
    let mut gs = GameState::new(2, BB, SB, [STACK; MAX_PLAYERS], holes, board, 0);
    while gs.street < target_street && !gs.is_terminal() {
        let acts = legal_actions(&gs);
        let act = if acts.contains(&Action::Check) { Action::Check } else { Action::Call };
        gs.apply_action(act);
    }
    gs
}

fn card(rank: u8, suit: u8) -> u8 {
    rank * 4 + suit
}

/// Time a few iterations and extrapolate — the expensive modes take tens of
/// minutes at their real iteration counts, which is itself the finding.
fn bench(label: &str, mut solver: VectorCfr, target_iters: u64) {
    let build = Instant::now();
    let build_s = build.elapsed().as_secs_f64();
    let nodes = solver.public_node_count();
    // Enough probe iterations that the per-iteration figure is stable: at 4 the
    // river arm swung 14–21 ms/iter run to run, which is wider than most of the
    // differences worth measuring here.
    const PROBE: u64 = 20;
    let t = Instant::now();
    solver.run(PROBE);
    let per_iter = t.elapsed().as_secs_f64() / PROBE as f64;
    let total = build_s + per_iter * target_iters as f64;
    println!(
        "{label:<36} {nodes:>8} nodes  {:>8.1} ms/iter  x{target_iters:>5} it = {:>9.1} s/decision{}",
        per_iter * 1000.0,
        total,
        if total > 5.0 { "   <-- UNUSABLE LIVE" } else { "" }
    );
}

fn main() {
    // A♣ K♦ 9♥ 4♠ (turn) and the same with a river.
    let turn = [card(12, 0), card(11, 1), card(7, 2), card(2, 3), NO_CARD];
    let river = [card(12, 0), card(11, 1), card(7, 2), card(2, 3), card(0, 0)];
    let flop = [card(12, 0), card(11, 1), card(7, 2), NO_CARD, NO_CARD];
    let ranges = [BeliefState::uniform(), BeliefState::uniform()];

    // The full-river and flop arms build tens of thousands of decision nodes,
    // each carrying two `1326 × actions` accumulator arrays — several GB of
    // resident stores.  That is a finding about those modes, but it also means
    // the benchmark cannot run on a small machine, so those arms are opt-in.
    let heavy = std::env::args().any(|a| a == "all");

    println!("Resolve cost at 200bb (blinds {SB}/{BB}), raise cap 3, full 1326-hand ranges.");
    println!("VectorCfr is SINGLE-THREADED — these are one-core timings.");
    if heavy {
        println!("Running the HEAVY arms too — expect several GB of RSS.\n");
    } else {
        println!("Light arms only; pass `all` for the multi-GB full-river/flop modes.\n");
    }

    let r = public_root_at(river, 3);
    bench("river (exact, 1500 it)", VectorCfr::new_capped(&r, &ranges, 3), 1500);

    let t = public_root_at(turn, 2);
    bench(
        "turn checkdown K=4 (500 it)",
        VectorCfr::new_capped_multi(&t, &ranges, 3, vec![0.0, 0.75, 1.5, 3.0]),
        500,
    );
    if !heavy {
        return;
    }
    bench(
        "turn FULL-RIVER (500 it)  [default]",
        VectorCfr::new_full(&t, &ranges, 3, vec![0.0], true),
        500,
    );
    bench(
        "turn FULL-RIVER (100 it)",
        VectorCfr::new_full(&t, &ranges, 3, vec![0.0], true),
        100,
    );
    bench(
        "turn FULL-RIVER cap1 (500 it)",
        VectorCfr::new_full(&t, &ranges, 1, vec![0.0], true),
        500,
    );

    let f = public_root_at(flop, 1);
    bench(
        "flop checkdown K=4 (500 it)",
        VectorCfr::new_capped_multi(&f, &ranges, 3, vec![0.0, 0.75, 1.5, 3.0]),
        500,
    );
}
