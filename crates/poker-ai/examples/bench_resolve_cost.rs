//! Per-decision cost of each play-time resolve mode at deployment depth.
//!
//! A resolve's cost depends only on the PUBLIC tree (stack, pot, raise cap) —
//! not on the blueprint — so this runs anywhere.  Reports public decision
//! nodes and wall-clock per resolve, which is what decides whether a mode is
//! usable in a live match (Slumbot allows ~a few seconds per decision).
//!
//!   cargo run --release --example bench_resolve_cost
//!
//! Each arm reports the minimum per-iteration cost over `POKER_AI_BENCH_REPS`
//! batches of `POKER_AI_BENCH_PROBE` iterations (default 5 x 100).  Raise the
//! probe when sweeping small differences on a busy machine; lower it for the
//! opt-in `all` arms, which are minutes per batch.

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

/// `POKER_AI_BENCH_PROBE` / `POKER_AI_BENCH_REPS`, or the defaults.
fn env_or(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(default)
}

/// Time `reps` batches of `probe` iterations and report the **minimum**,
/// extrapolating to the mode's real iteration count — the expensive modes take
/// tens of minutes at that count, which is itself the finding.
///
/// ## Why the shape of this function matters
///
/// It previously took a built `VectorCfr` by value and ran one 20-iteration
/// batch.  Both halves of that were wrong once the arms got fast:
///
/// * **Construction escaped the clock.**  The solver was built in the *caller's*
///   argument expression, so `build.elapsed()` was reading a stopwatch started
///   one line earlier and always reported ~0.  Table construction is not free
///   and is itself parallel, so it also polluted any whole-process CPU%
///   measurement.  Taking a builder closure puts it back inside the clock.
/// * **The window was far too short.**  20 iterations was tuned when an
///   iteration cost 14–21 ms.  After the denominator hoist the river arm is
///   ~4 ms, making the measured window ~90 ms — short enough on a shared box
///   that a thread-count sweep produced a non-monotone ordering (1 thread
///   apparently beating 16).  That was jitter being read as signal.
///
/// Minimum-of-reps rather than mean: on a machine with other tenants the fast
/// tail is the machine's real capability and the slow tail is someone else's
/// job.  A warm-up batch runs first so pool construction and first-touch page
/// faults land outside the measurement.
fn bench(label: &str, build: impl Fn() -> VectorCfr, target_iters: u64) {
    let probe = env_or("POKER_AI_BENCH_PROBE", 100);
    let reps = env_or("POKER_AI_BENCH_REPS", 5);

    let t = Instant::now();
    let mut solver = build();
    let build_s = t.elapsed().as_secs_f64();
    let nodes = solver.public_node_count();

    solver.run(5);

    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        solver.run(probe);
        let per_iter = t.elapsed().as_secs_f64() / probe as f64;
        best = best.min(per_iter);
        worst = worst.max(per_iter);
    }

    let total = build_s + best * target_iters as f64;
    println!(
        "{label:<36} {nodes:>8} nodes  {:>7.1} ms/iter (max {:>6.1})  build {:>5.1}s  x{target_iters:>5} it = {:>8.1} s/decision{}",
        best * 1000.0,
        worst * 1000.0,
        build_s,
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
    println!(
        "Minimum of {} x {} iterations, after a warm-up batch; build time is measured separately.",
        env_or("POKER_AI_BENCH_REPS", 5),
        env_or("POKER_AI_BENCH_PROBE", 100),
    );
    if heavy {
        println!("Running the HEAVY arms too — expect several GB of RSS.\n");
    } else {
        println!("Light arms only; pass `all` for the multi-GB full-river/flop modes.\n");
    }

    let r = public_root_at(river, 3);
    bench("river (exact, 1500 it)", || VectorCfr::new_capped(&r, &ranges, 3), 1500);

    let t = public_root_at(turn, 2);
    bench(
        "turn checkdown K=4 (500 it)",
        || VectorCfr::new_capped_multi(&t, &ranges, 3, vec![0.0, 0.75, 1.5, 3.0]),
        500,
    );
    if !heavy {
        return;
    }
    bench(
        "turn FULL-RIVER (500 it)  [default]",
        || VectorCfr::new_full(&t, &ranges, 3, vec![0.0], true),
        500,
    );
    bench(
        "turn FULL-RIVER (100 it)",
        || VectorCfr::new_full(&t, &ranges, 3, vec![0.0], true),
        100,
    );
    bench(
        "turn FULL-RIVER cap1 (500 it)",
        || VectorCfr::new_full(&t, &ranges, 1, vec![0.0], true),
        500,
    );

    let f = public_root_at(flop, 1);
    bench(
        "flop checkdown K=4 (500 it)",
        || VectorCfr::new_capped_multi(&f, &ranges, 3, vec![0.0, 0.75, 1.5, 3.0]),
        500,
    );
}
