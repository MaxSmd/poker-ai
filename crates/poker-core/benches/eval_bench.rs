//! Criterion benchmarks for the hot-path evaluator functions and game-state
//! operations.
//!
//! Run with:
//!   cargo bench -p poker-core
//!
//! Results are written to `target/criterion/` as HTML reports.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use poker_core::{
    evaluate_5, evaluate_5_lut, evaluate_6, evaluate_6_lut, evaluate_7, evaluate_7_lut,
    legal_actions, make_card, Action, GameState, MAX_PLAYERS, NO_CARD,
};

// ── Sample hands ─────────────────────────────────────────────────────────────

fn hand5() -> [u8; 5] {
    [
        make_card(12, 0), // Ac
        make_card(11, 1), // Kd
        make_card(10, 2), // Qh
        make_card(9, 3),  // Js
        make_card(8, 0),  // Tc  → royal-straight (no flush)
    ]
}

fn hand6() -> [u8; 6] {
    [
        make_card(12, 0),
        make_card(11, 1),
        make_card(10, 2),
        make_card(9, 3),
        make_card(8, 0),
        make_card(7, 1), // 9d → straight
    ]
}

fn hand7() -> [u8; 7] {
    [
        make_card(12, 0),
        make_card(11, 1),
        make_card(10, 2),
        make_card(9, 3),
        make_card(8, 0),
        make_card(7, 1),
        make_card(6, 2), // 8h → straight
    ]
}

// ── Evaluator benchmarks ──────────────────────────────────────────────────────

fn bench_evaluate_5(c: &mut Criterion) {
    let h = hand5();
    c.bench_function("evaluate_5", |b| {
        b.iter(|| evaluate_5(black_box(&h)))
    });
}

fn bench_evaluate_5_lut(c: &mut Criterion) {
    let h = hand5();
    c.bench_function("evaluate_5_lut", |b| {
        b.iter(|| evaluate_5_lut(black_box(&h)))
    });
}

fn bench_evaluate_6(c: &mut Criterion) {
    let h = hand6();
    c.bench_function("evaluate_6", |b| {
        b.iter(|| evaluate_6(black_box(&h)))
    });
}

fn bench_evaluate_6_lut(c: &mut Criterion) {
    let h = hand6();
    c.bench_function("evaluate_6_lut", |b| {
        b.iter(|| evaluate_6_lut(black_box(&h)))
    });
}

fn bench_evaluate_7(c: &mut Criterion) {
    let h = hand7();
    c.bench_function("evaluate_7", |b| {
        b.iter(|| evaluate_7(black_box(&h)))
    });
}

fn bench_evaluate_7_lut(c: &mut Criterion) {
    let h = hand7();
    c.bench_function("evaluate_7_lut", |b| {
        b.iter(|| evaluate_7_lut(black_box(&h)))
    });
}

// ── Game-state benchmarks ─────────────────────────────────────────────────────

fn make_bench_state() -> GameState {
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    for i in 0..6 {
        holes[i] = [make_card(i as u8, 0), make_card(i as u8 + 1, 1)];
    }
    // Board uses ranks 7-11 (above every hole rank, which top out at 6) so no
    // card collides with a hole card — the uniqueness invariant is enforced in
    // release builds, and benchmarks compile in release.
    let board = [
        make_card(7, 2),
        make_card(8, 2),
        make_card(9, 2),
        make_card(10, 3),
        make_card(11, 3),
    ];
    GameState::new(6, 10, 5, [1000u32; MAX_PLAYERS], holes, board, 0)
}

/// Benchmark a single apply_action + undo_action cycle.
fn bench_apply_undo(c: &mut Criterion) {
    let base = make_bench_state();
    c.bench_function("apply_action+undo", |b| {
        let mut gs = base.clone();
        b.iter(|| {
            gs.apply_action(black_box(Action::Call));
            gs.undo_action();
        })
    });
}

/// Benchmark legal_actions generation.
fn bench_legal_actions(c: &mut Criterion) {
    let gs = make_bench_state();
    c.bench_function("legal_actions", |b| {
        b.iter(|| poker_core::legal_actions(black_box(&gs)))
    });
}

// ── End-to-end CFR-traversal benchmark ────────────────────────────────────────

/// Build a heads-up game with a fixed board so that every leaf reaches a real
/// showdown (exercising `terminal_payoffs` → `player_hand_rank` → the LUT).
///
/// Modest stacks keep the abstracted betting tree finite and fast to walk while
/// still hitting every action type (fold / check / call / raise / all-in) and a
/// mix of fold-out and showdown leaves.  `stack` tunes the tree size: larger
/// stacks allow more raise rounds and grow the tree super-linearly.
fn make_cfr_root(stack: u32) -> GameState {
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    holes[0] = [make_card(12, 0), make_card(11, 0)]; // A K (suited)
    holes[1] = [make_card(10, 1), make_card(9, 1)]; // Q J (suited)
    let board = [
        make_card(8, 2), // T
        make_card(7, 3), // 9
        make_card(6, 0), // 8
        make_card(5, 1), // 7
        make_card(3, 2), // 5  → board makes a straight available to both
    ];
    let mut stacks = [0u32; MAX_PLAYERS];
    stacks[0] = stack;
    stacks[1] = stack;
    GameState::new(2, 10, 5, stacks, holes, board, 0)
}

/// Recursively walk the entire game tree exactly as a vanilla-CFR pass would:
/// `legal_actions` at every decision node, `apply_action` / `undo_action`
/// around each child, and `terminal_payoffs` at every leaf.  Returns the number
/// of nodes visited so the caller can report per-node throughput.
fn walk_tree(gs: &mut GameState, nodes: &mut u64) {
    *nodes += 1;
    if gs.is_terminal() {
        black_box(gs.terminal_payoffs());
        return;
    }
    let actions = legal_actions(gs);
    for &a in actions.iter() {
        gs.apply_action(a);
        walk_tree(gs, nodes);
        gs.undo_action();
    }
}

/// Benchmark a full game-tree traversal — the realistic CFR workload, rather
/// than an isolated microbenchmark.  Reports throughput in nodes/sec so the
/// number is stable across tree-size tuning.
fn bench_cfr_traversal(c: &mut Criterion) {
    // Stack chosen so the tree is large enough to be representative yet small
    // enough that each criterion sample stays in the millisecond range.
    const STACK: u32 = 60;

    // Count the tree once to set throughput (and confirm the walk is non-trivial).
    let mut node_count = 0u64;
    walk_tree(&mut make_cfr_root(STACK), &mut node_count);

    let mut group = c.benchmark_group("cfr_traversal");
    group.throughput(Throughput::Elements(node_count));
    group.bench_function("heads_up_full_tree", |b| {
        b.iter(|| {
            let mut gs = make_cfr_root(black_box(STACK));
            let mut nodes = 0u64;
            walk_tree(&mut gs, &mut nodes);
            black_box(nodes)
        })
    });
    group.finish();
}

// ── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    eval_benches,
    bench_evaluate_5,
    bench_evaluate_5_lut,
    bench_evaluate_6,
    bench_evaluate_6_lut,
    bench_evaluate_7,
    bench_evaluate_7_lut,
);
criterion_group!(state_benches, bench_apply_undo, bench_legal_actions);
criterion_group!(cfr_benches, bench_cfr_traversal);
criterion_main!(eval_benches, state_benches, cfr_benches);
