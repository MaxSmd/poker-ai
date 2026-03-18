//! Criterion benchmarks for the hot-path evaluator functions and game-state
//! operations.
//!
//! Run with:
//!   cargo bench -p poker-core
//!
//! Results are written to `target/criterion/` as HTML reports.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use poker_core::{
    evaluate_5, evaluate_6, evaluate_7, make_card,
    Action, GameState, MAX_PLAYERS, NO_CARD,
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

fn bench_evaluate_6(c: &mut Criterion) {
    let h = hand6();
    c.bench_function("evaluate_6", |b| {
        b.iter(|| evaluate_6(black_box(&h)))
    });
}

fn bench_evaluate_7(c: &mut Criterion) {
    let h = hand7();
    c.bench_function("evaluate_7", |b| {
        b.iter(|| evaluate_7(black_box(&h)))
    });
}

// ── Game-state benchmarks ─────────────────────────────────────────────────────

fn make_bench_state() -> GameState {
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    for i in 0..6 {
        holes[i] = [make_card(i as u8, 0), make_card(i as u8 + 1, 1)];
    }
    let board = [
        make_card(2, 0),
        make_card(3, 1),
        make_card(4, 2),
        make_card(5, 3),
        make_card(6, 0),
    ];
    GameState::new(6, 10, [1000u32; MAX_PLAYERS], holes, board, 0)
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

// ── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    eval_benches,
    bench_evaluate_5,
    bench_evaluate_6,
    bench_evaluate_7
);
criterion_group!(state_benches, bench_apply_undo, bench_legal_actions);
criterion_main!(eval_benches, state_benches);
