//! Tree traversal benchmark.
//! Output: full Kuhn Poker tree traversals per second.
//!
//! Kuhn Poker is mapped onto poker-core's 2-player heads-up NLHE engine:
//!   - 3-card deck: Jack (rank 0), Queen (rank 1), King (rank 2)
//!   - 6 possible deals (one card per player, no repeats)
//!   - Minimal stack configuration (3 chips each, BB=2) to keep the tree small
//!
//! Two traversal algorithms are benchmarked and compared:
//!   1. **Recursive** – straightforward DFS; Rust's optimizer can keep
//!      intermediate values in registers across recursive calls.
//!   2. **Explicit stack** – iterative DFS; avoids OS call-stack overhead by
//!      maintaining a fixed-size frame array.
//!
//! Neither algorithm allocates on the heap in the hot path; the undo history
//! is maintained inside `GameState`'s pre-allocated `UndoStack`.

use std::time::Instant;

use poker_core::betting::abstract_raise_amounts;
use poker_core::evaluator::make_card;
use poker_core::{Action, GameState, MAX_PLAYERS, NO_CARD};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum legal actions per node (mirrors `legal_actions`'s pre-allocated capacity).
const MAX_ACTIONS: usize = 8;

/// Maximum tree depth (consecutive `apply_action` calls without an `undo`).
/// A 2-player 3-chip-stack game across ≤4 streets is well within 64 plies.
const MAX_DEPTH: usize = 64;

/// All valid Kuhn Poker deals: `(player-0 rank, player-1 rank)`.
/// Ranks: 0 = Jack, 1 = Queen, 2 = King.  Each deal assigns a distinct card
/// to each player from the 3-card Kuhn deck.
const KUHN_DEALS: [(u8, u8); 6] = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];

// ─────────────────────────────────────────────────────────────────────────────
// Game setup
// ─────────────────────────────────────────────────────────────────────────────

/// Build a Kuhn-Poker-like [`GameState`] for the given card deal.
///
/// # Mapping to poker-core heads-up NLHE
///
/// | Parameter       | Value                                |
/// |-----------------|--------------------------------------|
/// | Players         | 2 (heads-up)                         |
/// | Button          | player 0                             |
/// | Big blind       | 2 chips                              |
/// | Starting stacks | 3 chips each                         |
/// | Hole cards      | 1 Kuhn card per player, duplicated   |
/// | Board cards     | none (`NO_CARD`)                     |
///
/// After posting blinds SB (player 1) has 2 chips left and BB (player 0) has
/// 1 chip left, which tightly limits the number of legal raise sizes and keeps
/// the per-deal game tree small.
///
/// Each player's single Kuhn card is encoded in both hole-card slots using
/// different suits so that the hand evaluator never receives a `NO_CARD` value.
fn make_kuhn_state(p0_rank: u8, p1_rank: u8) -> GameState {
    let mut stacks = [0u32; MAX_PLAYERS];
    stacks[0] = 3;
    stacks[1] = 3;

    // Encode each Kuhn card as two copies of the same rank (different suits)
    // to satisfy poker-core's 2-hole-card requirement.
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    holes[0][0] = make_card(p0_rank, 0); // e.g. Jack of clubs
    holes[0][1] = make_card(p0_rank, 1); // e.g. Jack of diamonds
    holes[1][0] = make_card(p1_rank, 2); // e.g. Queen of hearts
    holes[1][1] = make_card(p1_rank, 3); // e.g. Queen of spades

    // No board cards – showdown hand evaluation is not invoked during traversal.
    let board = [NO_CARD; 5];

    GameState::new(2, 2, stacks, holes, board, 0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Allocation-free legal action enumeration
// ─────────────────────────────────────────────────────────────────────────────

/// Fill `out` with the legal actions for the current player in `state`.
///
/// Replicates the logic of [`poker_core::legal_actions`] but writes into a
/// caller-supplied fixed-size array so that **no heap allocation is needed**.
///
/// Returns the number of valid entries written into `out[..n]`.
#[inline]
fn fill_legal_actions(state: &GameState, out: &mut [Action; MAX_ACTIONS]) -> usize {
    let mut n = 0usize;

    if state.is_terminal() {
        return 0;
    }

    let p = state.to_act as usize;
    let to_call = state.current_bet.saturating_sub(state.street_bets[p]);
    let max_bet = state.stacks[p] + state.street_bets[p];

    // ── passive options ───────────────────────────────────────────────────────
    if to_call == 0 {
        out[n] = Action::Check;
        n += 1;
    } else {
        out[n] = Action::Fold;
        n += 1;
        if state.stacks[p] > to_call {
            out[n] = Action::Call;
            n += 1;
        } else {
            // Calling would commit all remaining chips – offer AllIn, not Call.
            out[n] = Action::AllIn;
            n += 1;
            return n; // No further raise options when forced all-in to call.
        }
    }

    // ── aggressive options ────────────────────────────────────────────────────
    let min_raise_total = state.current_bet + state.min_raise;
    if max_bet <= state.current_bet {
        // Player cannot raise (no chips above the call amount).
        return n;
    }

    let pot = state.pot();
    let (abstract_bets, nb) =
        abstract_raise_amounts(pot, state.current_bet, state.street, state.big_blind);

    let mut allin_added = false;
    for &bet_level in abstract_bets[..nb].iter() {
        if bet_level < min_raise_total {
            continue;
        }
        if bet_level >= max_bet {
            if !allin_added && n < MAX_ACTIONS {
                out[n] = Action::AllIn;
                n += 1;
                allin_added = true;
            }
            break;
        }
        if n < MAX_ACTIONS {
            out[n] = Action::Raise(bet_level);
            n += 1;
        }
    }

    if !allin_added && max_bet >= min_raise_total && n < MAX_ACTIONS {
        out[n] = Action::AllIn;
        n += 1;
    }

    n
}

// ─────────────────────────────────────────────────────────────────────────────
// Version 1 – Recursive traversal
// ─────────────────────────────────────────────────────────────────────────────

/// Recursively traverse the game tree rooted at `state` to all terminal leaves.
///
/// Returns the number of terminal nodes reached.  No heap allocation occurs in
/// the hot path: the action buffer is stack-allocated on each call frame.
fn traverse_recursive(state: &mut GameState) -> u64 {
    if state.is_terminal() {
        return 1;
    }

    let mut actions = [Action::Fold; MAX_ACTIONS];
    let n = fill_legal_actions(state, &mut actions);
    let mut count = 0u64;

    for &action in actions[..n].iter() {
        state.apply_action(action);
        count += traverse_recursive(state);
        state.undo_action();
    }

    count
}

/// Traverse the complete Kuhn Poker tree (all 6 deals) recursively.
///
/// Returns the total number of terminal nodes visited across all deals.
fn full_traversal_recursive() -> u64 {
    let mut total = 0u64;
    for &(p0, p1) in &KUHN_DEALS {
        let mut state = make_kuhn_state(p0, p1);
        total += traverse_recursive(&mut state);
    }
    total
}

// ─────────────────────────────────────────────────────────────────────────────
// Version 2 – Explicit stack traversal
// ─────────────────────────────────────────────────────────────────────────────

/// One frame of the explicit traversal stack.
///
/// Stores the legal actions available at a particular node and the index of the
/// next action to explore.  The game state itself is maintained in-place via
/// `apply_action` / `undo_action`, so no game-state snapshot is stored here.
#[derive(Copy, Clone)]
struct StackFrame {
    actions: [Action; MAX_ACTIONS],
    n_actions: usize,
    next_idx: usize,
}

impl StackFrame {
    #[inline]
    fn empty() -> Self {
        Self {
            actions: [Action::Fold; MAX_ACTIONS],
            n_actions: 0,
            next_idx: 0,
        }
    }
}

/// Traverse the game tree rooted at `state` using an explicit stack.
///
/// The algorithm is semantically identical to [`traverse_recursive`] but
/// manages its own call stack with a fixed-size array of [`StackFrame`]s,
/// avoiding OS-managed call-stack overhead for deep trees.
///
/// Returns the number of terminal nodes reached.  No heap allocation in the
/// hot path.
fn traverse_explicit_stack(state: &mut GameState) -> u64 {
    let mut stack = [StackFrame::empty(); MAX_DEPTH];
    let mut depth = 0usize;
    let mut count = 0u64;

    // Seed the root frame with the initial legal actions.
    stack[0].n_actions = fill_legal_actions(state, &mut stack[0].actions);
    stack[0].next_idx = 0;

    loop {
        let frame = &mut stack[depth];

        if frame.next_idx >= frame.n_actions {
            // All actions at this level have been explored.
            if depth == 0 {
                break; // Root exhausted – traversal is complete.
            }
            depth -= 1;
            state.undo_action(); // Reverse the action that opened this level.
            continue;
        }

        let action = frame.actions[frame.next_idx];
        frame.next_idx += 1;

        state.apply_action(action);

        if state.is_terminal() {
            count += 1;
            state.undo_action();
        } else {
            // Descend: push a new frame for the children of this node.
            depth += 1;
            stack[depth].n_actions = fill_legal_actions(state, &mut stack[depth].actions);
            stack[depth].next_idx = 0;
        }
    }

    count
}

/// Traverse the complete Kuhn Poker tree (all 6 deals) with the explicit stack.
///
/// Returns the total number of terminal nodes visited across all deals.
fn full_traversal_explicit_stack() -> u64 {
    let mut total = 0u64;
    for &(p0, p1) in &KUHN_DEALS {
        let mut state = make_kuhn_state(p0, p1);
        total += traverse_explicit_stack(&mut state);
    }
    total
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

use poker_core::evaluator::make_card as mc;
use poker_core::UndoRecord;

fn main() {
    // ── UndoRecord memory footprint ───────────────────────────────────────────
    println!(
        "UndoRecord size (delta):  {} bytes  (old snapshot: ~88 bytes, stack depth 256 → ~{} KB)",
        std::mem::size_of::<UndoRecord>(),
        std::mem::size_of::<UndoRecord>() * 256 / 1024,
    );
    println!();

    // ── Sanity check: both methods must agree on terminal-node count ──────────
    let terminals_rec = full_traversal_recursive();
    let terminals_stk = full_traversal_explicit_stack();

    assert_eq!(
        terminals_rec,
        terminals_stk,
        "BUG: traversal mismatch – recursive={terminals_rec} vs explicit_stack={terminals_stk}",
    );

    println!("Terminal nodes per full traversal (6 Kuhn deals): {terminals_rec}");
    println!();

    // ── Benchmark configuration ───────────────────────────────────────────────
    const ITERATIONS: u64 = 500_000;

    // ── Version 1: Recursive ─────────────────────────────────────────────────
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = std::hint::black_box(full_traversal_recursive());
    }
    let rec_elapsed = t0.elapsed();
    let rec_tps = ITERATIONS as f64 / rec_elapsed.as_secs_f64();

    println!(
        "Recursive:      {:>12.0} traversals/sec  ({:.3}s for {ITERATIONS} iters)",
        rec_tps,
        rec_elapsed.as_secs_f64(),
    );

    // ── Version 2: Explicit stack ─────────────────────────────────────────────
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = std::hint::black_box(full_traversal_explicit_stack());
    }
    let stk_elapsed = t0.elapsed();
    let stk_tps = ITERATIONS as f64 / stk_elapsed.as_secs_f64();

    println!(
        "Explicit stack: {:>12.0} traversals/sec  ({:.3}s for {ITERATIONS} iters)",
        stk_tps,
        stk_elapsed.as_secs_f64(),
    );

    // ── Summary ───────────────────────────────────────────────────────────────
    println!();
    if rec_tps >= stk_tps {
        println!(
            "Winner: Recursive  ({:.2}× faster than explicit stack)",
            rec_tps / stk_tps
        );
    } else {
        println!(
            "Winner: Explicit stack  ({:.2}× faster than recursive)",
            stk_tps / rec_tps
        );
    }

    // ── Evaluator benchmark ───────────────────────────────────────────────────
    println!();
    println!("── Evaluator benchmark ──────────────────────────────────────────");

    // A typical 7-card river hand: two hole cards + five board cards.
    let hand7: [u8; 7] = [
        mc(12, 0), mc(11, 1), // Ah Kd (hole)
        mc(10, 2), mc(9, 3), mc(8, 0), mc(7, 1), mc(6, 2), // Qh Js Tc 9d 8h (board)
    ];
    let hand5: [u8; 5] = [mc(12, 0), mc(11, 1), mc(10, 2), mc(9, 3), mc(8, 0)];
    let hand6: [u8; 6] = [mc(12, 0), mc(11, 1), mc(10, 2), mc(9, 3), mc(8, 0), mc(7, 1)];

    const EVAL_ITERS: u64 = 5_000_000;

    let t0 = Instant::now();
    for _ in 0..EVAL_ITERS {
        let _ = std::hint::black_box(poker_core::evaluate_5(&hand5));
    }
    let e5_elapsed = t0.elapsed();
    let e5_eps = EVAL_ITERS as f64 / e5_elapsed.as_secs_f64();
    println!(
        "evaluate_5:  {:>12.0} evals/sec  ({:.3}s for {EVAL_ITERS} iters)",
        e5_eps,
        e5_elapsed.as_secs_f64(),
    );

    let t0 = Instant::now();
    for _ in 0..EVAL_ITERS {
        let _ = std::hint::black_box(poker_core::evaluate_6(&hand6));
    }
    let e6_elapsed = t0.elapsed();
    let e6_eps = EVAL_ITERS as f64 / e6_elapsed.as_secs_f64();
    println!(
        "evaluate_6:  {:>12.0} evals/sec  ({:.3}s for {EVAL_ITERS} iters)",
        e6_eps,
        e6_elapsed.as_secs_f64(),
    );

    let t0 = Instant::now();
    for _ in 0..EVAL_ITERS {
        let _ = std::hint::black_box(poker_core::evaluate_7(&hand7));
    }
    let e7_elapsed = t0.elapsed();
    let e7_eps = EVAL_ITERS as f64 / e7_elapsed.as_secs_f64();
    println!(
        "evaluate_7:  {:>12.0} evals/sec  ({:.3}s for {EVAL_ITERS} iters)",
        e7_eps,
        e7_elapsed.as_secs_f64(),
    );
}
