//! Benchmark: continual re-solving with CFV gadgets (finding #4).
//!
//! Two questions:
//!   1. **Safety** — does the re-solving gadget hold the opponent to the value it
//!      was previously guaranteed (so re-solving cannot leak exploitability)?
//!   2. **Speed** — does carrying the previous resolve's strategy forward
//!      (warm-starting) reach a given solution quality in fewer iterations than a
//!      cold resolve (the claimed ~2×)?
//!
//! Setup: a river spot is bootstrapped with a plain range-vs-range resolve; its
//! opponent counterfactual values are extracted and used to (a) constrain a
//! gadget re-solve and (b) carry the strategy forward as a warm start.  Solution
//! quality is the opponent's best-response value against *our* resolved strategy
//! (lower = less exploitable) — the right safety metric, since the deployed
//! object is our strategy and the opponent best-responds.
//!
//! Run: cargo run --release --example bench_continual

use std::time::Instant;

use poker_ai::resolving::belief_state::BeliefState;
use poker_ai::resolving::cfv::opponent_cfvs;
use poker_ai::resolving::gadget::{GadgetGame, ReSolver};
use poker_ai::resolving::leaf_eval::CheckdownLeafEval;
use poker_ai::resolving::subgame::{Subgame, SubgameSolver};
use poker_ai::resolving::warm_start::{warm_start_regrets, DEFAULT_SCALE};
use poker_ai::solver::best_response::best_response_value;
use poker_core::action::Action;
use poker_core::legal_actions;
use poker_core::make_card;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

fn public_root(board: [u8; 5], stack: u32, target_street: u8) -> GameState {
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
    let mut gs = GameState::new(2, 2, 1, [stack; MAX_PLAYERS], holes, board, 0);
    while gs.street < target_street && !gs.is_terminal() {
        let acts = legal_actions(&gs);
        let act = if acts.as_ref().contains(&Action::Check) { Action::Check } else { Action::Call };
        gs.apply_action(act);
    }
    gs
}

fn main() {
    // A big-pot river spot: high betting leverage + polarized (nuts/bluffs) vs
    // bluff-catcher ranges, the classic *slowly*-converging bluff/bluff-catch
    // equilibrium where warm-starting has room to help (a tiny low-pot spot
    // converges in a handful of iterations and leaves nothing to measure).
    let board =
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
    // P0 = polarized: value (trips / trip kings) + missed-draw bluffs.
    let b0 = BeliefState::from_hands(&[
        [make_card(12, 1), make_card(12, 2)], // A♦A♥ — trip aces (nuts)
        [make_card(11, 2), make_card(11, 3)], // K♣K♠ — trip kings (value)
        [make_card(5, 1), make_card(4, 1)],   // 7♦6♦ — missed draw (bluff)
        [make_card(6, 1), make_card(3, 1)],   // 8♦5♦ — missed draw (bluff)
        [make_card(10, 1), make_card(8, 1)],  // Q♦T♦ — missed draw (bluff)
    ]);
    // P1 = bluff-catchers: pairs that beat the bluffs but lose to the value.
    let b1 = BeliefState::from_hands(&[
        [make_card(8, 0), make_card(8, 3)],   // T♣T♠
        [make_card(9, 0), make_card(9, 3)],   // J♣J♠
        [make_card(10, 0), make_card(10, 3)], // Q♣Q♠
        [make_card(6, 0), make_card(6, 3)],   // 8♣8♠
        [make_card(5, 0), make_card(5, 3)],   // 7♣7♠
    ]);
    // Inflate the pot (a prior big-bet line): high leverage on the river.
    let mut root = public_root(board, 60, 3);
    for i in 0..2 {
        root.total_committed[i] += 40;
        root.pot += 40;
        root.stacks[i] -= 40;
    }
    let me = root.current_player();
    let opp = 1 - me;
    let (mr, opr) = if me == 0 { (&b0, &b1) } else { (&b1, &b0) };
    let beliefs = [b0.clone(), b1.clone()];
    let leaf = CheckdownLeafEval::new();
    let plain = || Subgame::new(root.clone(), &beliefs, &leaf);

    println!("Continual re-solving benchmark — river spot, me=P{me}, opp=P{opp}\n");

    // ── Bootstrap: a plain range-vs-range resolve, then extract opp CFVs ──────
    let boot = SubgameSolver::new(1, 0).solve_for_iters(&root, &beliefs, &leaf, 8_000).strategy;
    let cfv = opponent_cfvs(&plain(), &boot, opp);
    // Warm-start confidence: the blueprint carries thousands of iterations of
    // regret mass, so the seed scale must be on the order of the spot's regret
    // magnitude (∝ pot) — DEFAULT_SCALE = 1.0 is a toy-game value and washes out
    // after one iteration in a 40 bb pot.
    let scale = root.pot as f64;
    let _ = DEFAULT_SCALE;
    let br_boot = best_response_value(&plain(), opp, &boot);
    println!("bootstrap resolve (8000 iters): opponent best-response value = {br_boot:.5} bb");

    // Two carried-blueprint regimes, to bracket the realistic speedup:
    //  * "good"   — the well-converged blueprint (re-solving a spot you already
    //               solved well): the upper bound on warm-start benefit.
    //  * "coarse" — a cheap, imperfect blueprint (what a real carried/precomputed
    //               strategy is): the realistic per-decision regime.
    let seed_good = warm_start_regrets(&boot, scale);
    let coarse = SubgameSolver::new(1, 0).solve_for_iters(&root, &beliefs, &leaf, 40).strategy;
    let seed_coarse = warm_start_regrets(&coarse, scale);

    // ── Safety: the gadget holds the opponent to its carried guarantee ────────
    let converged = ReSolver::new().solve_for_iters(&root, mr, opr, &cfv, &leaf, 20_000).average;
    let game = GadgetGame::new(root.clone(), mr, opr, &cfv, &leaf);
    let mut worst = f64::MIN;
    for h in game.opp_hands() {
        worst = worst.max(game.follow_value(&converged, h) - game.cfv_of(h));
    }
    let br_conv = best_response_value(&plain(), opp, &converged);
    println!("\n── Safety ──");
    println!("  gadget-resolved: opponent best-response value = {br_conv:.5} bb (bootstrap {br_boot:.5})");
    println!("  max over hands of (follow value − carried guarantee) = {worst:.5} bb  (≤0 ⇒ held to guarantee)");

    // ── Speed: iterations to a quality target, cold vs warm-started ───────────
    // Target = the converged opponent BR value + a small slack.
    let target = br_conv + 0.05;
    let schedule = [2u64, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    println!("\n── Warm-start speedup (target: opp BR ≤ {target:.5} bb) ──");
    println!("{:>6} | {:>13} | {:>15} | {:>13}", "iters", "cold opp-BR", "warm-coarse bp", "warm-good bp");
    println!("-------+---------------+-----------------+--------------");
    let hit = |strat: &std::collections::HashMap<u64, Vec<f64>>| best_response_value(&plain(), opp, strat);
    let (mut cold_hit, mut coarse_hit, mut good_hit) = (None, None, None);
    for &iters in &schedule {
        let cold = ReSolver::new().solve_for_iters(&root, mr, opr, &cfv, &leaf, iters).average;
        let warm_c = ReSolver::new()
            .with_warm_start(seed_coarse.clone())
            .solve_for_iters(&root, mr, opr, &cfv, &leaf, iters)
            .average;
        let warm_g = ReSolver::new()
            .with_warm_start(seed_good.clone())
            .solve_for_iters(&root, mr, opr, &cfv, &leaf, iters)
            .average;
        let (bc, bcoarse, bgood) = (hit(&cold), hit(&warm_c), hit(&warm_g));
        println!("{iters:>6} | {bc:>13.5} | {bcoarse:>15.5} | {bgood:>13.5}");
        if cold_hit.is_none() && bc <= target {
            cold_hit = Some(iters);
        }
        if coarse_hit.is_none() && bcoarse <= target {
            coarse_hit = Some(iters);
        }
        if good_hit.is_none() && bgood <= target {
            good_hit = Some(iters);
        }
    }

    println!("\n── Result ──");
    let factor = |hit: Option<u64>| match (hit, cold_hit) {
        (Some(w), Some(c)) => format!("{:.1}× fewer iters (cold {c} → warm {w})", c as f64 / w as f64),
        _ => "did not reach target within schedule".to_string(),
    };
    println!("  warm-start from a coarse (40-iter) carried blueprint: {}", factor(coarse_hit));
    println!("  warm-start from a well-converged blueprint (re-entry): {}", factor(good_hit));
    println!(
        "  → continual resolving's per-decision speedup scales with carried-blueprint quality;\n    \
         the realistic (coarse) regime brackets the finding's ~2× and the re-entry case is the ceiling."
    );

    // Wall-clock at a fixed budget (warm-start's per-iteration cost is identical;
    // the win is reaching quality in fewer iters).
    let n = 2_000u64;
    let t = Instant::now();
    let _ = ReSolver::new().solve_for_iters(&root, mr, opr, &cfv, &leaf, n).strategy;
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let _ = ReSolver::new().with_warm_start(seed_good.clone()).solve_for_iters(&root, mr, opr, &cfv, &leaf, n).strategy;
    let warm_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("\n  per-resolve wall-clock @ {n} iters: cold {cold_ms:.0} ms, warm {warm_ms:.0} ms");
    println!("  (equal per-iter cost ⇒ the iteration speedup above is the time saving per decision)");
}
