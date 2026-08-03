//! Tests for the explicit-deal subgame — the resolver's correctness oracle.
//!
//! These are what make the fast vectorized solver trustworthy: exact best
//! response over a real, small, fully-enumerable subgame.

use super::*;
use crate::validation::resolving::leaf_eval::CheckdownLeafEval;
use crate::validation::resolving::warm_start::warm_start_regrets;
use crate::validation::solver::best_response::exploitability;
use poker_core::action::Action;
use poker_core::make_card;
use poker_core::state::MAX_PLAYERS;

/// Drive a fresh hand to the start of `target_street` by checking/calling
/// through, producing a clean public root via real poker-core mechanics.
/// Hole cards are placeholders (overwritten per deal in the subgame).
fn public_root(board: [u8; 5], stack: u32, target_street: u8) -> GameState {
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    // Placeholders distinct from the board and each other (skip empty slots).
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
        // Prefer Check; otherwise Call — never put extra money in.
        let act = if acts.contains(&Action::Check) {
            Action::Check
        } else {
            Action::Call
        };
        gs.apply_action(act);
    }
    gs
}

fn river_board() -> [u8; 5] {
    // A♣ K♦ 9♥ 4♠ 2♣
    [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)]
}

#[test]
fn deals_respect_card_removal() {
    let root = public_root(river_board(), 20, 3);
    // Ranges that include hands overlapping the board and each other.
    let b0 = BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)], river_board()[0..2].try_into().unwrap()]);
    let b1 = BeliefState::from_hands(&[[make_card(10, 0), make_card(10, 1)], [make_card(12, 1), make_card(8, 0)]]);
    let leaf = CheckdownLeafEval::new();
    let sg = Subgame::new(root, &[b0, b1], &leaf);
    // No deal may reuse a board card or share a card across the two hands.
    let board_mask: u64 = river_board().iter().fold(0, |m, &c| m | (1 << c));
    for d in &sg.deals {
        let m0 = (1u64 << d.h0[0]) | (1u64 << d.h0[1]);
        let m1 = (1u64 << d.h1[0]) | (1u64 << d.h1[1]);
        assert_eq!(m0 & board_mask, 0, "hole on board");
        assert_eq!(m1 & board_mask, 0, "hole on board");
        assert_eq!(m0 & m1, 0, "hands share a card");
    }
    assert!(sg.num_deals() > 0);
}

#[test]
fn payoffs_are_zero_sum_everywhere() {
    let root = public_root(river_board(), 20, 3);
    let b0 = BeliefState::from_hands(&[
        [make_card(12, 1), make_card(12, 2)], // trip aces
        [make_card(10, 0), make_card(9, 0)],  // weak
    ]);
    let b1 = BeliefState::from_hands(&[
        [make_card(11, 0), make_card(11, 2)], // trip kings
        [make_card(8, 0), make_card(8, 1)],   // pair
    ]);
    let leaf = CheckdownLeafEval::new();
    let sg = Subgame::new(root, &[b0, b1], &leaf);

    fn walk(g: &Subgame, s: &SubgameNode) {
        if g.is_terminal(s) {
            let (u0, u1) = (g.utility(s, 0), g.utility(s, 1));
            assert!((u0 + u1).abs() < 1e-9, "payoffs must sum to zero: {u0} + {u1}");
            return;
        }
        if g.is_chance(s) {
            for (c, _) in g.chance_outcomes(s) {
                walk(g, &c);
            }
        } else {
            for a in 0..g.num_actions(s) {
                walk(g, &g.apply(s, a));
            }
        }
    }
    walk(&sg, &sg.root());
}

#[test]
fn river_subgame_resolves_to_low_exploitability() {
    // The end-to-end resolver check: belief ranges → subgame → CFR+, and the
    // resolved strategy is near-optimal *within the subgame* (measured by the
    // exact best response, which the enumerable chance makes feasible).
    let root = public_root(river_board(), 20, 3);
    let b0 = BeliefState::from_hands(&[
        [make_card(12, 1), make_card(12, 2)], // nuts-ish (trips)
        [make_card(6, 0), make_card(5, 0)],   // air
    ]);
    let b1 = BeliefState::from_hands(&[
        [make_card(8, 0), make_card(8, 1)],   // medium pair (bluff-catcher)
        [make_card(10, 0), make_card(9, 1)],  // weak
    ]);
    let leaf = CheckdownLeafEval::new();

    let solver = SubgameSolver::new(1, 0);
    let resolved = solver.solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 3_000);
    assert!(resolved.info_sets > 0);

    // Rebuild an identical subgame to score exploitability of the strategy.
    let sg = Subgame::new(public_root(river_board(), 20, 3), &[b0, b1], &leaf);
    let expl = exploitability(&sg, &resolved.strategy);
    assert!(expl < 0.05, "resolved river subgame exploitability {expl} bb should be small");
}

#[test]
fn turn_subgame_uses_the_leaf_evaluator() {
    // A turn root with depth 1 ⇒ the tree is cut at the river and scored by
    // the check-down evaluator.  It must still be a well-formed, zero-sum,
    // solvable game.
    let turn_board =
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD];
    let root = public_root(turn_board, 20, 2);
    assert_eq!(root.street, 2, "root should be on the turn");

    let b0 = BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)], [make_card(6, 0), make_card(5, 0)]]);
    let b1 = BeliefState::from_hands(&[[make_card(8, 0), make_card(8, 1)], [make_card(10, 0), make_card(9, 1)]]);
    let leaf = CheckdownLeafEval::new();

    let solver = SubgameSolver::new(1, 0);
    let resolved = solver.solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 1_000);
    assert!(resolved.info_sets > 0, "turn subgame should discover info sets");

    // Strategies are valid distributions.
    for probs in resolved.strategy.values() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
    }
}

// ----- Warm-start, DCFR fallback, comparison, stress -----

/// A clean river root with an arbitrarily inflated pot — the public state the
/// resolver receives after an **off-tree** (e.g. overbet) line on a prior
/// street put `extra_each` extra chips in from each player.  Built by real
/// mechanics, then the (street-start, nobody-owes) pot is scaled up while
/// conserving chips, so it is a valid public state the abstraction never
/// would have produced.
fn river_root_with_extra_pot(board: [u8; 5], stack: u32, extra_each: u32) -> GameState {
    let mut gs = public_root(board, stack, 3);
    for i in 0..2 {
        gs.total_committed[i] += extra_each;
        gs.pot += extra_each;
        gs.stacks[i] -= extra_each;
    }
    gs
}

fn duel_ranges() -> (BeliefState, BeliefState) {
    let b0 = BeliefState::from_hands(&[
        [make_card(12, 1), make_card(12, 2)], // trips (nuts-ish)
        [make_card(6, 0), make_card(5, 0)],   // air
    ]);
    let b1 = BeliefState::from_hands(&[
        [make_card(8, 0), make_card(8, 1)],   // bluff-catcher
        [make_card(10, 0), make_card(9, 1)],  // weak
    ]);
    (b0, b1)
}

#[test]
fn off_tree_overbet_pot_river_resolves_low_exploitability() {
    // The resolver does not need the bet size in any abstraction: it resolves
    // from whatever public state it is handed.  Here a big off-tree pot is
    // already in (a prior overbet line); the river subgame must still resolve
    // near-optimally (exact BR, complete board).
    let (b0, b1) = duel_ranges();
    let leaf = CheckdownLeafEval::new();
    let root = river_root_with_extra_pot(river_board(), 60, 40);
    assert!(root.pot >= 84, "pot should be inflated by the off-tree line: {}", root.pot);

    let resolved = SubgameSolver::new(1, 0).solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 3_000);
    let sg = Subgame::new(river_root_with_extra_pot(river_board(), 60, 40), &[b0, b1], &leaf);
    let expl = exploitability(&sg, &resolved.strategy);
    assert!(expl < 0.05, "off-tree-pot river resolved to {expl} bb, should be small");
}

#[test]
fn dcfr_fallback_resolves_the_subgame() {
    // The multiway fallback path (plan caveat): the same subgame tree solved
    // with DCFR instead of predictive RM⁺.  Validated heads-up (we have no
    // multiway subgame yet) — it must reach a near-optimal strategy too.
    let (b0, b1) = duel_ranges();
    let leaf = CheckdownLeafEval::new();
    let root = public_root(river_board(), 20, 3);

    let resolved = SubgameSolver::new(1, 0)
        .with_solver(SolverKind::Dcfr)
        .solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 4_000);

    let sg = Subgame::new(public_root(river_board(), 20, 3), &[b0, b1], &leaf);
    let expl = exploitability(&sg, &resolved.strategy);
    assert!(expl < 0.05, "DCFR fallback resolved to {expl} bb, should be small");
}

#[test]
fn predictive_matches_or_beats_dcfr_on_the_subgame() {
    // A recorded comparison of predictive vs DCFR
    // subgame solving at an equal budget.  Both reach a good strategy; the
    // predictive (CFR⁺) last iterate should be at least as good as DCFR's
    // average — the reason the resolver defaults to it.
    let (b0, b1) = duel_ranges();
    let leaf = CheckdownLeafEval::new();
    let iters = 2_000;

    let pred = SubgameSolver::new(1, 0)
        .solve_for_iters(&public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf, iters);
    let dcfr = SubgameSolver::new(1, 0)
        .with_solver(SolverKind::Dcfr)
        .solve_for_iters(&public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf, iters);

    let expl_pred = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf), &pred.strategy);
    let expl_dcfr = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &[b0, b1], &leaf), &dcfr.strategy);

    // Recorded comparison (visible with `--nocapture`).
    println!("subgame resolve @ {iters} iters — predictive: {expl_pred:.5} bb, DCFR: {expl_dcfr:.5} bb");
    assert!(expl_pred < 0.05 && expl_dcfr < 0.05, "both solvers should resolve well");
    assert!(
        expl_pred <= expl_dcfr + 1e-3,
        "predictive ({expl_pred}) should be at least as good as DCFR ({expl_dcfr})"
    );
}

#[test]
fn warm_start_speeds_convergence() {
    // Warm-starting from a blueprint (here a converged strategy on the *same*
    // subgame, so the info-set keys match) reaches a far lower exploitability
    // in a handful of iterations than a cold (uniform) start does.
    let (b0, b1) = duel_ranges();
    let leaf = CheckdownLeafEval::new();
    let beliefs = [b0.clone(), b1.clone()];

    // A near-equilibrium "blueprint" for this subgame.
    let blueprint = SubgameSolver::new(1, 0)
        .solve_for_iters(&public_root(river_board(), 20, 3), &beliefs, &leaf, 4_000)
        .strategy;
    let seed = warm_start_regrets(&blueprint, 50.0);

    let few = 3;
    let cold = SubgameSolver::new(1, 0).solve_for_iters(&public_root(river_board(), 20, 3), &beliefs, &leaf, few);
    let warm = SubgameSolver::new(1, 0)
        .with_warm_start(seed)
        .solve_for_iters(&public_root(river_board(), 20, 3), &beliefs, &leaf, few);

    let expl_cold = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &beliefs, &leaf), &cold.strategy);
    let expl_warm = exploitability(&Subgame::new(public_root(river_board(), 20, 3), &beliefs, &leaf), &warm.strategy);
    println!("after {few} iters — cold: {expl_cold:.5} bb, warm-started: {expl_warm:.5} bb");
    assert!(expl_warm < expl_cold, "warm start ({expl_warm}) should beat cold ({expl_cold}) at {few} iters");
}

#[test]
fn flop_subgame_cuts_at_turn_and_resolves() {
    // A flop root (board = 3 cards + two NO_CARD slots): play resolves the
    // flop betting — including off-tree all-in lines — and is cut at the turn,
    // scored by the check-down leaf evaluator.  Must be well-formed: info sets
    // discovered, valid distributions, zero-sum.
    let flop_board =
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), NO_CARD, NO_CARD];
    let root = public_root(flop_board, 20, 1);
    assert_eq!(root.street, 1, "root should be on the flop");

    let b0 = BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)], [make_card(6, 0), make_card(5, 0)]]);
    let b1 = BeliefState::from_hands(&[[make_card(8, 0), make_card(8, 1)], [make_card(10, 0), make_card(9, 1)]]);
    let leaf = CheckdownLeafEval::new();

    let resolved = SubgameSolver::new(1, 0).solve_for_iters(&root, &[b0.clone(), b1.clone()], &leaf, 1_000);
    assert!(resolved.info_sets > 0, "flop subgame should discover info sets");
    for probs in resolved.strategy.values() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
    }

    // Zero-sum at every leaf (terminal or depth-cut).
    let sg = Subgame::new(root, &[b0, b1], &leaf);
    fn walk(g: &Subgame, s: &SubgameNode) {
        if g.is_terminal(s) {
            assert!((g.utility(s, 0) + g.utility(s, 1)).abs() < 1e-9, "zero-sum at leaf");
            return;
        }
        if g.is_chance(s) {
            for (c, _) in g.chance_outcomes(s) {
                walk(g, &c);
            }
        } else {
            for a in 0..g.num_actions(s) {
                walk(g, &g.apply(s, a));
            }
        }
    }
    walk(&sg, &sg.root());
}

// ----- Finding #1: multi-valued leaf states -----

fn turn_board_with_hole_room() -> [u8; 5] {
    // A♣ K♦ 9♥ 4♠ + (river unknown) — a depth-limit cut at the river.
    [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD]
}

#[test]
fn multi_valued_leaf_inserts_a_chooser_node_and_stays_zero_sum() {
    // With K > 1 the depth-limit leaf becomes the opponent's K-way choice
    // node; the tree must still be well-formed and zero-sum everywhere, and
    // a continuation node (K actions, owned by the chooser) must exist.
    let root = public_root(turn_board_with_hole_room(), 20, 2);
    let (b0, b1) = duel_ranges();
    let leaf = crate::validation::resolving::leaf_eval::MultiContinuationLeaf::new();
    let sg = Subgame::new(root.clone(), &[b0, b1], &leaf);
    let chooser = 1 - root.current_player();

    fn walk(g: &Subgame, s: &SubgameNode, chooser: usize, saw_choice: &mut bool) {
        if g.is_terminal(s) {
            assert!((g.utility(s, 0) + g.utility(s, 1)).abs() < 1e-9, "zero-sum at leaf");
            return;
        }
        if g.is_chance(s) {
            for (c, _) in g.chance_outcomes(s) {
                walk(g, &c, chooser, saw_choice);
            }
            return;
        }
        if g.pending_continuation(s) {
            *saw_choice = true;
            assert_eq!(g.current_player(s), chooser, "the opponent chooses the continuation");
            assert_eq!(g.num_actions(s), 4, "one action per continuation");
        }
        for a in 0..g.num_actions(s) {
            walk(g, &g.apply(s, a), chooser, saw_choice);
        }
    }
    let mut saw_choice = false;
    walk(&sg, &sg.root(), chooser, &mut saw_choice);
    assert!(saw_choice, "the subgame must contain at least one continuation-choice node");
}

#[test]
fn multi_continuation_resolve_is_more_robust_than_single() {
    // The depth-limited-solving headline (Brown et al. 2018): a strategy
    // resolved while the opponent may pick among K continuations is less
    // exploitable — measured IN the K-continuation game by exact BR (which
    // may choose continuations adversarially) — than one resolved assuming a
    // single (check-down) continuation.
    let (b0, b1) = duel_ranges();
    let beliefs = [b0, b1];
    let iters = 4_000;
    let root = || public_root(turn_board_with_hole_room(), 20, 2);

    let multi = crate::validation::resolving::leaf_eval::MultiContinuationLeaf::new();
    let single = CheckdownLeafEval::new(); // == multi's continuation 0

    // A: resolved aware of the K = 4 choice.  B: resolved assuming one.
    let a = SubgameSolver::new(1, 0).solve_for_iters(&root(), &beliefs, &multi, iters);
    let b = SubgameSolver::new(1, 0).solve_for_iters(&root(), &beliefs, &single, iters);

    // Both scored in the SAME multi-valued game (the real, robust opponent).
    let game = Subgame::new(root(), &beliefs, &multi);
    let expl_a = exploitability(&game, &a.strategy);
    let expl_b = exploitability(&game, &b.strategy);
    println!(
        "multi-valued-leaf robustness — K=4-resolved: {expl_a:.5} bb, single-resolved: {expl_b:.5} bb"
    );
    assert!(
        expl_a < expl_b,
        "the continuation-aware resolve ({expl_a}) must be less exploitable than the naive one ({expl_b})"
    );
}

#[test]
fn check_raise_line_is_in_the_subgame_tree() {
    // The resolver solves over real betting, so check-raise lines (a common
    // resolving failure mode) are genuinely in the tree, not abstracted away.
    // Confirm a [check, then aggressive] line is reachable for some deal.
    let (b0, b1) = duel_ranges();
    let leaf = CheckdownLeafEval::new();
    let sg = Subgame::new(public_root(river_board(), 40, 3), &[b0, b1], &leaf);

    // From the first deal, the first player checks; the second bets/raises;
    // the first then raises = a check-raise.
    let deal = sg.chance_outcomes(&sg.root())[0].0.clone();
    let acts0 = {
        let gs = deal.gs.as_ref().unwrap();
        poker_core::legal_actions(gs)
    };
    let check_idx = acts0.iter().position(|&a| a == Action::Check).expect("first player can check");
    let after_check = sg.apply(&deal, check_idx);

    let acts1 = {
        let gs = after_check.gs.as_ref().unwrap();
        poker_core::legal_actions(gs)
    };
    let bet_idx = acts1
        .iter()
        .position(|&a| matches!(a, Action::Raise(_)) || a == Action::AllIn)
        .expect("second player can bet into the check");
    let after_bet = sg.apply(&after_check, bet_idx);

    let acts2 = {
        let gs = after_bet.gs.as_ref().unwrap();
        poker_core::legal_actions(gs)
    };
    assert!(
        acts2.iter().any(|&a| matches!(a, Action::Raise(_)) || a == Action::AllIn),
        "the checker can raise over the bet — a check-raise line exists in the subgame"
    );
}
