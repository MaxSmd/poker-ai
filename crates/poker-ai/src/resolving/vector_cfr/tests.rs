//! Tests for the vectorized public-tree solver.
//!
//! The load-bearing ones cross-check against the explicit-deal `Subgame`
//! oracle — the whole reason that slower solver is kept.

use super::keys::MARKER_CONTINUATION;
use super::node::{valid_reach, NodeKind};
use super::*;
use crate::validation::resolving::leaf_eval::CheckdownLeafEval;
use crate::validation::resolving::subgame::Subgame;
use crate::validation::solver::best_response::{best_response_value, exploitability, profile_value};
use poker_core::action::Action;
use poker_core::make_card;
use poker_core::state::MAX_PLAYERS;

fn river_board() -> [u8; 5] {
    // A♣ K♦ 9♥ 4♠ 2♣
    [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)]
}

fn turn_board() -> [u8; 5] {
    // A♣ K♦ 9♥ 4♠ + (river undealt)
    [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD]
}

fn flop_board() -> [u8; 5] {
    // A♣ K♦ 9♥ + (turn, river undealt)
    [make_card(12, 0), make_card(11, 1), make_card(7, 2), NO_CARD, NO_CARD]
}

/// A clean heads-up public root reached by checking/calling to `target_street`
/// (no extra money); holes are placeholders overwritten per hand.
fn public_root_at(board: [u8; 5], stack: u32, target_street: u8) -> GameState {
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
        let act = if acts.contains(&Action::Check) { Action::Check } else { Action::Call };
        gs.apply_action(act);
    }
    gs
}

/// A clean heads-up river public root (check/call to the river).
fn public_root(board: [u8; 5], stack: u32) -> GameState {
    public_root_at(board, stack, 3)
}

fn duel_ranges() -> [BeliefState; 2] {
    [
        BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)], // trips (nuts-ish)
            [make_card(6, 0), make_card(5, 0)],   // air
        ]),
        BeliefState::from_hands(&[
            [make_card(8, 0), make_card(8, 1)],   // bluff-catcher
            [make_card(10, 0), make_card(9, 1)],  // weak
        ]),
    ]
}

#[test]
fn vectorized_resolve_agrees_with_explicit_oracle() {
    // The headline #2 cross-check: the vectorized public-tree solver and the
    // explicit-deal Subgame solve the SAME river game.  The vectorized
    // strategy, scored inside the explicit oracle by exact best response,
    // must reach the same near-optimal exploitability.
    let beliefs = duel_ranges();
    let resolved = solve_vectorized(&public_root(river_board(), 20), &beliefs, 1_200);
    assert!(resolved.info_sets > 0, "must emit strategy");

    let leaf = CheckdownLeafEval::new(); // unused on a complete board
    let oracle = Subgame::new(public_root(river_board(), 20), &beliefs, &leaf);
    let expl = exploitability(&oracle, &resolved.strategy);
    println!("vectorized river resolve exploitability (in the explicit oracle): {expl:.5} bb");
    assert!(expl < 0.05, "vectorized resolve should be near-optimal in the oracle game, got {expl}");
}

#[test]
fn vectorized_turn_resolve_agrees_with_explicit_oracle() {
    // Turn resolving: the vectorized public-tree solver cuts at the undealt
    // river and scores each turn leaf by the runout-averaged check-down
    // showdown (`RunoutShowdown`).  The explicit-deal `Subgame` with the
    // `CheckdownLeafEval` cuts the SAME tree at the river with the SAME
    // check-down value, so the vectorized strategy, scored by exact best
    // response inside that oracle, must reach the same near-optimal
    // exploitability — exactly the river cross-check, one street earlier.
    let beliefs = duel_ranges();
    let root = public_root_at(turn_board(), 20, 2);
    assert_eq!(root.street, 2, "root should be on the turn");

    let resolved = solve_vectorized(&root, &beliefs, 1_500);
    assert!(resolved.info_sets > 0, "must emit strategy");

    let leaf = CheckdownLeafEval::new();
    let oracle = Subgame::new(public_root_at(turn_board(), 20, 2), &beliefs, &leaf);
    let expl = exploitability(&oracle, &resolved.strategy);
    println!("vectorized turn resolve exploitability (in the explicit oracle): {expl:.5} bb");
    assert!(expl < 0.05, "vectorized turn resolve should be near-optimal in the oracle game, got {expl}");
}

#[test]
fn k_continuation_inserts_a_chooser_node_owned_by_the_opponent() {
    // Fast structural guard for the finding-#1 wiring (the exploitability
    // cross-check below proves the semantics but is minutes-slow): a K > 1
    // turn resolve must insert, at its depth-limit leaf, a continuation-choice
    // Decision owned by the chooser with one `RunoutShowdown` child per scale
    // at a non-decreasing (inflated) pot — and a K = 1 resolve must not.
    let beliefs = duel_ranges();
    let root = public_root_at(turn_board(), 20, 2);
    let chooser = 1 - root.current_player();
    let scales = vec![0.0, 0.75, 1.5, 3.0];

    let solver = VectorCfr::new_capped_multi(&root, &beliefs, u32::MAX, scales.clone());
    let mut choosers = 0;
    for k in &solver.kinds {
        let NodeKind::Decision { player, children, marker: MARKER_CONTINUATION, .. } = k else {
            continue;
        };
        choosers += 1;
        assert_eq!(*player, chooser, "the opponent chooses the continuation");
        assert_eq!(children.len(), scales.len(), "one action per continuation");
        let pots: Vec<f64> = children
            .iter()
            .map(|&c| match solver.kinds[c] {
                NodeKind::RunoutShowdown { half_pot } => half_pot,
                _ => panic!("a continuation child must be a runout showdown"),
            })
            .collect();
        for w in pots.windows(2) {
            assert!(w[1] > w[0], "each later continuation inflates the pot: {pots:?}");
        }
    }
    assert!(choosers > 0, "a K > 1 turn resolve must contain a continuation-choice node");

    // K = 1 stays a plain leaf — no chooser nodes.
    let single = VectorCfr::new(&root, &beliefs);
    assert!(
        !single.kinds.iter().any(|k| matches!(k, NodeKind::Decision { marker: MARKER_CONTINUATION, .. })),
        "a single-continuation resolve must not insert chooser nodes"
    );

    // The emitted strategy (chooser nodes included) is valid — a handful of
    // iterations suffices, the strategy-sum normalizes at any count.
    let resolved = solve_vectorized_multi(&root, &beliefs, 20, u32::MAX, scales);
    for probs in resolved.strategy.values() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
    }
}

#[test]
#[ignore = "K-aware turn resolve + two exact-BR passes over the multi-valued \
            oracle is minutes-slow; k_continuation_inserts_a_chooser_node guards the wiring"]
fn vectorized_multi_continuation_is_more_robust_than_single() {
    // Finding #1, vectorized: a turn resolve that lets the opponent pick among
    // K continuations at the depth-limit leaf is less exploitable — measured
    // IN the explicit K-continuation oracle by exact BR (which may choose
    // continuations adversarially) — than one resolved assuming a single
    // check-down.  This is the depth-limited-solving headline, and it also
    // proves the vectorized chooser nodes key-match the oracle's (else the K=4
    // resolve's continuation policy would be ignored and buy nothing).
    use crate::validation::resolving::leaf_eval::MultiContinuationLeaf;
    let beliefs = duel_ranges();
    let scales = vec![0.0, 0.75, 1.5, 3.0]; // == MultiContinuationLeaf default
    let root = || public_root_at(turn_board(), 20, 2);

    // A: resolved aware of the K = 4 choice.  B: resolved assuming one.
    let a = solve_vectorized_multi(&root(), &beliefs, 2_000, u32::MAX, scales.clone());
    let b = solve_vectorized(&root(), &beliefs, 2_000);

    // Both scored in the SAME multi-valued oracle (the adapting opponent).
    let leaf = MultiContinuationLeaf::with_scales(scales);
    let game = Subgame::new(root(), &beliefs, &leaf);
    let expl_a = exploitability(&game, &a.strategy);
    let expl_b = exploitability(&game, &b.strategy);
    println!(
        "vectorized multi-valued-leaf robustness — K=4-resolved: {expl_a:.5} bb, single-resolved: {expl_b:.5} bb"
    );
    assert!(
        expl_a < expl_b,
        "the continuation-aware resolve ({expl_a}) must be less exploitable than the naive one ({expl_b})"
    );
}

#[test]
#[ignore = "flop's two-card runout + exact-BR oracle is minutes-slow; \
            the 990-divisor is guarded fast by flop_runout_cfvs_matches_hand_vs_hand_equity"]
fn vectorized_flop_resolve_agrees_with_explicit_oracle() {
    // Flop resolving: the vectorized solver cuts at the undealt turn and
    // scores each flop leaf by the two-card runout average (`RunoutShowdown`
    // over C(45,2)=990 turn+river completions).  The explicit-deal `Subgame`
    // with `CheckdownLeafEval` cuts the SAME tree at the turn with the SAME
    // check-down-over-runout value, so the vectorized strategy scored by exact
    // best response in that oracle must be near-optimal.  A small stack keeps
    // the uncapped betting tree (and thus the count of expensive runout
    // leaves) modest so the two-card runout stays affordable in a unit test.
    let beliefs = duel_ranges();
    let root = public_root_at(flop_board(), 6, 1);
    assert_eq!(root.street, 1, "root should be on the flop");

    let resolved = solve_vectorized(&root, &beliefs, 600);
    assert!(resolved.info_sets > 0, "must emit strategy");

    let leaf = CheckdownLeafEval::new();
    let oracle = Subgame::new(public_root_at(flop_board(), 6, 1), &beliefs, &leaf);
    let expl = exploitability(&oracle, &resolved.strategy);
    println!("vectorized flop resolve exploitability (in the explicit oracle): {expl:.5} bb");
    assert!(expl < 0.05, "vectorized flop resolve should be near-optimal in the oracle game, got {expl}");
}

#[test]
#[ignore = "throughput demonstration; run with --ignored"]
fn turn_full_range_solve_is_fast() {
    // The runout table is built once and shared, so a full-range turn resolve
    // (both ranges the whole 1081-combo grid) solves at a play-viable rate —
    // NOT the per-iteration evaluate+sort the naive runout would cost.
    use std::time::Instant;
    let mut b0 = BeliefState::uniform();
    let mut b1 = BeliefState::uniform();
    b0.remove_board(&turn_board());
    b1.remove_board(&turn_board());

    let root = public_root_at(turn_board(), 20, 2);
    let build = Instant::now();
    let mut solver = VectorCfr::new(&root, &[b0, b1]);
    let build_ms = build.elapsed().as_millis();
    let solve = Instant::now();
    solver.run(500);
    let solve_ms = solve.elapsed().as_millis();
    let resolved = solver.into_resolved();
    println!(
        "turn full-range resolve: {} public nodes, {} info sets — build {build_ms} ms, 500 iters {solve_ms} ms",
        resolved.public_nodes, resolved.info_sets
    );
    assert!(resolved.info_sets > 1000, "full ranges yield many per-hand info sets");
}

#[test]
fn single_hand_each_is_solved() {
    // One hand per player ⇒ no reach mixing; a clean check of the core
    // recursion against the explicit oracle (which solves this trivially).
    let beliefs = [
        BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)]]), // trips
        BeliefState::from_hands(&[[make_card(8, 0), make_card(8, 1)]]),   // pair
    ];
    let resolved = solve_vectorized(&public_root(river_board(), 20), &beliefs, 1_000);
    let leaf = CheckdownLeafEval::new();
    let oracle = Subgame::new(public_root(river_board(), 20), &beliefs, &leaf);
    let expl = exploitability(&oracle, &resolved.strategy);
    assert!(expl < 0.05, "single-hand-each should solve cleanly, got {expl}");
}

#[test]
#[ignore = "throughput demonstration over full 1326-combo ranges; run with --ignored"]
fn vectorized_solves_full_ranges_the_explicit_solver_cannot() {
    // Throughput deliverable: full uniform ranges = ~1081×1081 ≈ 1.1 M deals,
    // which the explicit-deal Subgame cannot enumerate, are solved by walking
    // a tiny PUBLIC tree once with a per-hand value vector.  The public node
    // count is independent of range breadth — the whole point.
    use std::time::Instant;
    let mut b0 = BeliefState::uniform();
    let mut b1 = BeliefState::uniform();
    b0.remove_board(&river_board());
    b1.remove_board(&river_board());

    let start = Instant::now();
    let resolved = solve_vectorized(&public_root(river_board(), 20), &[b0, b1], 200);
    let elapsed = start.elapsed();
    println!(
        "vectorized full-range river: {} public nodes, {} info sets, 200 iters in {:?}",
        resolved.public_nodes, resolved.info_sets, elapsed
    );
    // Tiny public tree, but thousands of per-hand info sets emitted.
    assert!(resolved.public_nodes < 100, "betting tree is small regardless of range breadth");
    assert!(resolved.info_sets > 1000, "full ranges yield many per-hand info sets");
}

#[test]
fn raise_cap_bounds_the_public_tree() {
    // Deep-ish stacks with a small pot: the raise chain is what the cap
    // prunes.  The capped tree must be strictly smaller, still solve to
    // valid distributions, and a generous cap must reproduce the uncapped
    // tree exactly.
    let beliefs = duel_ranges();
    let root = public_root(river_board(), 200);
    let capped = {
        let mut s = VectorCfr::new_capped(&root, &beliefs, 1);
        s.run(200);
        s.into_resolved()
    };
    let uncapped = solve_vectorized(&root, &beliefs, 200);
    assert!(
        capped.public_nodes < uncapped.public_nodes,
        "cap-1 tree ({}) must be smaller than uncapped ({})",
        capped.public_nodes,
        uncapped.public_nodes
    );
    for probs in capped.strategy.values() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
    let generous = VectorCfr::new_capped(&root, &beliefs, 1_000);
    assert_eq!(
        generous.kinds.len(),
        VectorCfr::new(&root, &beliefs).kinds.len(),
        "a cap the tree never reaches changes nothing"
    );
    // The root menu helper is index-aligned with the built tree's root.
    let acts = capped_root_actions(&root, 1);
    let NodeKind::Decision { children, .. } = &VectorCfr::new_capped(&root, &beliefs, 1).kinds
        [VectorCfr::new_capped(&root, &beliefs, 1).root]
    else {
        panic!("root is a decision node");
    };
    assert_eq!(acts.len(), children.len(), "root menu width matches the tree");
}

#[test]
fn strategies_are_valid_distributions() {
    let beliefs = duel_ranges();
    let resolved = solve_vectorized(&public_root(river_board(), 20), &beliefs, 500);
    for probs in resolved.strategy.values() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
        assert!(probs.iter().all(|&p| p >= 0.0), "no negative probabilities");
    }
}

// ------------------------------------------------------------------
// Full-river turn resolving (real river betting inside the subgame).
// ------------------------------------------------------------------

/// Exact best response in the TRUE turn+river game (real river betting,
/// no leaf model), over small explicit supports — the independent gate
/// for the full-river mode.  Support-vector BR: per-BR-hand values, the
/// profile seat's reach weighted by the resolved strategy (uniform where
/// it stored nothing), explicit per-pair collision skips at terminals,
/// direct 7-card rank comparison at showdowns, per-pair river divisor.
struct TrueBr<'a> {
    strategy: &'a HashMap<u64, Vec<f64>>,
    cap: u32,
    br: usize,
    hands: [&'a [[u8; 2]]; 2],
    big_blind: f64,
}

impl TrueBr<'_> {
    /// `(br₀ + br₁)/2` in bb — exploitability of `strategy` in the true
    /// game, deals uniform over non-colliding support pairs.
    fn exploitability(
        root: &GameState,
        hands: [&[[u8; 2]]; 2],
        strategy: &HashMap<u64, Vec<f64>>,
        cap: u32,
    ) -> f64 {
        let mut sum = 0.0;
        for br in 0..2 {
            let me = TrueBr { strategy, cap, br, hands, big_blind: root.big_blind as f64 };
            let opp = 1 - br;
            let reach = vec![1.0f64; hands[opp].len()];
            let v = me.node(&mut root.clone(), &mut Vec::new(), 0, &reach);
            // Normalize by the number of consistent (h, j) pairs.
            let pairs: usize = hands[br]
                .iter()
                .map(|h| {
                    hands[opp].iter().filter(|j| !collide(h, j, &root.board)).count()
                })
                .sum();
            sum += v.iter().sum::<f64>() / pairs as f64 / me.big_blind;
        }
        sum / 2.0
    }

    fn node(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u32, reach: &[f64]) -> Vec<f64> {
        let acts = test_capped(gs, raises, self.cap);
        let actor = gs.current_player();
        let n = acts.len();
        if actor == self.br {
            let mut out = vec![f64::NEG_INFINITY; self.hands[self.br].len()];
            for (i, &a) in acts.iter().enumerate() {
                let child = self.descend(gs, hist, raises, reach, a, i);
                for (o, c) in out.iter_mut().zip(&child) {
                    *o = o.max(*c);
                }
            }
            out
        } else {
            let mut out = vec![0.0f64; self.hands[self.br].len()];
            for (i, &a) in acts.iter().enumerate() {
                let mut child_reach = vec![0.0f64; reach.len()];
                for (j, cr) in child_reach.iter_mut().enumerate() {
                    if reach[j] == 0.0 {
                        continue;
                    }
                    let mut hole = self.hands[actor][j];
                    hole.sort_unstable();
                    let key = subgame_info_key(actor, hole, &gs.board, hist);
                    let sigma = match self.strategy.get(&key) {
                        Some(p) if p.len() == n => p[i],
                        _ => 1.0 / n as f64,
                    };
                    *cr = reach[j] * sigma;
                }
                let child = self.descend(gs, hist, raises, &child_reach, a, i);
                for (o, c) in out.iter_mut().zip(&child) {
                    *o += c;
                }
            }
            out
        }
    }

    fn descend(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u32, reach: &[f64], act: Action, i: usize) -> Vec<f64> {
        let (old_street, old_bet) = (gs.street, gs.current_bet);
        gs.apply_action(act);
        hist.push(i as u8);
        let new_raises = if gs.street != old_street {
            0
        } else if gs.current_bet > old_bet {
            raises + 1
        } else {
            raises
        };
        let undealt = gs.board[..gs.board_cards_count()].contains(&NO_CARD);
        let out = if gs.is_terminal() {
            if gs.folded != 0 {
                self.fold_value(gs, reach)
            } else if undealt {
                self.deal_river(gs, hist, new_raises, reach)
            } else {
                self.showdown(gs, reach)
            }
        } else if undealt {
            self.deal_river(gs, hist, new_raises, reach)
        } else {
            self.node(gs, hist, new_raises, reach)
        };
        hist.pop();
        gs.undo_action();
        out
    }

    fn deal_river(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u32, reach: &[f64]) -> Vec<f64> {
        let opp = 1 - self.br;
        let mut used = 0u64;
        for &c in &gs.board[..4] {
            used |= 1 << c;
        }
        let mut out = vec![0.0f64; self.hands[self.br].len()];
        for c in 0..52u8 {
            if used & (1 << c) != 0 {
                continue;
            }
            gs.board[4] = c;
            let mut child_reach = reach.to_vec();
            for (j, cr) in child_reach.iter_mut().enumerate() {
                let h = self.hands[opp][j];
                if h[0] == c || h[1] == c {
                    *cr = 0.0;
                }
            }
            let child = if gs.is_terminal() {
                self.showdown(gs, &child_reach)
            } else {
                self.node(gs, hist, raises, &child_reach)
            };
            for (h, (o, cv)) in out.iter_mut().zip(&child).enumerate() {
                let hb = self.hands[self.br][h];
                if hb[0] != c && hb[1] != c {
                    *o += cv;
                }
            }
        }
        gs.board[4] = NO_CARD;
        for o in &mut out {
            *o /= 44.0;
        }
        out
    }

    fn fold_value(&self, gs: &GameState, reach: &[f64]) -> Vec<f64> {
        let folder = if gs.folded & 1 != 0 { 0usize } else { 1 };
        let sign = if folder == self.br { -1.0 } else { 1.0 };
        let amount = gs.total_committed[folder] as f64;
        let opp = 1 - self.br;
        self.hands[self.br]
            .iter()
            .map(|h| {
                let mut v = 0.0;
                for (j, &r) in reach.iter().enumerate() {
                    if r != 0.0 && !collide(h, &self.hands[opp][j], &gs.board) {
                        v += sign * amount * r;
                    }
                }
                v
            })
            .collect()
    }

    fn showdown(&self, gs: &GameState, reach: &[f64]) -> Vec<f64> {
        use poker_core::lut_eval::evaluate_7_lut;
        let matched = gs.total_committed[0].min(gs.total_committed[1]) as f64;
        let b = &gs.board;
        let opp = 1 - self.br;
        self.hands[self.br]
            .iter()
            .map(|h| {
                let hr = evaluate_7_lut(&[h[0], h[1], b[0], b[1], b[2], b[3], b[4]]);
                let mut v = 0.0;
                for (j, &r) in reach.iter().enumerate() {
                    let jh = self.hands[opp][j];
                    if r == 0.0 || collide(h, &jh, b) {
                        continue;
                    }
                    let jr = evaluate_7_lut(&[jh[0], jh[1], b[0], b[1], b[2], b[3], b[4]]);
                    v += r * matched
                        * if hr > jr {
                            1.0
                        } else if hr < jr {
                            -1.0
                        } else {
                            0.0
                        };
                }
                v
            })
            .collect()
    }
}

/// Two holdings collide with each other or the visible board.
fn collide(h: &[u8; 2], j: &[u8; 2], board: &[u8; 5]) -> bool {
    let mut used = 0u64;
    for &c in board {
        if c != NO_CARD {
            used |= 1 << c;
        }
    }
    for &c in h {
        if used & (1 << c) != 0 {
            return true;
        }
        used |= 1 << c;
    }
    j.iter().any(|&c| used & (1 << c) != 0)
}

/// Test-local mirror of the solver's raise-cap filter (per-street reset).
fn test_capped(gs: &GameState, raises: u32, cap: u32) -> Vec<Action> {
    let full = legal_actions(gs);
    if raises < cap {
        return full.to_vec();
    }
    let has_passive = full.iter().any(|a| matches!(a, Action::Check | Action::Call));
    full.iter()
        .copied()
        .filter(|a| !(matches!(a, Action::Raise(_)) || (matches!(a, Action::AllIn) && has_passive)))
        .collect()
}

/// Richer-than-duel supports on the turn board (sets, pairs, draws, air)
/// so the river betting actually matters.
fn turn_supports() -> ([[u8; 2]; 3], [[u8; 2]; 3]) {
    (
        [
            [make_card(12, 1), make_card(12, 2)], // A♦A♥: top set
            [make_card(11, 2), make_card(10, 2)], // K♥Q♥: top-ish pair
            [make_card(4, 0), make_card(3, 0)],   // 6♣5♣: air
        ],
        [
            [make_card(6, 0), make_card(6, 1)],   // 8♣8♦: bluff-catcher
            [make_card(12, 3), make_card(2, 0)],  // A♠4♣: two pair
            [make_card(9, 0), make_card(8, 1)],   // J♣T♦: gutshot air
        ],
    )
}

/// The headline gate for full-river turn resolving: solved WITH the real
/// river betting, the strategy is near-equilibrium in the TRUE turn+river
/// game (measured by the independent support-vector exact BR above).  The
/// leaf-cut resolves cannot even express river play (their keys stop at
/// the reveal → uniform river in the true game), which is exactly the gap
/// this mode closes — their true-game exploitability must be clearly worse.
#[test]
fn full_river_turn_resolve_is_near_equilibrium_in_the_true_game() {
    let (s0, s1) = turn_supports();
    let beliefs = [BeliefState::from_hands(&s0), BeliefState::from_hands(&s1)];
    let root = public_root_at(turn_board(), 16, 2);
    let cap = 1;

    // 150 iterations already lands well under the bound (600 → 0.0014 bb,
    // bound 0.05 — huge slack); the cut arm only needs to be *worse*,
    // which it is by three orders of magnitude (~1.4 bb: it cannot
    // express river play at all).
    let full = solve_vectorized_full_river(&root, &beliefs, 150, cap);
    let cut = solve_vectorized_capped(&root, &beliefs, 150, cap);
    assert!(
        full.public_nodes > cut.public_nodes,
        "full-river tree must contain the river betting ({} vs {})",
        full.public_nodes,
        cut.public_nodes
    );

    let expl_full = TrueBr::exploitability(&root, [&s0, &s1], &full.strategy, cap);
    let expl_cut = TrueBr::exploitability(&root, [&s0, &s1], &cut.strategy, cap);
    println!("true-game exploitability: full-river {expl_full:.4} bb, leaf-cut {expl_cut:.4} bb");
    assert!(
        expl_full < 0.05,
        "full-river resolve should be near-optimal in the true game, got {expl_full}"
    );
    assert!(
        expl_full < expl_cut,
        "solving the real river betting must beat the leaf cut in the true game \
         ({expl_full} vs {expl_cut})"
    );
}

// ------------------------------------------------------------------
// Continual re-solving: the vectorized CFV gadget.
// ------------------------------------------------------------------

/// The safety + no-distortion gate for the vectorized gadget (mirrors the
/// explicit `gadget.rs`/`continual.rs` tests):
///
/// 1. extracted CFVs are consistent — their reach-weighted mean equals the
///    profile's value for the opponent in the explicit oracle;
/// 2. re-solving the same spot constrained by those CFVs leaves the
///    opponent's exact best response vs our deployed strategy no better
///    than it was against the bootstrap (re-entry stays safe);
/// 3. our own deployed strategy stays near-optimal (feeding
///    near-equilibrium CFVs does not distort the resolve).
#[test]
fn gadget_resolve_is_safe_and_true_cfvs_do_not_distort() {
    let beliefs = duel_ranges();
    let root = public_root(river_board(), 20);

    let mut boot = VectorCfr::new(&root, &beliefs);
    boot.run(1_000);
    let cfvs = boot.opponent_cfvs();
    let me = root.current_player();
    let opp = 1 - me;
    let strat_a = boot.into_resolved().strategy;

    // (1) CFV consistency against the explicit oracle's profile value.
    let leaf = CheckdownLeafEval::new();
    let oracle = Subgame::new(root.clone(), &beliefs, &leaf);
    let opp_reach = if opp == 0 { &beliefs[0] } else { &beliefs[1] };
    let mut prior = [0.0f64; NUM_COMBOS];
    for (i, p) in prior.iter_mut().enumerate() {
        let [a, b] = combo_cards(i);
        *p = opp_reach.prob(a, b);
    }
    let me_reach = if me == 0 { &beliefs[0] } else { &beliefs[1] };
    let mut me_prior = [0.0f64; NUM_COMBOS];
    for (i, p) in me_prior.iter_mut().enumerate() {
        let [a, b] = combo_cards(i);
        *p = me_reach.prob(a, b);
    }
    let mass = valid_reach(&root.board, &me_prior);
    let (mut num, mut den) = (0.0, 0.0);
    for j in 0..NUM_COMBOS {
        let w = prior[j] * mass[j];
        num += w * cfvs[j];
        den += w;
    }
    let mean_cfv = num / den;
    let pv = profile_value(&oracle, &strat_a, opp);
    assert!(
        (mean_cfv - pv).abs() < 0.02,
        "reach-weighted mean CFV {mean_cfv:.4} should match the oracle profile value {pv:.4}"
    );

    // (2)+(3) Gadget re-solve constrained by the carried values.
    let mut gadget = VectorCfr::new(&root, &beliefs).with_opponent_gadget(cfvs);
    gadget.run(1_000);
    let strat_b = gadget.into_resolved().strategy;

    // Safety AND no-distortion are both measured by the opponent's exact
    // BR against OUR deployed strategy (the Step-27 lesson: full-profile
    // exploitability is spuriously high here, because the opponent's own
    // emitted betting strategy is untrained on hands the gadget would
    // Terminate — junk we never deploy).  No better than the bootstrap =
    // safe; no more than ε worse = the near-equilibrium CFVs did not
    // distort our side of the resolve.
    let br_vs_boot = best_response_value(&oracle, opp, &strat_a);
    let br_vs_gadget = best_response_value(&oracle, opp, &strat_b);
    assert!(
        br_vs_gadget <= br_vs_boot + 0.02,
        "opponent BR vs the gadget re-solve ({br_vs_gadget:.4}) must not beat \
         its value vs the bootstrap ({br_vs_boot:.4})"
    );
    assert!(
        br_vs_gadget >= br_vs_boot - 0.05,
        "gadget resolve distorted our strategy: opp BR collapsed from \
         {br_vs_boot:.4} to {br_vs_gadget:.4} (untrained lines would show here)"
    );
}

/// Full-river resolves must emit valid distributions on river betting
/// nodes too (keys distinguished by each node's own board card).
#[test]
fn full_river_strategies_are_valid_distributions() {
    let (s0, s1) = turn_supports();
    let beliefs = [BeliefState::from_hands(&s0), BeliefState::from_hands(&s1)];
    let root = public_root_at(turn_board(), 20, 2);
    let resolved = solve_vectorized_full_river(&root, &beliefs, 50, 1);
    assert!(resolved.info_sets > 0);
    for probs in resolved.strategy.values() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
        assert!(probs.iter().all(|&p| p >= 0.0), "no negative probabilities");
    }
}
