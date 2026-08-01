//! Tests for the equity features and their sweeps.
//!
//! The load-bearing ones prove the fast sweeps equal the slow per-hand
//! primitives exactly (`board_equities` vs `river_equity` for all 1081 holes,
//! `board_ochs` vs an O(n²) per-cluster oracle) — that equality is what lets
//! the offline build use the sweeps.

use super::*;
use crate::abstraction::canonical::preflop_index;
use poker_core::{evaluate_7_lut, make_card};

// A dry, uncoordinated board for clean tests: A♣ K♦ 9♥ 4♠ 2♣.
fn dry_board() -> [u8; 5] {
    [
        make_card(12, 0),
        make_card(11, 1),
        make_card(7, 2),
        make_card(2, 3),
        make_card(0, 0),
    ]
}

/// Against a single opponent hand, the runout-averaged showdown CFV must
/// equal `half_pot·(2·equity − 1)` with the equity from the independent
/// enumerator `hand_vs_hand_equity` — a direct check of the runout
/// normalization (44 on the turn, 990 on the flop).
fn check_runout_matches_equity(board: [u8; 5], real: usize, g: [u8; 2]) {
    let visible: Vec<u8> = board[..real].to_vec();
    let half_pot = 7.0;
    let mut reach = [0.0_f64; 1326];
    reach[combo_index(g[0], g[1])] = 1.0;
    let mut cfv = [0.0_f64; 1326];
    board_runout_cfvs(board, &reach, half_pot, &mut cfv);

    let used: u64 = board[..real].iter().chain(g.iter()).fold(0, |m, &c| m | (1 << c));
    for a in 0..52u8 {
        for b in (a + 1)..52u8 {
            if used & (1 << a) != 0 || used & (1 << b) != 0 {
                assert_eq!(cfv[combo_index(a, b)], 0.0, "blocked hero combo is zero");
                continue;
            }
            let eq = hand_vs_hand_equity([a, b], g, &visible);
            let expected = half_pot * (2.0 * eq - 1.0);
            assert!(
                (cfv[combo_index(a, b)] - expected).abs() < 1e-9,
                "hero {a},{b}: {} vs {expected}",
                cfv[combo_index(a, b)]
            );
        }
    }
}

#[test]
fn turn_runout_cfvs_matches_hand_vs_hand_equity() {
    // One-card (river) runout: divisor 44.
    use poker_core::state::NO_CARD;
    let turn = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD];
    check_runout_matches_equity(turn, 4, [make_card(11, 0), make_card(11, 2)]);
}

#[test]
fn flop_runout_cfvs_matches_hand_vs_hand_equity() {
    // Two-card (turn+river) runout: divisor C(45, 2) = 990.
    use poker_core::state::NO_CARD;
    let flop = [make_card(12, 0), make_card(11, 1), make_card(7, 2), NO_CARD, NO_CARD];
    check_runout_matches_equity(flop, 3, [make_card(11, 0), make_card(11, 2)]);
}

#[test]
fn board_cfvs_uniform_reach_matches_board_equities() {
    // With every opponent at reach 1, the reach-weighted showdown sweep is
    // the unit-count sweep: out[h] = half_pot · 990 · (2·equity − 1).
    let board = dry_board();
    let mut eq = [f32::NAN; 1326];
    board_equities(board, &mut eq);
    let reach = [1.0_f64; 1326];
    let half_pot = 7.0;
    let mut cfv = [0.0_f64; 1326];
    board_cfvs(board, &reach, half_pot, &mut cfv);

    for h in 0..1326 {
        if eq[h].is_nan() {
            assert_eq!(cfv[h], 0.0, "blocked hero combo is zero");
            continue;
        }
        let expected = half_pot * 990.0 * (2.0 * eq[h] as f64 - 1.0);
        // board_equities stores f32, so allow its rounding (≈ 7e-4 at this
        // scale); board_cfvs itself is exact f64.
        assert!((cfv[h] - expected).abs() < 1e-2, "combo {h}: {} vs {expected}", cfv[h]);
    }
}

#[test]
fn board_cfvs_one_hot_reach_matches_hand_vs_hand() {
    // A single opponent hand at reach 1: the hero's value is ±half_pot (win /
    // lose) or 0 (tie) — i.e. half_pot·(2·hand_vs_hand_equity − 1).
    let board = dry_board();
    let opp = [make_card(5, 1), make_card(3, 1)]; // some specific hand
    let mut reach = [0.0_f64; 1326];
    reach[combo_index(opp[0], opp[1])] = 1.0;
    let half_pot = 5.0;
    let mut cfv = [0.0_f64; 1326];
    board_cfvs(board, &reach, half_pot, &mut cfv);

    let hero = [make_card(12, 1), make_card(12, 2)]; // trip aces — beats opp
    let e = hand_vs_hand_equity(hero, opp, &board);
    let expected = half_pot * (2.0 * e - 1.0);
    assert!(
        (cfv[combo_index(hero[0], hero[1])] - expected).abs() < 1e-9,
        "one-hot reach must equal hand-vs-hand"
    );
    // A hero sharing a card with the only opponent has no valid showdown ⇒ 0.
    let blocker = [make_card(5, 1), make_card(9, 0)];
    assert_eq!(cfv[combo_index(blocker[0], blocker[1])], 0.0, "blocker ⇒ no opponent ⇒ 0");
}

#[test]
fn equity_in_unit_interval() {
    let board = dry_board();
    let hole = [make_card(12, 1), make_card(12, 2)]; // pair of aces (with board A) → trips
    let e = river_equity(hole, board);
    assert!((0.0..=1.0).contains(&e), "equity {e} out of range");
}

#[test]
fn nut_hand_has_full_equity() {
    // Board T♠ J♠ Q♠ K♠ 2♥ — hero holds A♠ for a royal flush; nothing beats it.
    let board = [
        make_card(8, 3),
        make_card(9, 3),
        make_card(10, 3),
        make_card(11, 3),
        make_card(0, 2),
    ];
    let hole = [make_card(12, 3), make_card(3, 1)]; // A♠ + junk
    let e = river_equity(hole, board);
    assert!((e - 1.0).abs() < 1e-9, "royal flush equity {e} should be 1.0");
}

#[test]
fn stronger_hand_has_more_equity() {
    let board = dry_board();
    let trips = [make_card(12, 1), make_card(12, 2)]; // trip aces
    let weak = [make_card(5, 1), make_card(3, 2)]; // no pair, low cards
    assert!(river_equity(trips, board) > river_equity(weak, board));
}

#[test]
fn mean_equity_over_all_hands_is_one_half() {
    // Exact zero-sum invariant: averaged over every possible hole-card hand
    // on a fixed board, equity vs a random opponent is exactly 0.5.
    let board = dry_board();
    let mut used = 0u64;
    for &c in &board {
        used |= 1 << c;
    }
    let deck: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
    let mut sum = 0.0;
    let mut n = 0u64;
    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            sum += river_equity([deck[i], deck[j]], board);
            n += 1;
        }
    }
    let mean = sum / n as f64;
    assert!((mean - 0.5).abs() < 1e-9, "mean equity {mean} should be exactly 0.5");
}

#[test]
fn histogram_is_a_distribution() {
    // Turn board (4 cards) → 46 completions binned into 20 buckets.
    let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3)];
    let hole = [make_card(10, 0), make_card(10, 1)];
    let hist = ehs_histogram(&hole, &board, 20);
    assert_eq!(hist.len(), 20);
    let sum: f64 = hist.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "histogram should sum to 1, got {sum}");
    assert!(hist.iter().all(|&h| h >= 0.0));
}

// A coordinated, flushy board to exercise ties and flushes in the sweep.
fn wet_board() -> [u8; 5] {
    [
        make_card(10, 0), // Tc
        make_card(9, 0),  // 9c
        make_card(8, 0),  // 8c
        make_card(3, 1),  // 5d
        make_card(2, 2),  // 4h
    ]
}

#[test]
fn board_equities_match_river_equity_for_every_hole() {
    // The sweep must equal the O(n²) oracle exactly (same integer counts ⇒
    // bit-identical f32) for every one of the 1081 holes.
    for board in [dry_board(), wet_board()] {
        let mut out = [f32::NAN; 1326];
        board_equities(board, &mut out);
        let mut used = 0u64;
        for &c in &board {
            used |= 1 << c;
        }
        let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
        for i in 0..live.len() {
            for j in (i + 1)..live.len() {
                let (a, b) = (live[i], live[j]);
                let want = river_equity([a, b], board) as f32;
                assert_eq!(out[combo_index(a, b)], want, "sweep ≠ oracle for {a},{b}");
            }
        }
    }
}

#[test]
fn board_equities_mean_is_one_half() {
    let board = dry_board();
    let mut out = [f32::NAN; 1326];
    board_equities(board, &mut out);
    let vals: Vec<f64> = out.iter().filter(|e| !e.is_nan()).map(|&e| e as f64).collect();
    assert_eq!(vals.len(), 1081);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    assert!((mean - 0.5).abs() < 1e-6, "mean equity {mean} should be ~0.5");
}

#[test]
fn ochs_clusters_are_balanced_and_ordered_by_strength() {
    let clusters = ochs_opponent_clusters();
    // All K clusters are used.
    let max = *clusters.iter().max().unwrap();
    assert_eq!(max as usize, OCHS_K - 1, "all {OCHS_K} clusters used");

    // Mass-balanced: each cluster carries ≈ 1326/K of the combos.
    let mut mass = [0u32; OCHS_K];
    for ci in 0..1326 {
        let [a, b] = combo_cards(ci);
        mass[clusters[preflop_index(&[a, b]) as usize] as usize] += 1;
    }
    let target = 1326 / OCHS_K as u32;
    for (c, &m) in mass.iter().enumerate() {
        assert!(m >= target / 2 && m <= 2 * target, "cluster {c} mass {m} far from {target}");
    }

    // Ordered weakest→strongest: AA (pair of aces, class 12) sits at the top,
    // 7-2 offsuit near the bottom.
    let aa = preflop_index(&[make_card(12, 0), make_card(12, 1)]) as usize;
    let seven_two = preflop_index(&[make_card(5, 0), make_card(0, 1)]) as usize;
    assert_eq!(clusters[aa] as usize, OCHS_K - 1, "AA is the strongest tier");
    assert!(clusters[seven_two] < clusters[aa], "72o weaker than AA");
}

#[test]
fn board_ochs_matches_oracle_and_averages_to_uniform() {
    let clusters = ochs_opponent_clusters();
    for board in [dry_board(), wet_board()] {
        let mut ochs = [[f32::NAN; OCHS_K]; 1326];
        board_ochs(board, &clusters, &mut ochs);
        let mut uniform = [f32::NAN; 1326];
        board_equities(board, &mut uniform);

        let mut used = 0u64;
        for &c in &board {
            used |= 1 << c;
        }
        let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

        // Check every hero against an O(n²) per-cluster oracle.
        for i in 0..live.len() {
            for j in (i + 1)..live.len() {
                let (a, b) = (live[i], live[j]);
                let hero = evaluate_7_lut(&[a, b, board[0], board[1], board[2], board[3], board[4]]);
                let (mut win, mut tie, mut tot) = ([0.0f64; OCHS_K], [0.0f64; OCHS_K], [0.0f64; OCHS_K]);
                for &x in &live {
                    for &y in &live {
                        if x >= y || x == a || x == b || y == a || y == b {
                            continue;
                        }
                        let cl = clusters[preflop_index(&[x, y]) as usize] as usize;
                        let opp = evaluate_7_lut(&[x, y, board[0], board[1], board[2], board[3], board[4]]);
                        tot[cl] += 1.0;
                        if hero > opp {
                            win[cl] += 1.0;
                        } else if hero == opp {
                            tie[cl] += 1.0;
                        }
                    }
                }
                let row = ochs[combo_index(a, b)];
                let mut num = 0.0; // weighted reconstruction of equity-vs-uniform
                let mut den = 0.0;
                for c in 0..OCHS_K {
                    let want = if tot[c] > 0.0 { (win[c] + 0.5 * tie[c]) / tot[c] } else { 0.5 };
                    assert!(
                        (row[c] as f64 - want).abs() < 1e-4,
                        "cluster {c} eq {} vs oracle {want} for {a},{b}",
                        row[c]
                    );
                    num += win[c] + 0.5 * tie[c];
                    den += tot[c];
                }
                // Combo-weighted average of the K cluster equities = vs-uniform.
                assert!(
                    (num / den - uniform[combo_index(a, b)] as f64).abs() < 1e-4,
                    "OCHS weighted mean ≠ equity-vs-uniform for {a},{b}"
                );
            }
        }
    }
}

#[test]
fn board_histograms_match_exact_enumeration() {
    // board_histograms must reproduce a direct per-runout enumeration exactly
    // (same f32 equities, same binning, same per-hole denominator).  Checked
    // on a turn board (4 cards) and a flop board (3 cards) for a few holes.
    let bins = 50;
    for board in [&dry_board()[..4], &dry_board()[..3]] {
        let rows = board_histograms(board, bins);
        let mut used = 0u64;
        for &c in board {
            used |= 1 << c;
        }
        let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
        // A handful of representative holes (full sweep is exercised by the
        // equity gate above; here we check the runout accumulation).
        for &(a, b) in &[(live[0], live[1]), (live[5], live[20]), (live[10], live[40])] {
            // Reference: enumerate completions with f32 equities, bin identically.
            let mut reference = vec![0f32; bins];
            let mut n = 0u32;
            let mut completion = |full: [u8; 5]| {
                let e = river_equity([a, b], full) as f32;
                let bin = ((e * bins as f32) as usize).min(bins - 1);
                reference[bin] += 1.0;
                n += 1;
            };
            let mut full = [0u8; 5];
            full[..board.len()].copy_from_slice(board);
            if board.len() == 4 {
                for &r in live.iter().filter(|&&c| c != a && c != b) {
                    full[4] = r;
                    completion(full);
                }
            } else {
                let run: Vec<u8> = live.iter().copied().filter(|&c| c != a && c != b).collect();
                for x in 0..run.len() {
                    for y in (x + 1)..run.len() {
                        full[3] = run[x];
                        full[4] = run[y];
                        completion(full);
                    }
                }
            }
            for r in &mut reference {
                *r /= n as f32;
            }
            let row = &rows[combo_index(a, b) * bins..][..bins];
            assert_eq!(row, &reference[..], "histogram ≠ enumeration for {a},{b}");
        }
    }
}

#[test]
fn ehs_equals_river_equity_on_complete_board() {
    let board = dry_board();
    let hole = [make_card(10, 0), make_card(10, 1)];
    let direct = river_equity(hole, board);
    let via_ehs = ehs(&hole, &board);
    assert!((direct - via_ehs).abs() < 1e-12);
}
