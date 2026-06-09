//! Hand-strength and equity features for card abstraction (Phase 2).
//!
//! The quality of card bucketing sets the strategic ceiling of the whole bot,
//! and bucketing clusters on these features.  The atomic primitive is
//! [`river_equity`]: the exact probability that a hand beats a uniformly random
//! opponent hand on a *complete* board.  Everything else — expected hand
//! strength over future runouts, its second moment, draw potential, and the
//! equity-distribution histogram the clusterer actually consumes — is built by
//! averaging that primitive over the possible board completions.
//!
//! These are computed exactly (full enumeration).  Exact is the right choice
//! for correctness and for the river/turn; the flop's ~10⁶-evaluation cost per
//! hand is why Phase 2 caches results by suit-isomorphic key and (later) uses
//! Monte-Carlo rollouts for the widest layers.  Correctness first, speed via the
//! cache second.

use poker_core::evaluate_7_lut;

/// Exact equity of `hole` on a complete 5-card `board` against a uniformly
/// random opponent hand drawn from the remaining 45 cards.
///
/// Returns `P(win) + 0.5·P(tie)`, in `[0, 1]`.
pub fn river_equity(hole: [u8; 2], board: [u8; 5]) -> f64 {
    let mut used = 0u64;
    for &c in hole.iter().chain(board.iter()) {
        used |= 1 << c;
    }
    let hero = evaluate_7_lut(&[hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]]);
    let remaining: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

    let (mut win, mut tie, mut total) = (0u64, 0u64, 0u64);
    for i in 0..remaining.len() {
        for j in (i + 1)..remaining.len() {
            let opp = evaluate_7_lut(&[
                remaining[i], remaining[j], board[0], board[1], board[2], board[3], board[4],
            ]);
            total += 1;
            if hero > opp {
                win += 1;
            } else if hero == opp {
                tie += 1;
            }
        }
    }
    (win as f64 + 0.5 * tie as f64) / total as f64
}

/// Exact equity of `h0` against the *specific* opponent hand `h1` on a partial
/// `board` (length 3, 4, or 5), enumerating every runout.
///
/// Returns `P(h0 wins) + 0.5·P(tie)`, in `[0, 1]`; `h1`'s equity is the
/// complement.  This is the all-in showdown value used by the resolver's
/// depth-limited leaf evaluator, where both hands are known and the remaining
/// board is rolled out.
pub fn hand_vs_hand_equity(h0: [u8; 2], h1: [u8; 2], board: &[u8]) -> f64 {
    assert!(board.len() <= 5, "board must have at most 5 cards");
    let mut used = 0u64;
    for &c in h0.iter().chain(h1.iter()).chain(board.iter()) {
        used |= 1 << c;
    }
    let remaining: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
    let need = 5 - board.len();

    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);

    let (mut win, mut tie, mut total) = (0u64, 0u64, 0u64);
    let showdown = |full: &[u8; 5], win: &mut u64, tie: &mut u64, total: &mut u64| {
        let r0 = evaluate_7_lut(&[h0[0], h0[1], full[0], full[1], full[2], full[3], full[4]]);
        let r1 = evaluate_7_lut(&[h1[0], h1[1], full[0], full[1], full[2], full[3], full[4]]);
        *total += 1;
        if r0 > r1 {
            *win += 1;
        } else if r0 == r1 {
            *tie += 1;
        }
    };

    match need {
        0 => showdown(&full, &mut win, &mut tie, &mut total),
        1 => {
            for &c in &remaining {
                full[4] = c;
                showdown(&full, &mut win, &mut tie, &mut total);
            }
        }
        2 => {
            for i in 0..remaining.len() {
                for j in (i + 1)..remaining.len() {
                    full[3] = remaining[i];
                    full[4] = remaining[j];
                    showdown(&full, &mut win, &mut tie, &mut total);
                }
            }
        }
        _ => unreachable!("board has 3–5 cards"),
    }
    (win as f64 + 0.5 * tie as f64) / total as f64
}

/// Call `f` with every completed 5-card board reachable from a partial `board`
/// (length 3, 4, or 5) given that `hole` is held.
fn for_each_completion(hole: [u8; 2], board: &[u8], mut f: impl FnMut([u8; 5])) {
    assert!((3..=5).contains(&board.len()), "board must have 3–5 cards");
    let mut used = 0u64;
    for &c in hole.iter().chain(board.iter()) {
        used |= 1 << c;
    }
    let remaining: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
    let need = 5 - board.len();

    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);

    match need {
        0 => f(full),
        1 => {
            for &c in &remaining {
                full[4] = c;
                f(full);
            }
        }
        2 => {
            for i in 0..remaining.len() {
                for j in (i + 1)..remaining.len() {
                    full[3] = remaining[i];
                    full[4] = remaining[j];
                    f(full);
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Expected Hand Strength: the mean of [`river_equity`] over all completions of
/// `board` (length 3/4/5).  On the river this is just the equity itself.
pub fn ehs(hole: &[u8; 2], board: &[u8]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u64;
    for_each_completion(*hole, board, |full| {
        sum += river_equity(*hole, full);
        n += 1;
    });
    sum / n as f64
}

/// Second moment of hand strength over board completions, `E[equity²]`.
/// Together with [`ehs`] it captures the *spread* of outcomes (a draw has the
/// same mean as a made hand but a much wider distribution).
pub fn ehs2(hole: &[u8; 2], board: &[u8]) -> f64 {
    let mut sum_sq = 0.0;
    let mut n = 0u64;
    for_each_completion(*hole, board, |full| {
        let e = river_equity(*hole, full);
        sum_sq += e * e;
        n += 1;
    });
    sum_sq / n as f64
}

/// Draw potential: the fraction of board completions on which the hand becomes
/// strong (equity ≥ `0.6`).  A rough scalar proxy for upside; the histogram
/// captures the full picture.
pub fn draw_potential(hole: &[u8; 2], board: &[u8]) -> f64 {
    let mut strong = 0u64;
    let mut n = 0u64;
    for_each_completion(*hole, board, |full| {
        if river_equity(*hole, full) >= 0.6 {
            strong += 1;
        }
        n += 1;
    });
    strong as f64 / n as f64
}

/// Discretized equity-distribution histogram — the feature the clusterer
/// consumes.  Bins the river equity over all board completions into `bins`
/// equal-width buckets on `[0, 1]`; the returned vector sums to 1.
///
/// This implicitly captures EHS, its variance, and draw potential: a flush draw
/// produces a characteristic bimodal histogram (low when it misses, high when
/// it hits) that clusters apart from a made hand of the same average strength.
pub fn ehs_histogram(hole: &[u8; 2], board: &[u8], bins: usize) -> Vec<f64> {
    let mut hist = vec![0.0; bins];
    let mut n = 0u64;
    for_each_completion(*hole, board, |full| {
        let e = river_equity(*hole, full);
        let bin = ((e * bins as f64) as usize).min(bins - 1);
        hist[bin] += 1.0;
        n += 1;
    });
    if n > 0 {
        for h in &mut hist {
            *h /= n as f64;
        }
    }
    hist
}

/// Monte-Carlo equity-distribution histogram for the flop and turn.
///
/// Exact [`ehs_histogram`] enumerates every runout — ~10⁶ showdowns per flop
/// hand — which is too slow to evaluate for every canonical situation.  This
/// samples `samples` random board completions instead (the showdown at each
/// completion stays *exact* via [`river_equity`], so only the runout is
/// approximated), drawing uniform units from `next_unit`.  On a complete board
/// it defers to the exact histogram, since sampling would add only noise.
///
/// The result is the same potential-aware feature the clusterer consumes: a
/// `bins`-bucket distribution of final equity that sums to 1.
pub fn ehs_histogram_mc(
    hole: &[u8; 2],
    board: &[u8],
    bins: usize,
    samples: usize,
    mut next_unit: impl FnMut() -> f64,
) -> Vec<f64> {
    assert!((3..=5).contains(&board.len()), "board must have 3–5 cards");
    let need = 5 - board.len();
    if need == 0 {
        return ehs_histogram(hole, board, bins);
    }

    let mut used = 0u64;
    for &c in hole.iter().chain(board.iter()) {
        used |= 1 << c;
    }
    let mut deck: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);

    let mut hist = vec![0.0; bins];
    for _ in 0..samples {
        // Partial Fisher–Yates over the live deck yields a uniform `need`-subset
        // each draw, regardless of the deck's running order.
        let last = deck.len() - 1;
        for k in 0..need {
            let span = deck.len() - k;
            let j = (k + (next_unit() * span as f64) as usize).min(last);
            deck.swap(k, j);
            full[board.len() + k] = deck[k];
        }
        let e = river_equity(*hole, full);
        let bin = ((e * bins as f64) as usize).min(bins - 1);
        hist[bin] += 1.0;
    }
    if samples > 0 {
        for h in &mut hist {
            *h /= samples as f64;
        }
    }
    hist
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::make_card;

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

    /// A tiny deterministic unit source for the MC tests.
    fn unit_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed | 1;
        move || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (v >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn mc_histogram_is_a_distribution_and_defers_on_river() {
        // Turn board (4 cards): MC samples runouts and still sums to 1.
        let turn = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3)];
        let hole = [make_card(10, 0), make_card(10, 1)];
        let h = ehs_histogram_mc(&hole, &turn, 20, 500, unit_stream(1));
        assert_eq!(h.len(), 20);
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-9);

        // On a complete board MC must defer to the exact histogram exactly.
        let river = dry_board();
        let exact = ehs_histogram(&hole, &river, 20);
        let mc = ehs_histogram_mc(&hole, &river, 20, 500, unit_stream(1));
        assert_eq!(exact, mc, "river is exact; sampling must not change it");
    }

    #[test]
    fn mc_histogram_mean_approximates_exact_ehs() {
        // The MC histogram's mean equity must track the exact EHS within
        // sampling error.  A turn board keeps the exact reference cheap (46
        // runouts) while the MC path samples them.
        let turn = [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3)];
        let hole = [make_card(10, 0), make_card(10, 1)];
        let exact = ehs(&hole, &turn);

        let bins = 50;
        let h = ehs_histogram_mc(&hole, &turn, bins, 4_000, unit_stream(7));
        // Bin-centre reconstruction of the mean.
        let mc_mean: f64 =
            h.iter().enumerate().map(|(i, &p)| p * (i as f64 + 0.5) / bins as f64).sum();
        assert!((mc_mean - exact).abs() < 0.03, "MC mean {mc_mean} vs exact EHS {exact}");
    }

    #[test]
    fn ehs_equals_river_equity_on_complete_board() {
        let board = dry_board();
        let hole = [make_card(10, 0), make_card(10, 1)];
        let direct = river_equity(hole, board);
        let via_ehs = ehs(&hole, &board);
        assert!((direct - via_ehs).abs() < 1e-12);
    }
}
