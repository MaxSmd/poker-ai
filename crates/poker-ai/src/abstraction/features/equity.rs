//! Per-hand exact equity primitives.
//!
//! [`river_equity`] is the atomic one: the exact probability a hand beats a
//! uniformly random opponent on a complete board.  Everything above it here
//! ([`ehs`], [`ehs2`], [`draw_potential`], [`ehs_histogram`]) averages it over
//! board completions, one hand at a time — the readable definition of each
//! feature.  The offline build does not call these per hand; it uses the
//! all-hands-at-once sweeps in [`super::sweep`], which are exact and ~1000×
//! cheaper.  These stay as the definitional reference (and the sweeps' test
//! oracle), plus [`hand_vs_hand_equity`] for the resolver's leaf evaluation.

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

/// depth-limited leaf evaluator, where both hands are known and the remaining
/// board is rolled out.  An empty board enumerates all C(48,5) ≈ 1.7M runouts
/// (a few tens of ms with the LUT) — used by the luck-adjusted match scorer,
/// which needs the exact tower property `E[eq after a reveal] = eq before`.
/// Exact equity of `h0` against the *specific* opponent hand `h1` on a partial
/// `board` (length 0 to 5), enumerating every runout.
///
/// Returns `P(h0 wins) + 0.5·P(tie)`, in `[0, 1]`; `h1`'s equity is the
/// complement.  This is the all-in showdown value used by the resolver's
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

    // Enumerate every `need`-card completion (combinations of `remaining`) by
    // recursion on the slot index; `need` ≤ 2 covers the resolver's hot path
    // exactly as the old hand-rolled loops did, deeper boards are the offline
    // luck-scorer path.
    fn complete(
        remaining: &[u8],
        from: usize,
        slot: usize,
        full: &mut [u8; 5],
        f: &mut impl FnMut(&[u8; 5]),
    ) {
        if slot == 5 {
            f(full);
            return;
        }
        for i in from..remaining.len() {
            full[slot] = remaining[i];
            complete(remaining, i + 1, slot + 1, full, f);
        }
    }
    if need == 0 {
        showdown(&full, &mut win, &mut tie, &mut total);
    } else {
        complete(&remaining, 0, 5 - need, &mut full, &mut |b: &[u8; 5]| {
            showdown(b, &mut win, &mut tie, &mut total)
        });
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
