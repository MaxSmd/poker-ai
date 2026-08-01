//! OCHS — Opponent Cluster Hand Strength (the river feature).
//!
//! Equity-vs-*uniform* is a lossy 1-D projection: two hands equal against a
//! random opponent can be opposite against a value-heavy range, and no number
//! of scalar buckets ever separates them again.  OCHS scores a hand against
//! each of [`OCHS_K`] opponent strength tiers instead, and measurably dominates
//! the scalar feature at every bucket count (see `examples/bench_ochs.rs`).

use poker_core::evaluate_7_lut;

use crate::abstraction::canonical::preflop_index;
use crate::util::rng::xorshift_next_unit;

use super::{combo_cards, combo_index};

/// Number of opponent strength clusters for the OCHS river feature.
pub const OCHS_K: usize = 8;

/// The 169 suit-canonical pre-flop hand classes ([`preflop_index`] range).
const PREFLOP_CLASSES: usize = 169;

/// Monte-Carlo pre-flop all-in equity of `hole` versus a random opponent over a
/// random 5-card runout (`samples` deals from a `seed`-seeded stream).  Used
/// only to *order* the 169 pre-flop classes into strength tiers, so a coarse MC
/// estimate is plenty; the ordering is what matters, not the third decimal.
fn preflop_equity_mc(hole: [u8; 2], samples: usize, seed: u64) -> f64 {
    let mut used = 0u64;
    for &c in &hole {
        used |= 1 << c;
    }
    let mut deck: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
    let mut s = seed | 1;
    let last = deck.len() - 1;
    let (mut win, mut tie) = (0u64, 0u64);
    for _ in 0..samples {
        // Partial Fisher–Yates for 7 distinct cards: opponent (2) + board (5).
        for k in 0..7 {
            let span = deck.len() - k;
            let j = (k + (xorshift_next_unit(&mut s) * span as f64) as usize).min(last);
            deck.swap(k, j);
        }
        let h = evaluate_7_lut(&[hole[0], hole[1], deck[2], deck[3], deck[4], deck[5], deck[6]]);
        let o = evaluate_7_lut(&[deck[0], deck[1], deck[2], deck[3], deck[4], deck[5], deck[6]]);
        if h > o {
            win += 1;
        } else if h == o {
            tie += 1;
        }
    }
    (win as f64 + 0.5 * tie as f64) / samples as f64
}

/// The OCHS opponent partition: assign each of the 169 pre-flop hand classes
/// ([`preflop_index`] layout) to one of [`OCHS_K`] strength clusters, ordered
/// weakest→strongest by pre-flop equity and **mass-balanced by combo count**
/// (so each cluster carries ≈ 1326/K of the opponent range).  Deterministic
/// (fixed sampling seed) — recompute once and reuse.
///
/// These clusters are the opponent "buckets" the river OCHS feature scores a
/// hand against.  A real opponent does not hold a *uniform* range, so equity
/// against each strength tier separates hands that equity-vs-uniform conflates
/// (e.g. a thin made hand that beats only bluffs vs a draw that got there —
/// equal vs random, opposite vs a value-heavy tier).
pub fn ochs_opponent_clusters() -> [u8; PREFLOP_CLASSES] {
    // A representative combo and the combo count for each pre-flop class.
    let mut rep = [[u8::MAX; 2]; PREFLOP_CLASSES];
    let mut count = [0u32; PREFLOP_CLASSES];
    for ci in 0..1326 {
        let [a, b] = combo_cards(ci);
        let cls = preflop_index(&[a, b]) as usize;
        count[cls] += 1;
        if rep[cls][0] == u8::MAX {
            rep[cls] = [a, b];
        }
    }
    // Pre-flop equity per class (independent deterministic MC stream per class).
    const SAMPLES: usize = 4000;
    let eq: Vec<f64> = (0..PREFLOP_CLASSES)
        .map(|c| {
            let seed = 0x5EED_0000_0000_0000 ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            preflop_equity_mc(rep[c], SAMPLES, seed)
        })
        .collect();
    // Sort classes weakest→strongest, then split into K mass-balanced groups so
    // each strength tier carries roughly equal opponent probability.
    let mut order: Vec<usize> = (0..PREFLOP_CLASSES).collect();
    order.sort_by(|&i, &j| eq[i].partial_cmp(&eq[j]).unwrap());
    let total: u32 = count.iter().sum();
    let mut clusters = [0u8; PREFLOP_CLASSES];
    let (mut cum, mut cl) = (0u32, 0u32);
    for &c in &order {
        clusters[c] = cl as u8;
        cum += count[c];
        if cl + 1 < OCHS_K as u32 && cum >= (cl + 1) * total / OCHS_K as u32 {
            cl += 1;
        }
    }
    clusters
}

/// Exact river **OCHS** (Opponent Cluster Hand Strength) feature for every hero
/// combo on a complete `board`: `out[combo_index(a, b)][c]` is the hero's equity
/// (`P(win) + 0.5·P(tie)`) versus a uniform draw from opponent strength-cluster
/// `c` (see [`ochs_opponent_clusters`]), with card removal applied.  Blocked
/// hero combos (using a board card) are left `f32::NAN`; a cluster with no live
/// opponent for a given hero is reported as the neutral `0.5`.
///
/// One O(n log n) rank-sort + sweep fills all [`OCHS_K`] equities for all 1081
/// holes at once — the same tier sweep as [`board_equities`](super::board_equities), but carrying a
/// per-cluster weaker/tied/total tally instead of a single one.  The
/// combo-weighted average of the K cluster equities equals the
/// [`board_equities`](super::board_equities) equity-vs-uniform, since the clusters partition the
/// opponents (asserted in tests).
pub fn board_ochs(
    board: [u8; 5],
    opp_cluster: &[u8; PREFLOP_CLASSES],
    out: &mut [[f32; OCHS_K]; 1326],
) {
    for row in out.iter_mut() {
        *row = [f32::NAN; OCHS_K];
    }
    let mut used = 0u64;
    for &c in &board {
        used |= 1 << c;
    }
    let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

    // Rank each hole combo once, tagged with the opponent cluster it belongs to,
    // and accumulate per-cluster totals (for the card-removal denominator).
    let mut combos: Vec<(u32, u8, u8, u8)> = Vec::with_capacity(1081); // (rank, a, b, cluster)
    let mut ctot = [0.0f64; OCHS_K];
    let mut ccard = [[0.0f64; 52]; OCHS_K];
    for i in 0..live.len() {
        let a = live[i];
        for &b in &live[i + 1..] {
            let rank = evaluate_7_lut(&[a, b, board[0], board[1], board[2], board[3], board[4]]);
            let cl = opp_cluster[preflop_index(&[a, b]) as usize] as usize;
            combos.push((rank, a, b, cl as u8));
            ctot[cl] += 1.0;
            ccard[cl][a as usize] += 1.0;
            ccard[cl][b as usize] += 1.0;
        }
    }
    combos.sort_unstable_by_key(|&(r, _, _, _)| r);

    let mut g_below = [0.0f64; OCHS_K]; // reach of strictly-weaker tiers, per cluster
    let mut below = [[0.0f64; 52]; OCHS_K]; // …holding card c
    let mut tier_card = [[0.0f64; 52]; OCHS_K]; // current-tier reach holding card c
    let mut tier_count = [0.0f64; OCHS_K]; // current-tier combo count per cluster

    let mut i = 0;
    while i < combos.len() {
        let rank = combos[i].0;
        let mut j = i;
        while j < combos.len() && combos[j].0 == rank {
            let (_, a, b, cl) = combos[j];
            let cl = cl as usize;
            tier_card[cl][a as usize] += 1.0;
            tier_card[cl][b as usize] += 1.0;
            tier_count[cl] += 1.0;
            j += 1;
        }

        for &(_, a, b, cl_h) in &combos[i..j] {
            let (ai, bi, clh) = (a as usize, b as usize, cl_h as usize);
            let mut row = [0.0f32; OCHS_K];
            for (c, slot) in row.iter_mut().enumerate() {
                // Re-add the hero's own combo (subtracted twice via cards a, b)
                // only in its own cluster, so it is excluded exactly once.
                let hero_in = if c == clh { 1.0 } else { 0.0 };
                let weaker = g_below[c] - below[c][ai] - below[c][bi];
                let tied = tier_count[c] - tier_card[c][ai] - tier_card[c][bi] + hero_in;
                let valid = ctot[c] - ccard[c][ai] - ccard[c][bi] + hero_in;
                *slot = if valid > 0.0 { ((weaker + 0.5 * tied) / valid) as f32 } else { 0.5 };
            }
            out[combo_index(a, b)] = row;
        }

        // Fold this tier into the running totals and clear its per-tier tally.
        for &(_, a, b, cl) in &combos[i..j] {
            let cl = cl as usize;
            g_below[cl] += 1.0;
            below[cl][a as usize] += 1.0;
            below[cl][b as usize] += 1.0;
            tier_card[cl][a as usize] = 0.0;
            tier_card[cl][b as usize] = 0.0;
            tier_count[cl] = 0.0;
        }
        i = j;
    }
}
