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

use crate::abstraction::canonical::preflop_index;
use crate::util::rng::xorshift_next_unit;

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

/// Lower-triangular index of a hole pair `{a, b}` over the 52 cards into
/// `0..1326` (order-independent).
pub fn combo_index(a: u8, b: u8) -> usize {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    (hi as usize) * (hi as usize - 1) / 2 + lo as usize
}

/// Inverse of [`combo_index`]: the `(lo, hi)` cards (`lo < hi`) for a combo index
/// in `0..1326`.
pub fn combo_cards(index: usize) -> [u8; 2] {
    let mut hi = 1usize;
    while (hi + 1) * hi / 2 <= index {
        hi += 1;
    }
    let lo = index - hi * (hi - 1) / 2;
    [lo as u8, hi as u8]
}

/// Exact equity-vs-random for **every** hole combo on a complete `board`, in one
/// O(n log n) sweep instead of an O(n²) enumeration per hand.
///
/// `out[combo_index(a, b)]` receives `P(win) + 0.5·P(tie)` for each of the 1081
/// holes that avoid the board; combos that use a board card are left as `NaN`.
///
/// Each combo is ranked once; sorting by rank, a single sweep keeps a running
/// count of strictly-weaker combos plus a per-card tally, so for hero `{a, b}`
/// the opponents it beats are `weaker_total − weaker_with_a − weaker_with_b`
/// (the only combo holding both `a` and `b` is the hero itself, so there is no
/// double-count to add back).
pub fn board_equities(board: [u8; 5], out: &mut [f32; 1326]) {
    out.fill(f32::NAN);
    let mut used = 0u64;
    for &c in &board {
        used |= 1 << c;
    }
    let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

    // Rank every hole combo once.  `live` is ascending, so `a < b`.
    let mut combos: Vec<(u32, u8, u8)> = Vec::with_capacity(1081);
    for i in 0..live.len() {
        let a = live[i];
        for &b in &live[i + 1..] {
            let r = evaluate_7_lut(&[a, b, board[0], board[1], board[2], board[3], board[4]]);
            combos.push((r, a, b));
        }
    }
    combos.sort_unstable_by_key(|&(r, _, _)| r);

    // Every hero faces C(45,2) = 990 opponent combos (52 − 5 board − 2 hero).
    const OPPONENTS: f64 = 990.0;
    let mut global_below = 0u32; // combos in strictly-weaker tiers
    let mut below = [0u32; 52]; // …of which, those containing card c
    let mut tier_card = [0u32; 52]; // combos in the current tier containing card c

    let mut i = 0;
    while i < combos.len() {
        let rank = combos[i].0;
        let mut j = i;
        while j < combos.len() && combos[j].0 == rank {
            tier_card[combos[j].1 as usize] += 1;
            tier_card[combos[j].2 as usize] += 1;
            j += 1;
        }
        let tier = (j - i) as u32;

        for &(_, a, b) in &combos[i..j] {
            let (a, b) = (a as usize, b as usize);
            let weaker = global_below - below[a] - below[b];
            // Tied opponents: tier combos holding neither a nor b (+1 re-adds the
            // hero, the lone combo holding both, which was subtracted twice).
            // Add the +1 first so the intermediate never goes negative (u32).
            let tied = tier + 1 - tier_card[a] - tier_card[b];
            let equity = (weaker as f64 + 0.5 * tied as f64) / OPPONENTS;
            out[combo_index(a as u8, b as u8)] = equity as f32;
        }

        // Fold this tier into the running totals and clear its tally.
        global_below += tier;
        for &(_, a, b) in &combos[i..j] {
            below[a as usize] += 1;
            below[b as usize] += 1;
            tier_card[a as usize] = 0;
            tier_card[b as usize] = 0;
        }
        i = j;
    }
}

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
/// holes at once — the same tier sweep as [`board_equities`], but carrying a
/// per-cluster weaker/tied/total tally instead of a single one.  The
/// combo-weighted average of the K cluster equities equals the
/// [`board_equities`] equity-vs-uniform, since the clusters partition the
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

/// Reach-weighted river showdown **counterfactual values** for every hero combo
/// on a complete `board` — the vectorized terminal of public-tree CFR (finding
/// #2).  `opp_reach[o]` is the opponent's reach probability for combo `o`;
/// `out[h]` becomes the hero's net-chip value if both reach the showdown with
/// `half_pot` chips each at risk:
///
/// ```text
/// out[h] = half_pot · Σ_o opp_reach[o] · (+1 win / 0 tie / −1 lose)
/// ```
///
/// This is exactly [`board_equities`] with unit opponent counts replaced by
/// reach weights — one O(n log n) sort + sweep over all 1081 combos instead of
/// the 1326×1326 pairwise showdown.  **Card removal (blockers) is automatic**:
/// the `below[a]/below[b]` / `card[a]/card[b]` subtractions drop every opponent
/// combo that shares a card with the hero (or the board), so a one-hot
/// `opp_reach` reproduces [`hand_vs_hand_equity`] and a uniform one reproduces
/// `board_equities`.  Hero combos that use a board card are left `0.0`.
pub fn board_cfvs(board: [u8; 5], opp_reach: &[f64; 1326], half_pot: f64, out: &mut [f64; 1326]) {
    out.fill(0.0);
    let mut used = 0u64;
    for &c in &board {
        used |= 1 << c;
    }
    let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

    // Total opponent reach and per-card reach, for the blocker-corrected
    // "valid opponents" denominator under each hero hand.
    let mut total_w = 0.0;
    let mut card_w = [0.0; 52];
    let mut combos: Vec<(u32, u8, u8)> = Vec::with_capacity(1081);
    for i in 0..live.len() {
        let a = live[i];
        for &b in &live[i + 1..] {
            let r = opp_reach[combo_index(a, b)];
            total_w += r;
            card_w[a as usize] += r;
            card_w[b as usize] += r;
            let rank = evaluate_7_lut(&[a, b, board[0], board[1], board[2], board[3], board[4]]);
            combos.push((rank, a, b));
        }
    }
    combos.sort_unstable_by_key(|&(r, _, _)| r);

    let mut g_below = 0.0; // reach of strictly-weaker tiers
    let mut below = [0.0; 52]; // …holding card c
    let mut tier_card = [0.0; 52]; // current-tier reach holding card c

    let mut i = 0;
    while i < combos.len() {
        let rank = combos[i].0;
        let mut j = i;
        let mut tier_w = 0.0;
        while j < combos.len() && combos[j].0 == rank {
            let (_, a, b) = combos[j];
            let r = opp_reach[combo_index(a, b)];
            tier_card[a as usize] += r;
            tier_card[b as usize] += r;
            tier_w += r;
            j += 1;
        }

        for &(_, a, b) in &combos[i..j] {
            let (a, b) = (a as usize, b as usize);
            let rh = opp_reach[combo_index(a as u8, b as u8)];
            // Weaker / tied / stronger opponent reach, blockers removed.  Re-add
            // the hero's own combo (subtracted twice via card a and card b).
            let weaker = g_below - below[a] - below[b];
            let tied = tier_w - tier_card[a] - tier_card[b] + rh;
            let valid = total_w - card_w[a] - card_w[b] + rh;
            let stronger = valid - weaker - tied;
            out[combo_index(a as u8, b as u8)] = half_pot * (weaker - stronger);
        }

        g_below += tier_w;
        for &(_, a, b) in &combos[i..j] {
            below[a as usize] += opp_reach[combo_index(a, b)];
            below[b as usize] += opp_reach[combo_index(a, b)];
            tier_card[a as usize] = 0.0;
            tier_card[b as usize] = 0.0;
        }
        i = j;
    }
}

/// Exact equity-distribution histograms for **every** hole combo on a partial
/// `board` (length 3 or 4) — or the scalar river equity (length 5) — built by
/// running [`board_equities`] over every runout.  Returned row-major: row
/// `combo_index(a, b)` is a `bins`-bucket histogram summing to 1 (zeros for
/// holes that use a board card).  This is the exact, low-noise replacement for
/// the Monte-Carlo `ehs_histogram_mc` in the offline build.
pub fn board_histograms(board: &[u8], bins: usize) -> Vec<f32> {
    assert!((3..=5).contains(&board.len()), "board must have 3–5 cards");
    let mut used = 0u64;
    for &c in board {
        used |= 1 << c;
    }
    let runout_cards: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
    let need = 5 - board.len();

    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);
    let mut buf = [f32::NAN; 1326];
    let mut hist = vec![0f32; 1326 * bins];
    let mut counts = vec![0u32; 1326];

    let mut accumulate = |full: [u8; 5], hist: &mut [f32], counts: &mut [u32]| {
        board_equities(full, &mut buf);
        for (ci, &e) in buf.iter().enumerate() {
            if e.is_nan() {
                continue;
            }
            let bin = ((e * bins as f32) as usize).min(bins - 1);
            hist[ci * bins + bin] += 1.0;
            counts[ci] += 1;
        }
    };

    match need {
        0 => accumulate(full, &mut hist, &mut counts),
        1 => {
            for &c in &runout_cards {
                full[board.len()] = c;
                accumulate(full, &mut hist, &mut counts);
            }
        }
        2 => {
            for x in 0..runout_cards.len() {
                for y in (x + 1)..runout_cards.len() {
                    full[3] = runout_cards[x];
                    full[4] = runout_cards[y];
                    accumulate(full, &mut hist, &mut counts);
                }
            }
        }
        _ => unreachable!("board has 3–5 cards"),
    }

    for ci in 0..1326 {
        if counts[ci] > 0 {
            let n = counts[ci] as f32;
            for h in &mut hist[ci * bins..][..bins] {
                *h /= n;
            }
        }
    }
    hist
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

    /// A tiny deterministic unit source for the MC tests.
    fn unit_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed | 1;
        move || crate::util::rng::xorshift_next_unit(&mut s)
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
                        num += (win[c] + 0.5 * tie[c]);
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
}
