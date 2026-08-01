//! All-hands-at-once board sweeps — the offline build's and the resolver's
//! hot path.
//!
//! On a fixed board every hole's equity comes from ONE `O(n log n)` pass: rank
//! each of the 1326 combos once, sort, then sweep in rank tiers carrying a
//! running count of weaker combos plus a per-card tally, so each hero's
//! beaten-opponent count is three subtractions.  That replaces a per-hand
//! `O(n²)` enumeration (~1.07M evaluations for one board) and is *exact*, not
//! sampled.  [`PreparedShowdown`] and [`PreparedRunout`] hoist the sort out of
//! the iteration loop so a CFR resolve re-sorts nothing.

use poker_core::evaluate_7_lut;

use super::combo_index;

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
/// `opp_reach` reproduces [`hand_vs_hand_equity`](super::hand_vs_hand_equity) and a uniform one reproduces
/// `board_equities`.  Hero combos that use a board card are left `0.0`.
pub fn board_cfvs(board: [u8; 5], opp_reach: &[f64; 1326], half_pot: f64, out: &mut [f64; 1326]) {
    out.fill(0.0);
    PreparedShowdown::new(board).accumulate(opp_reach, half_pot, out);
}

/// A complete board with its 1081 hero combos **pre-sorted by 7-card rank**, so
/// the reach-weighted showdown sweep of [`board_cfvs`] can be repeated over many
/// different opponent-reach vectors without re-evaluating hands or re-sorting.
///
/// The rank order is a function of the board alone (not the reach), so it is the
/// one reusable part of the sweep.  Runout averaging (finding #1's turn leaf)
/// applies the *same* board's structure across every CFR iteration and, via
/// [`PreparedRunout`], across all 44 rivers — turning a per-iteration
/// `O(runouts · n log n)` evaluate+sort into a one-time sort plus a linear sweep.
pub struct PreparedShowdown {
    /// Hero combos `(rank, card a, card b)` sorted ascending by 7-card rank.
    sorted: Vec<(u32, u8, u8)>,
}

impl PreparedShowdown {
    /// Sort a complete `board`'s combos by rank once (the reusable, reach-free
    /// part of the showdown sweep).
    pub fn new(board: [u8; 5]) -> Self {
        let mut used = 0u64;
        for &c in &board {
            used |= 1 << c;
        }
        let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
        let mut sorted: Vec<(u32, u8, u8)> = Vec::with_capacity(1081);
        for i in 0..live.len() {
            let a = live[i];
            for &b in &live[i + 1..] {
                let rank = evaluate_7_lut(&[a, b, board[0], board[1], board[2], board[3], board[4]]);
                sorted.push((rank, a, b));
            }
        }
        sorted.sort_unstable_by_key(|&(r, _, _)| r);
        Self { sorted }
    }

    /// **Add** `half_pot · (weaker − stronger)` per hero combo into `out` (the
    /// blocker-corrected, reach-weighted showdown value) — the reach-dependent
    /// sweep of [`board_cfvs`], accumulating so a runout can sum over rivers.
    /// The caller zeroes `out` (single board) or accumulates across boards.
    pub fn accumulate(&self, opp_reach: &[f64; 1326], half_pot: f64, out: &mut [f64; 1326]) {
        // Total opponent reach and per-card reach, for the blocker-corrected
        // "valid opponents" denominator under each hero hand.
        let mut total_w = 0.0;
        let mut card_w = [0.0; 52];
        for &(_, a, b) in &self.sorted {
            let r = opp_reach[combo_index(a, b)];
            total_w += r;
            card_w[a as usize] += r;
            card_w[b as usize] += r;
        }

        let mut g_below = 0.0; // reach of strictly-weaker tiers
        let mut below = [0.0; 52]; // …holding card c
        let mut tier_card = [0.0; 52]; // current-tier reach holding card c

        let mut i = 0;
        while i < self.sorted.len() {
            let rank = self.sorted[i].0;
            let mut j = i;
            let mut tier_w = 0.0;
            while j < self.sorted.len() && self.sorted[j].0 == rank {
                let (_, a, b) = self.sorted[j];
                let r = opp_reach[combo_index(a, b)];
                tier_card[a as usize] += r;
                tier_card[b as usize] += r;
                tier_w += r;
                j += 1;
            }

            for &(_, a, b) in &self.sorted[i..j] {
                let (ua, ub) = (a as usize, b as usize);
                let rh = opp_reach[combo_index(a, b)];
                // Weaker / tied / stronger opponent reach, blockers removed.
                // Re-add the hero's own combo (subtracted twice via a and b).
                let weaker = g_below - below[ua] - below[ub];
                let tied = tier_w - tier_card[ua] - tier_card[ub] + rh;
                let valid = total_w - card_w[ua] - card_w[ub] + rh;
                let stronger = valid - weaker - tied;
                out[combo_index(a, b)] += half_pot * (weaker - stronger);
            }

            g_below += tier_w;
            for &(_, a, b) in &self.sorted[i..j] {
                below[a as usize] += opp_reach[combo_index(a, b)];
                below[b as usize] += opp_reach[combo_index(a, b)];
                tier_card[a as usize] = 0.0;
                tier_card[b as usize] = 0.0;
            }
            i = j;
        }
    }
}

/// Every board completion of a fixed **incomplete** board (turn or flop), each
/// pre-sorted as a [`PreparedShowdown`] — the reusable structure behind
/// [`board_runout_cfvs`].
///
/// Built once per resolve; a subgame's every depth-cut leaf shares the same
/// board, so the whole solve reuses this one table across all iterations and all
/// leaves, replacing an evaluate+sort pass *per completion per leaf per
/// iteration* with a linear sweep over precomputed ranks.  A turn board (one
/// missing card) enumerates 48 river completions; a flop board (two missing)
/// enumerates C(49, 2) = 1176 turn+river completions.
pub struct PreparedRunout {
    completions: Vec<PreparedShowdown>,
    /// Live completions per compatible (hero, opp) pair — `C(48 − real, missing)`
    /// — the exact normalizing denominator (see [`board_runout_cfvs`]).
    live_per_pair: f64,
}

impl PreparedRunout {
    /// Pre-sort every board completion of an incomplete `board` (trailing slots
    /// [`NO_CARD`](poker_core::state::NO_CARD)): 4 real cards ⇒ the 48 river runouts (turn); 3 real cards ⇒
    /// the 1176 turn+river runouts (flop).  Panics otherwise (a complete or
    /// under-specified board is not a depth-cut leaf board).
    pub fn new(board: [u8; 5]) -> Self {
        let mut used = 0u64;
        let mut real = 0;
        for &c in &board {
            if c != poker_core::state::NO_CARD {
                used |= 1 << c;
                real += 1;
            }
        }
        assert!(
            real == 3 || real == 4,
            "PreparedRunout needs a flop (3-card) or turn (4-card) board, got {real} cards"
        );
        let missing = 5 - real;
        let unused: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

        let mut completions = Vec::new();
        let mut full = board;
        match missing {
            1 => {
                for &r in &unused {
                    full[4] = r;
                    completions.push(PreparedShowdown::new(full));
                }
            }
            2 => {
                for i in 0..unused.len() {
                    for j in (i + 1)..unused.len() {
                        full[3] = unused[i];
                        full[4] = unused[j];
                        completions.push(PreparedShowdown::new(full));
                    }
                }
            }
            _ => unreachable!("real is 3 or 4"),
        }

        // C(48 − real, missing): completions avoiding board (real) + hero (2) +
        // opp (2), i.e. drawn from the 48 − real cards a compatible pair leaves.
        let free = 48 - real; // cards not on the board and not in either hand
        let live_per_pair = if missing == 1 {
            free as f64
        } else {
            (free * (free - 1) / 2) as f64
        };
        Self { completions, live_per_pair }
    }

    /// The showdown CFV averaged over the runout: sum every completion's
    /// reach-weighted showdown into `out`, then divide by the live-completion
    /// count (see [`board_runout_cfvs`]).
    pub fn evaluate(&self, opp_reach: &[f64; 1326], half_pot: f64, out: &mut [f64; 1326]) {
        out.fill(0.0);
        for c in &self.completions {
            c.accumulate(opp_reach, half_pot, out);
        }
        for o in out.iter_mut() {
            *o /= self.live_per_pair;
        }
    }
}

/// Depth-limit showdown counterfactual value, averaged over the board runout —
/// the check-down leaf value for **turn** (one-card) and **flop** (two-card)
/// subgame resolving.
///
/// `board` holds the known cards with trailing [`NO_CARD`](poker_core::state::NO_CARD) slots; `opp_reach` is
/// the opponent range in [`combo_index`] ordering, `half_pot` the stake at
/// showdown (bb).  For each board completion the exact showdown CFV
/// [`board_cfvs`] is accumulated, then divided by the number of completions a
/// compatible (hero, opp) pair leaves live — `44` on the turn, `C(45, 2) = 990`
/// on the flop.  Completions colliding with the hero's own cards contribute zero
/// (`board_cfvs` never writes a blocked hero combo), so summing over *all*
/// completions and dividing by that count is exact: it reproduces
/// `Σ_g π(g)·(eq(h,g) − ½)·pot`, the value the explicit
/// [`CheckdownLeafEval`](crate::validation::resolving::leaf_eval) oracle scores there.
///
/// This convenience wrapper rebuilds the sort each call; the hot resolve path
/// builds a [`PreparedRunout`] once and calls [`PreparedRunout::evaluate`].
pub fn board_runout_cfvs(board: [u8; 5], opp_reach: &[f64; 1326], half_pot: f64, out: &mut [f64; 1326]) {
    PreparedRunout::new(board).evaluate(opp_reach, half_pot, out);
}

/// Exact equity-distribution histograms for **every** hole combo on a partial
/// `board` (length 3 or 4) — or the scalar river equity (length 5) — built by
/// running [`board_equities`] over every runout.  Returned row-major: row
/// `combo_index(a, b)` is a `bins`-bucket histogram summing to 1 (zeros for
/// holes that use a board card).  This is the exact, low-noise replacement for
/// the Monte-Carlo rollouts the offline build originally used.
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
