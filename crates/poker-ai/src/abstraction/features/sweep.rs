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

use rayon::prelude::*;

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
/// one reusable part of the sweep.  Runout averaging (the turn-leaf case)
/// applies the *same* board's structure across every CFR iteration and, via
/// [`PreparedRunout`], across all 44 rivers — turning a per-iteration
/// `O(runouts · n log n)` evaluate+sort into a one-time sort plus a linear sweep.
pub struct PreparedShowdown {
    /// Hero combos `(rank, card a, card b, combo_index(a, b))` sorted ascending
    /// by 7-card rank.  The combo index is *stored*, not recomputed: the tier
    /// sweep touches every combo five times, and `combo_index` is a compare,
    /// a multiply and a shift each time.
    sorted: Vec<(u32, u8, u8, u16)>,
}

/// The reach denominators a showdown sweep divides by: total opponent reach and
/// the per-card reach behind the blocker correction.
///
/// These depend on the reach vector and on which cards the board kills, but
/// **not on hand ranks** — which is what lets [`PreparedRunout::evaluate`] pay
/// for them once for the whole runout instead of once per completion.
#[derive(Clone, Copy)]
struct ReachSums {
    /// Σ reach over every combo avoiding the board.
    total: f64,
    /// …of which, the reach of combos holding card `c`.
    card: [f64; 52],
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
        let mut sorted: Vec<(u32, u8, u8, u16)> = Vec::with_capacity(1081);
        for i in 0..live.len() {
            let a = live[i];
            for &b in &live[i + 1..] {
                let rank = evaluate_7_lut(&[a, b, board[0], board[1], board[2], board[3], board[4]]);
                sorted.push((rank, a, b, combo_index(a, b) as u16));
            }
        }
        sorted.sort_unstable_by_key(|&(r, _, _, _)| r);
        Self { sorted }
    }

    /// **Add** `half_pot · (weaker − stronger)` per hero combo into `out` (the
    /// blocker-corrected, reach-weighted showdown value) — the reach-dependent
    /// sweep of [`board_cfvs`], accumulating so a runout can sum over rivers.
    /// The caller zeroes `out` (single board) or accumulates across boards.
    pub fn accumulate(&self, opp_reach: &[f64; 1326], half_pot: f64, out: &mut [f64; 1326]) {
        self.accumulate_with(opp_reach, half_pot, &self.reach_sums(opp_reach), out);
    }

    /// This board's [`ReachSums`], by direct summation over its live combos —
    /// the O(1081) pass a runout replaces with an O(52) restriction of the
    /// partial board's sums.
    fn reach_sums(&self, opp_reach: &[f64; 1326]) -> ReachSums {
        let mut sums = ReachSums { total: 0.0, card: [0.0; 52] };
        for &(_, a, b, ci) in &self.sorted {
            let r = opp_reach[ci as usize];
            sums.total += r;
            sums.card[a as usize] += r;
            sums.card[b as usize] += r;
        }
        sums
    }

    /// [`accumulate`](Self::accumulate) with the reach denominators supplied,
    /// so a runout can derive them instead of re-summing per completion.
    /// `sums` must be this board's — [`reach_sums`](Self::reach_sums) is the
    /// reference definition.
    fn accumulate_with(
        &self,
        opp_reach: &[f64; 1326],
        half_pot: f64,
        sums: &ReachSums,
        out: &mut [f64; 1326],
    ) {
        let (total_w, card_w) = (sums.total, &sums.card);

        let mut g_below = 0.0; // reach of strictly-weaker tiers
        let mut below = [0.0; 52]; // …holding card c
        let mut tier_card = [0.0; 52]; // current-tier reach holding card c

        let mut i = 0;
        while i < self.sorted.len() {
            let rank = self.sorted[i].0;
            let mut j = i;
            let mut tier_w = 0.0;
            while j < self.sorted.len() && self.sorted[j].0 == rank {
                let (_, a, b, ci) = self.sorted[j];
                let r = opp_reach[ci as usize];
                tier_card[a as usize] += r;
                tier_card[b as usize] += r;
                tier_w += r;
                j += 1;
            }

            for &(_, a, b, ci) in &self.sorted[i..j] {
                let (ua, ub) = (a as usize, b as usize);
                let rh = opp_reach[ci as usize];
                // Weaker / tied / stronger opponent reach, blockers removed.
                // Re-add the hero's own combo (subtracted twice via a and b).
                let weaker = g_below - below[ua] - below[ub];
                let tied = tier_w - tier_card[ua] - tier_card[ub] + rh;
                let valid = total_w - card_w[ua] - card_w[ub] + rh;
                let stronger = valid - weaker - tied;
                out[ci as usize] += half_pot * (weaker - stronger);
            }

            g_below += tier_w;
            for &(_, a, b, ci) in &self.sorted[i..j] {
                let r = opp_reach[ci as usize];
                below[a as usize] += r;
                below[b as usize] += r;
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
///
/// [`evaluate`](Self::evaluate) additionally hoists the reach denominators out
/// of the completion loop — they are rank-free, so the completions share all but
/// an O(52) correction (1.45× on the turn-checkdown resolve, measured).
pub struct PreparedRunout {
    completions: Vec<PreparedShowdown>,
    /// The card(s) each completion adds to the partial board, parallel to
    /// `completions` (only `[..missing]` is meaningful).  This is what turns the
    /// partial board's [`ReachSums`] into a completion's — see
    /// [`restrict`](Self::restrict).
    added: Vec<[u8; 2]>,
    /// Cards not on the partial board: the combos the base sums range over.
    live: Vec<u8>,
    /// Cards each completion adds — 1 from a turn board, 2 from a flop.
    missing: usize,
    /// Live completions per compatible (hero, opp) pair — `C(48 − real, missing)`
    /// — the exact normalizing denominator (see [`board_runout_cfvs`]).
    live_per_pair: f64,
    /// A fixed shuffle of completion indices, stored **twice back to back** so
    /// that any window of up to `completions.len()` entries is one contiguous
    /// slice — which is what lets [`evaluate_sampled`](Self::evaluate_sampled)
    /// take a wrapping window without allocating or branching.
    ///
    /// Shuffled rather than sequential because consecutive completions share a
    /// board card: a contiguous window of the natural order is a systematically
    /// skewed sample (all the runouts bringing one particular card), while a
    /// contiguous window of a shuffle is a spread one.
    order: Vec<u32>,
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
        let mut added = Vec::new();
        let mut full = board;
        match missing {
            1 => {
                for &r in &unused {
                    full[4] = r;
                    completions.push(PreparedShowdown::new(full));
                    added.push([r, r]);
                }
            }
            2 => {
                for i in 0..unused.len() {
                    for j in (i + 1)..unused.len() {
                        full[3] = unused[i];
                        full[4] = unused[j];
                        completions.push(PreparedShowdown::new(full));
                        added.push([unused[i], unused[j]]);
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
        // Deterministic Fisher–Yates.  A fixed-seed LCG, not a thread RNG: the
        // sampling schedule has to be identical on every machine and every run,
        // or two resolves of the same spot stop being reproducible and
        // `resolve_is_bit_identical_across_thread_counts` stops meaning
        // anything.
        let n = completions.len();
        let mut perm: Vec<u32> = (0..n as u32).collect();
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..n).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            perm.swap(i, (s >> 33) as usize % (i + 1));
        }
        let mut order = perm.clone();
        order.extend_from_slice(&perm);

        Self { completions, added, live: unused, missing, live_per_pair, order }
    }

    /// The showdown CFV averaged over the runout: sum every completion's
    /// reach-weighted showdown into `out`, then divide by the live-completion
    /// count (see [`board_runout_cfvs`]).
    pub fn evaluate(&self, opp_reach: &[f64; 1326], half_pot: f64, out: &mut [f64; 1326]) {
        self.evaluate_sampled(opp_reach, half_pot, out, 0, self.completions.len());
    }

    /// [`evaluate`](Self::evaluate) over a **sample** of `sample` completions
    /// instead of all of them, rescaled so the estimate stays unbiased.
    ///
    /// ## Why sampling is nearly free here
    ///
    /// The completion loop is the resolver's hot spot: a turn leaf costs 48 tier
    /// walks and a flop leaf 1176, against ONE for a river leaf. That is the
    /// whole reason a flop resolve costs ~14× a turn one and ~350× a river one
    /// per iteration — 735 identical tree nodes, 1176× the leaf work.
    ///
    /// But a resolve does not need each *iteration's* leaf value to be exact —
    /// it needs the **average strategy** to converge, and that integrates over
    /// every iteration. Total leaf work across a resolve is `iters × sample`,
    /// so 500 iterations at `sample = 32` performs 16 000 tier walks per leaf:
    /// less work than one exact pass at 500 iterations by 36×, yet the averaged
    /// strategy has still seen *more* completions than the 1176 that exist.
    /// The per-iteration estimate gets noisier; the artifact you deploy does
    /// not. This is the same chance-sampling the trainer's external-sampling
    /// MCCFR already relies on, applied at the runout rather than at the deal.
    ///
    /// `round` advances the window, so successive iterations see disjoint
    /// samples and the schedule sweeps the whole completion set every
    /// `ceil(n / sample)` rounds rather than resampling the same subset. It is
    /// systematic sampling, not random: fully deterministic, independent of the
    /// thread schedule, and identical across machines.
    ///
    /// Callers that need the exact expectation — CFV extraction, which runs
    /// once per resolve rather than once per iteration — should use
    /// [`evaluate`](Self::evaluate) and pay for it there, where it is affordable.
    pub fn evaluate_sampled(
        &self,
        opp_reach: &[f64; 1326],
        half_pot: f64,
        out: &mut [f64; 1326],
        round: u64,
        sample: usize,
    ) {
        let n = self.completions.len();
        // `0` is the callers' "exact" sentinel (`VectorCfr::runout_sample`,
        // `BotConfig::runout_sample`), so it must mean *every* completion — not
        // one.  Clamping instead of mapping here silently turned every exact
        // resolve into a 1-completion sample, which read as a strategy
        // regression rather than an arithmetic one.
        let sample = if sample == 0 { n } else { sample.min(n) };

        // The reach denominators are rank-free, so they are *not* per-completion
        // work: across completions they differ only by the combos the new board
        // card(s) block.  Summing them once over the partial board and
        // restricting in O(52) replaces 48 (turn) or 1176 (flop) full passes
        // over ~1081 combos — exactly, no approximation.  The tier walk stays
        // per-completion: ranks change.
        let base = self.base_sums(opp_reach);

        // A contiguous window of the doubled `order`, so no wrap branch and no
        // index gather.
        let offset = (round as usize).wrapping_mul(sample) % n;
        let window = &self.order[offset..offset + sample];

        // The walks are independent, so this is a map-reduce.
        let chunk = Self::chunk_len_for(sample);
        let partials: Vec<[f64; 1326]> = window
            .par_chunks(chunk)
            .map(|block| {
                let mut acc = [0.0; 1326];
                for &ci in block {
                    let ci = ci as usize;
                    let sums = self.restrict(&base, opp_reach, &self.added[ci][..self.missing]);
                    self.completions[ci].accumulate_with(opp_reach, half_pot, &sums, &mut acc);
                }
                acc
            })
            .collect();

        // Fold in chunk order, not completion order: the partials are summed by
        // index, so the result is identical run to run whatever the thread
        // schedule was.  It is NOT bit-identical to the sequential fold — the
        // additions re-associate — but the difference is f64 rounding, four
        // orders below the 1e-9 the runout tests hold `evaluate` to.
        out.fill(0.0);
        for partial in &partials {
            for (o, &v) in out.iter_mut().zip(partial.iter()) {
                *o += v;
            }
        }

        // `live_per_pair` normalizes a full sweep; a sample of `sample`-in-`n`
        // carries `sample / n` of that mass, so scaling back up by `n / sample`
        // leaves the estimator unbiased.  At `sample == n` this is exactly the
        // old divisor.
        let scale = n as f64 / (self.live_per_pair * sample as f64);
        for o in out.iter_mut() {
            *o *= scale;
        }
    }

    /// Completions per parallel chunk.  Depends only on the completion count —
    /// a property of the board shape — so the chunk boundaries, and therefore
    /// the summation order, are the same on every machine and every run.
    ///
    /// `TARGET` sets the parallel width; the floor keeps a task worth more than
    /// the cost of scheduling it, which matters on the turn's 48 completions
    /// (a leaf is evaluated once per traverser per iteration, so these tasks
    /// are spawned in the hundreds of thousands over a resolve).
    fn chunk_len_for(n: usize) -> usize {
        const TARGET: usize = 64;
        const FLOOR: usize = 4;
        n.div_ceil(TARGET).max(FLOOR)
    }

    /// Reach sums over every combo avoiding the *partial* board — the superset
    /// each completion restricts.
    fn base_sums(&self, opp_reach: &[f64; 1326]) -> ReachSums {
        let mut sums = ReachSums { total: 0.0, card: [0.0; 52] };
        for i in 0..self.live.len() {
            let a = self.live[i];
            for &b in &self.live[i + 1..] {
                let r = opp_reach[combo_index(a, b)];
                sums.total += r;
                sums.card[a as usize] += r;
                sums.card[b as usize] += r;
            }
        }
        sums
    }

    /// `base` minus every combo holding a `dead` (completion) card — the
    /// completion's own [`ReachSums::total`]/[`card`](ReachSums::card).
    ///
    /// `total` drops `base.card[d]` per dead card, with the combos holding *two*
    /// dead cards added back (they were subtracted twice); `card[x]` drops the
    /// combos `{x, d}`.  A dead card's own slot is zeroed rather than fixed up:
    /// no live hero combo holds a board card, so the sweep never reads it.
    fn restrict(&self, base: &ReachSums, opp_reach: &[f64; 1326], dead: &[u8]) -> ReachSums {
        let mut sums = *base;
        for (k, &d) in dead.iter().enumerate() {
            sums.total -= base.card[d as usize];
            for &e in &dead[..k] {
                sums.total += opp_reach[combo_index(d, e)];
            }
        }
        for &x in &self.live {
            if dead.contains(&x) {
                sums.card[x as usize] = 0.0;
                continue;
            }
            for &d in dead {
                sums.card[x as usize] -= opp_reach[combo_index(x, d)];
            }
        }
        sums
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

#[cfg(test)]
mod runout_denominator_tests {
    use super::*;
    use poker_core::{make_card, state::NO_CARD};

    /// A deliberately lumpy reach vector: zeros, a wide dynamic range, and mass
    /// on combos that use board cards (which the sums must exclude).
    fn lumpy_reach() -> [f64; 1326] {
        let mut reach = [0.0; 1326];
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for (i, r) in reach.iter_mut().enumerate() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *r = if i % 7 == 0 { 0.0 } else { (x >> 40) as f64 / 1024.0 };
        }
        reach
    }

    /// The O(52) restriction of the partial board's sums must reproduce each
    /// completion's own O(1081) summation — this is the whole hoist.
    fn check_restriction(board: [u8; 5]) {
        let runout = PreparedRunout::new(board);
        let reach = lumpy_reach();
        let base = runout.base_sums(&reach);
        for (c, added) in runout.completions.iter().zip(&runout.added) {
            let derived = runout.restrict(&base, &reach, &added[..runout.missing]);
            let direct = c.reach_sums(&reach);
            assert!(
                (derived.total - direct.total).abs() < 1e-9,
                "total {} vs {}",
                derived.total,
                direct.total
            );
            // Only slots a live hero combo can index are read by the sweep.
            for &(_, a, b, _) in &c.sorted {
                for x in [a as usize, b as usize] {
                    assert!(
                        (derived.card[x] - direct.card[x]).abs() < 1e-9,
                        "card[{x}] {} vs {}",
                        derived.card[x],
                        direct.card[x]
                    );
                }
            }
        }
    }

    #[test]
    fn turn_runout_denominators_match_direct_summation() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(4, 2), make_card(2, 3), NO_CARD];
        check_restriction(board);
    }

    #[test]
    fn flop_runout_denominators_match_direct_summation() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(4, 2), NO_CARD, NO_CARD];
        check_restriction(board);
    }

    /// End to end: `evaluate` (hoisted denominators) must agree with the
    /// per-completion `accumulate` it replaced, to well inside f64 noise.
    #[test]
    fn evaluate_matches_per_completion_accumulate() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(4, 2), make_card(2, 3), NO_CARD];
        let runout = PreparedRunout::new(board);
        let reach = lumpy_reach();

        let mut fast = [0.0; 1326];
        runout.evaluate(&reach, 7.5, &mut fast);

        let mut slow = [0.0; 1326];
        for c in &runout.completions {
            c.accumulate(&reach, 7.5, &mut slow);
        }
        for o in slow.iter_mut() {
            *o /= runout.live_per_pair;
        }

        let worst = fast.iter().zip(&slow).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let scale = slow.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
        assert!(worst < 1e-9 * scale, "hoisted denominators changed the CFV: {worst} (scale {scale})");
    }

    /// **The correctness claim behind runout sampling.**
    ///
    /// The schedule is systematic, so consecutive rounds take disjoint windows
    /// of the shuffled order; when `sample` divides the completion count, one
    /// full cycle of `n / sample` rounds tiles the set exactly once. Averaging
    /// that cycle must therefore reproduce the exact sweep to f64 noise — which
    /// is what makes the estimator unbiased rather than merely close.
    ///
    /// If this fails, the `n / sample` rescaling in `evaluate_sampled` is wrong
    /// and every sampled resolve is quietly mis-scaled — a bug that would look
    /// like a strategy problem, not an arithmetic one.
    #[test]
    fn sampled_runout_averages_to_the_exact_sweep_over_one_cycle() {
        // A turn board: 48 completions, so 8 divides the set into 6 windows.
        let board = [make_card(12, 0), make_card(11, 1), make_card(4, 2), make_card(2, 3), NO_CARD];
        let runout = PreparedRunout::new(board);
        assert_eq!(runout.completions.len(), 48);
        let reach = lumpy_reach();

        let mut exact = [0.0; 1326];
        runout.evaluate(&reach, 7.5, &mut exact);

        const SAMPLE: usize = 8;
        let rounds = 48 / SAMPLE;
        let mut mean = [0.0; 1326];
        for round in 0..rounds {
            let mut one = [0.0; 1326];
            runout.evaluate_sampled(&reach, 7.5, &mut one, round as u64, SAMPLE);
            for (m, &v) in mean.iter_mut().zip(one.iter()) {
                *m += v / rounds as f64;
            }
        }

        let worst = exact.iter().zip(&mean).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
        assert!(worst < 1e-9 * scale, "sampling is biased: {worst} (scale {scale})");
    }

    /// Sampling noise must be a **knob**, not a cliff: a bigger sample has to
    /// mean a closer estimate, monotonically, or `runout_sample` is not
    /// something a caller can trade against latency.
    ///
    /// Measured RMS deviation from the exact sweep for one round on a 48-
    /// completion turn board, as a percentage of the value scale:
    ///
    /// ```text
    ///    4/48   11.8%      16/48   4.7%
    ///    8/48    9.2%      24/48   3.4%
    ///   12/48    7.1%      48/48   0.0%
    /// ```
    ///
    /// Those are per-*iteration* errors, and they are large on purpose — the
    /// point of sampling is that CFR averages them away across iterations, which
    /// `sampled_runout_averages_to_the_exact_sweep_over_one_cycle` pins. What
    /// this test guards is that the trade stays legible: pay more completions,
    /// get a better estimate.
    #[test]
    fn sampling_noise_falls_monotonically_with_sample_size() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(4, 2), make_card(2, 3), NO_CARD];
        let runout = PreparedRunout::new(board);
        let reach = lumpy_reach();

        let mut exact = [0.0; 1326];
        runout.evaluate(&reach, 7.5, &mut exact);
        let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);

        let rms = |sample: usize| {
            let mut one = [0.0; 1326];
            runout.evaluate_sampled(&reach, 7.5, &mut one, 0, sample);
            let n = exact.len() as f64;
            (exact.iter().zip(&one).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / n).sqrt() / scale
        };

        let errs: Vec<f64> = [4usize, 8, 12, 16, 24].iter().map(|&s| rms(s)).collect();
        for w in errs.windows(2) {
            assert!(w[1] < w[0], "a larger sample got worse: {errs:?}");
        }
        assert!(errs[0] < 0.20, "even a 4/48 sample should track the sweep: {}", errs[0]);
        assert!(rms(48) < 1e-12, "a full sample must be the exact sweep, got {}", rms(48));
    }

    /// `evaluate` must remain exactly the full sweep after being re-expressed in
    /// terms of the sampled path — the sampled window at `sample == n` is a
    /// permutation of every completion, and permuting a sum must not change it.
    #[test]
    fn exact_evaluate_is_unchanged_by_the_sampling_refactor() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(4, 2), NO_CARD, NO_CARD];
        let runout = PreparedRunout::new(board);
        let reach = lumpy_reach();

        let mut via_evaluate = [0.0; 1326];
        runout.evaluate(&reach, 3.25, &mut via_evaluate);

        let mut direct = [0.0; 1326];
        for c in &runout.completions {
            c.accumulate(&reach, 3.25, &mut direct);
        }
        for o in direct.iter_mut() {
            *o /= runout.live_per_pair;
        }

        let worst =
            via_evaluate.iter().zip(&direct).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let scale = direct.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
        assert!(worst < 1e-9 * scale, "the exact path drifted: {worst} (scale {scale})");

        // `sample = 0` is the "exact" sentinel every caller passes down.  An
        // earlier version clamped it to 1 instead of mapping it to `n`, turning
        // every supposedly-exact resolve into a one-completion sample — which
        // surfaced only as a turn resolve missing an exploitability bound by
        // 35%, i.e. as a strategy bug.  Pin the sentinel directly, at every
        // round, so it cannot regress silently again.
        for round in [0u64, 1, 7, 1000] {
            let mut sentinel = [0.0; 1326];
            runout.evaluate_sampled(&reach, 3.25, &mut sentinel, round, 0);
            let worst = sentinel
                .iter()
                .zip(&via_evaluate)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(worst < 1e-9 * scale, "sample=0 is not exact at round {round}: {worst}");
        }
    }
}
