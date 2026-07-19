//! Vectorized best response against a trained blueprint — the abstract-game
//! exploitability evaluator that replaces the broken sampled `--expl` number.
//!
//! Walks the blueprint's own abstract betting tree (same `capped_legal` menus,
//! same raise bookkeeping, same info keys as training) carrying a 1326-entry
//! opponent-reach vector, and computes the exact best response of one player
//! against the blueprint strategy of the other:
//!
//! * **betting** — exact: every abstract line is enumerated;
//! * **ranges** — exact: all 1326×1326 hand pairs via reach vectors, card
//!   removal by blocker-corrected sweeps ([`PreparedShowdown`]) at showdowns
//!   and inclusion–exclusion at folds;
//! * **flops** — Monte-Carlo: a sampled flop set stands in for all C(52,3),
//!   scaled by `|F| · C(48,3)/C(52,3)`;
//! * **turn / river** — Monte-Carlo (`board_samples > 0`): `k` cards drawn per
//!   reveal and averaged, OR exact enumeration (`board_samples == 0`) with the
//!   per-pair divisor (45 turns, 44 rivers).
//!
//! **On cost — the reason `board_samples` exists.** Exact turn/river
//! enumeration recurses into the *whole betting subtree* at each of ~48×44
//! cards.  On a tiny validation game that finishes; on the real 200 bb cap-3
//! blueprint it multiplies an already-billion-node tree by ~2000× and does
//! not finish in any practical time (an early exact run was killed after
//! 45 min with no single flop complete).  Sampling the runouts collapses the
//! 48×44 factor to `k²`, turning it into a minutes-to-hours job.  The
//! blueprint side stays unbiased; the BR's max over sampled continuations is
//! mildly upward-biased (shrinks with `k`), and with a fixed seed the bias is
//! reproducible so before/after abstraction A/Bs compare cleanly.
//!
//! `exploitability = (br₀ + br₁) / 2` (NashConv/2, the same convention as
//! `solver::best_response`), in bb/hand.  The number is an *abstract-game*
//! quality metric: it answers "has training converged?" and "did a finer
//! abstraction help?" — not "how exploitable is the bot in full NLHE?"
//! (real-game exploitability also pays for translation and abstraction gaps).
//!
//! Cost model: dominated by river betting nodes × 1326-wide arithmetic.  Work
//! parallelizes over flops (each flop subtree is independent; results are
//! reduced in fixed order, so the value is deterministic for a fixed flop set
//! and seed), with a per-flop progress line to stderr.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Global progress heartbeat: flop subtrees evaluated in the current BR pass.
/// A flop fan-out happens at *every* pre-flop→flop transition (many per pass),
/// so a per-fan-out counter would reset and repeat — this counts the total.
/// Reset at the start of each [`best_response_value`]; the tool runs one BR at
/// a time so a process-global counter is safe.
static FLOP_SUBTREES_DONE: AtomicU64 = AtomicU64::new(0);

use crate::abstraction::features::{combo_cards, PreparedShowdown};
use crate::games::blueprint::BlueprintHoldem;
use crate::play::CompactPolicy;
use crate::util::rng::Rng;
use poker_core::state::GameState;

/// Number of two-card combos (the crate-wide canonical count).
pub const COMBOS: usize = crate::util::combos::NUM_COMBOS;

/// `C(48,3) / C(52,3)`: the probability a uniform flop misses two disjoint
/// hole pairs — the per-pair consistency rate behind the sampled-flop scale.
const FLOP_CONSISTENT_RATE: f64 = 17296.0 / 22100.0;

/// The evaluator's verdict on a blueprint.
#[derive(Debug, Clone, Copy)]
pub struct BrReport {
    /// Best-response value against the blueprint, per BR seat, bb/hand.
    pub br_value_bb: [f64; 2],
    /// `(br₀ + br₁)/2` in milli-big-blinds per hand.
    pub exploitability_mbb: f64,
    /// Flop sample size the numbers were computed over.
    pub flops: usize,
}

/// Exploitability of `policy` in the abstract game `game`, over the given
/// flop set, sampling `board_samples` turn/river runouts per reveal
/// (`0` = exact enumeration; see [`best_response_value`]).
pub fn blueprint_exploitability(
    game: &BlueprintHoldem,
    policy: &CompactPolicy,
    flops: &[[u8; 3]],
    board_samples: usize,
    seed: u64,
) -> BrReport {
    let br0 = best_response_value(game, policy, 0, flops, board_samples, seed);
    let br1 = best_response_value(game, policy, 1, flops, board_samples, seed);
    BrReport {
        br_value_bb: [br0, br1],
        exploitability_mbb: (br0 + br1) / 2.0 * 1000.0,
        flops: flops.len(),
    }
}

/// Value (bb/hand) of the best response for seat `br` when the other seat
/// plays `policy` (uniform at info sets the blueprint never stored — matching
/// how the playing agent treats them).
///
/// `board_samples`: turn/river cards drawn per reveal.  `0` enumerates every
/// card — exact, but only tractable on tiny games (the deep 200 bb cap-3 tree
/// × 48 × 44 does not finish).  `k > 0` samples `k` per reveal; the blueprint
/// side stays unbiased, the BR's max over sampled continuations is mildly
/// upward-biased and shrinks with `k`.  With a fixed `seed` the result is
/// reproducible, so old-vs-new blueprint A/Bs share the sampling bias and
/// compare cleanly.
pub fn best_response_value(
    game: &BlueprintHoldem,
    policy: &CompactPolicy,
    br: usize,
    flops: &[[u8; 3]],
    board_samples: usize,
    seed: u64,
) -> f64 {
    let v = br_value_vector(game, policy, br, flops, board_samples, seed);
    // Missing constant weights: P(opp hand | ours) = 1/1225 and the average
    // over our own 1326 uniformly-dealt hands; then chips → bb.
    v.iter().sum::<f64>() / 1225.0 / 1326.0 / game.big_blind_chips() as f64
}

/// The raw per-hand best-response accumulator (counting measure, chips):
/// entry `combo_index(a, b)` is seat `br`'s value holding `{a, b}`, summed
/// over the blueprint seat's σ-weighted reach and the walk's chance branches.
/// `best_response_value` is its sum over hands divided by `1225 · 1326 · bb`;
/// per-hand diagnostics ("which holdings does the BR print money with?")
/// divide entry `h` by `1225 · bb` instead.
pub fn br_value_vector(
    game: &BlueprintHoldem,
    policy: &CompactPolicy,
    br: usize,
    flops: &[[u8; 3]],
    board_samples: usize,
    seed: u64,
) -> Vec<f64> {
    assert!(!flops.is_empty(), "need at least one flop");
    FLOP_SUBTREES_DONE.store(0, Ordering::Relaxed);
    let mut ctx = Ctx::new(game, policy, br, flops, board_samples, seed);
    let state = game.play_state([[48, 49], [50, 51]], [poker_core::state::NO_CARD; 5]);
    let mut gs = game.game_state(&state).clone();
    let buckets = game.bucket_vector(&[]);
    let reach = vec![1.0f64; COMBOS];
    let mut hist = Vec::new();
    ctx.node(&mut gs, &mut hist, 0, 0, &buckets, &reach, None)
}

/// `n` distinct uniform flops (unordered 3-card sets) for the Monte-Carlo
/// flop dimension.  Deterministic per seed.
pub fn sample_flops(n: usize, seed: u64) -> Vec<[u8; 3]> {
    let mut rng = Rng::new(seed);
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(n);
    while out.len() < n.min(22100) {
        let mut f = [0u8; 3];
        f[0] = (rng.unit() * 52.0) as u8;
        loop {
            f[1] = (rng.unit() * 52.0) as u8;
            if f[1] != f[0] {
                break;
            }
        }
        loop {
            f[2] = (rng.unit() * 52.0) as u8;
            if f[2] != f[0] && f[2] != f[1] {
                break;
            }
        }
        f.sort_unstable();
        if seen.insert(f) {
            out.push(f);
        }
    }
    out
}

/// Every one of the C(52,3) = 22100 flops — the zero-flop-noise mode.
pub fn all_flops() -> Vec<[u8; 3]> {
    let mut out = Vec::with_capacity(22100);
    for a in 0..50u8 {
        for b in (a + 1)..51 {
            for c in (b + 1)..52 {
                out.push([a, b, c]);
            }
        }
    }
    out
}

/// One best-response pass.  Holds the per-pass policy-probability cache
/// (info keys depend on betting history and bucket, not on raw cards, so the
/// same entries recur across every board — caching turns ~10⁹ binary searches
/// into one per distinct info set).  Flop-parallel workers get their own `Ctx`.
struct Ctx<'a> {
    game: &'a BlueprintHoldem,
    policy: &'a CompactPolicy,
    /// The best-responding seat; the other seat plays the blueprint.
    br: usize,
    flops: &'a [[u8; 3]],
    /// `|F| · C(48,3)/C(52,3)` — the flop-chance divisor (see module docs).
    flop_div: f64,
    /// Turn/river runout sampling: `0` enumerates every card (exact, only
    /// tractable on small games — the deep 200 bb tree × 48 × 44 does not
    /// finish); `k > 0` samples `k` cards per reveal and averages, the mode
    /// that makes a real blueprint measurable.  See [`sample_flops`]-style
    /// determinism note on [`best_response_value`].
    board_samples: usize,
    /// Base RNG seed; per-reveal streams are derived deterministically from
    /// this and the board prefix, so a fixed seed is fully reproducible even
    /// across the parallel flop fan-out.
    seed: u64,
    /// Start of the BR pass, for the progress heartbeat.
    start: Instant,
    cache: HashMap<u64, Box<[f64]>>,
    /// `combo_cards` for 0..1326, precomputed.
    cards: Vec<[u8; 2]>,
}

impl<'a> Ctx<'a> {
    fn new(
        game: &'a BlueprintHoldem,
        policy: &'a CompactPolicy,
        br: usize,
        flops: &'a [[u8; 3]],
        board_samples: usize,
        seed: u64,
    ) -> Self {
        Self::with_start(game, policy, br, flops, board_samples, seed, Instant::now())
    }

    #[allow(clippy::too_many_arguments)]
    fn with_start(
        game: &'a BlueprintHoldem,
        policy: &'a CompactPolicy,
        br: usize,
        flops: &'a [[u8; 3]],
        board_samples: usize,
        seed: u64,
        start: Instant,
    ) -> Self {
        Self {
            game,
            policy,
            br,
            flops,
            flop_div: flops.len() as f64 * FLOP_CONSISTENT_RATE,
            board_samples,
            seed,
            start,
            cache: HashMap::new(),
            cards: (0..COMBOS).map(combo_cards).collect(),
        }
    }

    /// A decision node: `reach` is the blueprint seat's per-hand reach (σ-
    /// products, card-removal zeros applied at chance nodes); the returned
    /// vector is the BR seat's accumulated value per hand under the walk's
    /// counting measure.  `buckets` are the current street's per-hand card
    /// buckets; `shown` is the river's prepared showdown once the board is
    /// complete.
    #[allow(clippy::too_many_arguments)]
    fn node(
        &mut self,
        gs: &mut GameState,
        hist: &mut Vec<u8>,
        raises: u8,
        revealed: usize,
        buckets: &[u64],
        reach: &[f64],
        shown: Option<&PreparedShowdown>,
    ) -> Vec<f64> {
        debug_assert!(!gs.is_terminal());
        debug_assert_eq!(revealed, gs.board_cards_count());
        let acts = self.game.capped_actions(gs, raises);
        let n = acts.len();
        let actor = gs.current_player();

        if actor == self.br {
            // Best responder: free to pick per hand — pointwise max over
            // actions (each walk node × hand is exactly one BR info set).
            let mut out = vec![f64::NEG_INFINITY; COMBOS];
            for (i, &act) in acts.iter().enumerate() {
                let child = self.descend(gs, hist, raises, revealed, buckets, reach, shown, act, i);
                for (o, c) in out.iter_mut().zip(&child) {
                    *o = o.max(*c);
                }
            }
            out
        } else {
            // Blueprint seat: weight reach by its per-hand strategy.
            let mut sigma = vec![0.0f64; COMBOS * n];
            for (j, &r) in reach.iter().enumerate() {
                if r == 0.0 {
                    continue;
                }
                let key = self.game.key_from_bucket(actor, revealed, buckets[j], hist);
                if !self.cache.contains_key(&key) {
                    let p = self.policy.probs_or_uniform(key, n).into_boxed_slice();
                    self.cache.insert(key, p);
                }
                sigma[j * n..j * n + n].copy_from_slice(&self.cache[&key]);
            }
            let mut out = vec![0.0f64; COMBOS];
            let mut child_reach = vec![0.0f64; COMBOS];
            for (i, &act) in acts.iter().enumerate() {
                for (j, cr) in child_reach.iter_mut().enumerate() {
                    *cr = reach[j] * sigma[j * n + i];
                }
                let child =
                    self.descend(gs, hist, raises, revealed, buckets, &child_reach, shown, act, i);
                for (o, c) in out.iter_mut().zip(&child) {
                    *o += c;
                }
            }
            out
        }
    }

    /// Apply one action (mutate-and-undo), dispatch on what it produced —
    /// terminal, street transition (chance), or more betting.
    #[allow(clippy::too_many_arguments)]
    fn descend(
        &mut self,
        gs: &mut GameState,
        hist: &mut Vec<u8>,
        raises: u8,
        revealed: usize,
        buckets: &[u64],
        reach: &[f64],
        shown: Option<&PreparedShowdown>,
        act: poker_core::action::Action,
        act_idx: usize,
    ) -> Vec<f64> {
        let (old_street, old_bet) = (gs.street, gs.current_bet);
        gs.apply_action(act);
        hist.push(act_idx as u8);
        let new_raises = self.game.raises_after(raises, old_street, old_bet, gs);

        let out = if gs.is_terminal() {
            if gs.folded != 0 {
                self.fold_value(gs, reach)
            } else if revealed == 5 {
                self.sweep(gs, reach, shown.expect("river showdown needs a prepared board"))
            } else {
                // All-in before the board completed: keep dealing (no more
                // betting), then score the showdown on each complete board.
                self.deal(gs, hist, new_raises, revealed, reach)
            }
        } else if gs.street != old_street {
            self.deal(gs, hist, new_raises, revealed, reach)
        } else {
            self.node(gs, hist, new_raises, revealed, buckets, reach, shown)
        };

        hist.pop();
        gs.undo_action();
        out
    }

    /// A chance layer: enumerate the next street's cards, zero the reach of
    /// blocked opponent combos per branch, skip blocked hero hands when
    /// accumulating, and divide by the per-pair-consistent branch count.
    fn deal(
        &mut self,
        gs: &mut GameState,
        hist: &mut Vec<u8>,
        raises: u8,
        revealed: usize,
        reach: &[f64],
    ) -> Vec<f64> {
        match revealed {
            0 => {
                // Flop fan-out: independent subtrees, parallel; summed in
                // fixed order so the result is deterministic.  Workers build
                // their own `Ctx` (the policy cache is not shared).
                let (game, policy, br, flops) = (self.game, self.policy, self.br, self.flops);
                let (board_samples, seed, start) = (self.board_samples, self.seed, self.start);
                let (gs0, hist0) = (&*gs, &*hist);
                let branches: Vec<Vec<f64>> = flops
                    .par_iter()
                    .map(|f| {
                        let mut ctx = Ctx::with_start(game, policy, br, flops, board_samples, seed, start);
                        let mut gs2 = gs0.clone();
                        gs2.board[..3].copy_from_slice(f);
                        let mut hist2 = hist0.clone();
                        let child_reach = masked_reach(reach, f, &ctx.cards);
                        let v = ctx.after_deal(&mut gs2, &mut hist2, raises, 3, &child_reach);
                        // Heartbeat every 200 flop subtrees (a pass has many
                        // pre-flop→flop transitions × |flops| of these).
                        let n = FLOP_SUBTREES_DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_multiple_of(200) {
                            eprintln!(
                                "    {n} flop subtrees evaluated  [{:.0}s]",
                                start.elapsed().as_secs_f64()
                            );
                        }
                        v
                    })
                    .collect();
                let mut out = vec![0.0f64; COMBOS];
                for (f, branch) in self.flops.iter().zip(&branches) {
                    let block = card_mask(f);
                    for (h, o) in out.iter_mut().enumerate() {
                        let [a, b] = self.cards[h];
                        if block & (1 << a) == 0 && block & (1 << b) == 0 {
                            *o += branch[h];
                        }
                    }
                }
                for o in &mut out {
                    *o /= self.flop_div;
                }
                out
            }
            3 | 4 => {
                // Turn (45) / river (44) reveal.  `board_samples == 0`
                // enumerates every card — exact, only tractable on tiny games.
                // Otherwise sample `board_samples` cards without replacement and
                // average: each sampled card is a uniform draw over the live
                // deck, so the mean is an unbiased estimate of the exact
                // per-pair conditional expectation for the blueprint side; the
                // BR's max over sampled continuations is mildly upward-biased,
                // shrinking with more samples (documented on the CLI).
                let prefix = card_mask(&gs.board[..revealed]);
                let live: Vec<u8> = (0..52u8).filter(|&c| prefix & (1 << c) == 0).collect();
                let cards: Vec<u8> = if self.board_samples == 0 || self.board_samples >= live.len() {
                    live
                } else {
                    // Deterministic per-path stream: seed from the board prefix
                    // so parallel flops stay reproducible.
                    let mut st = splitmix(self.seed ^ (prefix.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                    let mut pool = live;
                    let len = pool.len();
                    for i in 0..self.board_samples {
                        let j = i + (next_unit(&mut st) * (len - i) as f64) as usize;
                        pool.swap(i, j.min(len - 1));
                    }
                    pool.truncate(self.board_samples);
                    pool
                };
                let div = cards.len() as f64;
                let mut out = vec![0.0f64; COMBOS];
                for &c in &cards {
                    gs.board[revealed] = c;
                    let child_reach = masked_reach(reach, &[c], &self.cards);
                    let branch = self.after_deal(gs, hist, raises, revealed + 1, &child_reach);
                    for (h, o) in out.iter_mut().enumerate() {
                        let [a, b] = self.cards[h];
                        if a != c && b != c {
                            *o += branch[h];
                        }
                    }
                }
                for o in &mut out {
                    *o /= div;
                }
                out
            }
            _ => unreachable!("deal at revealed={revealed}"),
        }
    }

    /// Continue below a freshly dealt street: more dealing (all-in run-out),
    /// a showdown sweep (complete board, no betting left), or betting with
    /// the new street's bucket vector.
    fn after_deal(
        &mut self,
        gs: &mut GameState,
        hist: &mut Vec<u8>,
        raises: u8,
        revealed: usize,
        reach: &[f64],
    ) -> Vec<f64> {
        if revealed == 5 {
            let board5: [u8; 5] = gs.board;
            let prepared = PreparedShowdown::new(board5);
            if gs.is_terminal() {
                self.sweep(gs, reach, &prepared)
            } else {
                let buckets = self.game.bucket_vector(&gs.board[..5]);
                self.node(gs, hist, raises, revealed, &buckets, reach, Some(&prepared))
            }
        } else if gs.is_terminal() {
            self.deal(gs, hist, raises, revealed, reach)
        } else {
            let buckets = self.game.bucket_vector(&gs.board[..revealed]);
            self.node(gs, hist, raises, revealed, &buckets, reach, None)
        }
    }

    /// Terminal fold: the folder loses their whole commitment.  Per hero hand,
    /// the consistent opponent reach is `S − S_a − S_b + reach[h]`
    /// (inclusion–exclusion over shared cards; board blockers were already
    /// zeroed at the chance nodes above).
    fn fold_value(&self, gs: &GameState, reach: &[f64]) -> Vec<f64> {
        let folder = if gs.folded & 1 != 0 { 0usize } else { 1 };
        let sign = if folder == self.br { -1.0 } else { 1.0 };
        let amount = gs.total_committed[folder] as f64;

        let mut total = 0.0f64;
        let mut per_card = [0.0f64; 52];
        for (j, &r) in reach.iter().enumerate() {
            if r != 0.0 {
                let [a, b] = self.cards[j];
                total += r;
                per_card[a as usize] += r;
                per_card[b as usize] += r;
            }
        }
        let mut out = vec![0.0f64; COMBOS];
        for (h, o) in out.iter_mut().enumerate() {
            let [a, b] = self.cards[h];
            let consistent = total - per_card[a as usize] - per_card[b as usize] + reach[h];
            *o = sign * amount * consistent;
        }
        out
    }

    /// Terminal showdown on a complete board: blocker-corrected sweep, stakes
    /// = the matched commitment (a short all-in's excess is refunded, so the
    /// net swing is ±min(committed)).
    fn sweep(&self, gs: &GameState, reach: &[f64], prepared: &PreparedShowdown) -> Vec<f64> {
        let matched = gs.total_committed[0].min(gs.total_committed[1]) as f64;
        let mut out = vec![0.0f64; COMBOS];
        let reach_arr: &[f64; COMBOS] = reach.try_into().expect("reach is 1326 wide");
        let out_arr: &mut [f64; COMBOS] = (&mut out[..]).try_into().expect("out is 1326 wide");
        prepared.accumulate(reach_arr, matched, out_arr);
        out
    }
}

/// Bitmask of `cards`.
fn card_mask(cards: &[u8]) -> u64 {
    cards.iter().fold(0u64, |m, &c| m | 1 << c)
}

/// SplitMix64 state initializer (identity — the first [`next_unit`] advances).
fn splitmix(seed: u64) -> u64 {
    seed
}

/// One SplitMix64 draw in `[0, 1)`, advancing `state` in place.
fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// `reach` with every combo that uses one of `cards` zeroed.
fn masked_reach(reach: &[f64], cards: &[u8], combo: &[[u8; 2]]) -> Vec<f64> {
    let block = card_mask(cards);
    let mut out = reach.to_vec();
    for (j, o) in out.iter_mut().enumerate() {
        let [a, b] = combo[j];
        if block & (1 << a) != 0 || block & (1 << b) != 0 {
            *o = 0.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::features::combo_index;
    use poker_core::lut_eval::evaluate_7_lut;
    use poker_core::state::NO_CARD;

    /// 4 bb stacks, cap 1, no bucket maps (raw-index fallback = lossless
    /// card abstraction): a tree small enough for the scalar oracle.
    fn micro() -> BlueprintHoldem {
        BlueprintHoldem::new(8, 2, 1, 0).with_raise_cap(1)
    }

    /// Fully independent scalar reference for one hero hand: explicit
    /// opponent loop, terminal-time consistency masking (vs. the walker's
    /// chance-time reach zeroing + inclusion–exclusion), direct 7-card rank
    /// comparison at showdowns (vs. the walker's sorted sweep), and the
    /// card-based `key_for` (vs. the walker's hoisted `key_from_bucket`).
    /// Shares only the chance-divisor conventions, which are part of the
    /// estimator's definition.
    struct Oracle<'a> {
        game: &'a BlueprintHoldem,
        policy: &'a CompactPolicy,
        br: usize,
        flops: &'a [[u8; 3]],
        flop_div: f64,
        hero: [u8; 2],
    }

    impl Oracle<'_> {
        fn value(game: &BlueprintHoldem, policy: &CompactPolicy, br: usize, flops: &[[u8; 3]], hero: [u8; 2]) -> f64 {
            let o = Oracle {
                game,
                policy,
                br,
                flops,
                flop_div: flops.len() as f64 * FLOP_CONSISTENT_RATE,
                hero,
            };
            let state = game.play_state([[48, 49], [50, 51]], [NO_CARD; 5]);
            let mut gs = game.game_state(&state).clone();
            let reach = vec![1.0f64; COMBOS];
            o.node(&mut gs, &mut Vec::new(), 0, 0, &reach)
        }

        fn blocked(&self, j: usize, board: &[u8]) -> bool {
            let [a, b] = combo_cards(j);
            let mask = card_mask(board) | card_mask(&self.hero);
            mask & (1 << a) != 0 || mask & (1 << b) != 0
        }

        fn node(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64]) -> f64 {
            let acts = self.game.capped_actions(gs, raises);
            let n = acts.len();
            let actor = gs.current_player();
            if actor == self.br {
                (0..n)
                    .map(|i| self.descend(gs, hist, raises, revealed, reach, acts[i], i))
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                let mut total = 0.0;
                for i in 0..n {
                    let mut child = vec![0.0f64; COMBOS];
                    for (j, cr) in child.iter_mut().enumerate() {
                        if reach[j] == 0.0 {
                            continue;
                        }
                        let key = self.game.key_for(
                            actor,
                            combo_cards(j),
                            &gs.board[..revealed],
                            hist,
                        );
                        *cr = reach[j] * self.policy.probs_or_uniform(key, n)[i];
                    }
                    total += self.descend(gs, hist, raises, revealed, &child, acts[i], i);
                }
                total
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn descend(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64], act: poker_core::action::Action, i: usize) -> f64 {
            let (old_street, old_bet) = (gs.street, gs.current_bet);
            gs.apply_action(act);
            hist.push(i as u8);
            let new_raises = self.game.raises_after(raises, old_street, old_bet, gs);
            let out = if gs.is_terminal() {
                if gs.folded != 0 {
                    let folder = if gs.folded & 1 != 0 { 0usize } else { 1 };
                    let sign = if folder == self.br { -1.0 } else { 1.0 };
                    let amount = gs.total_committed[folder] as f64;
                    (0..COMBOS)
                        .filter(|&j| !self.blocked(j, &gs.board[..revealed]))
                        .map(|j| sign * amount * reach[j])
                        .sum()
                } else if revealed == 5 {
                    self.showdown(gs, reach)
                } else {
                    self.deal(gs, hist, new_raises, revealed, reach)
                }
            } else if gs.street != old_street {
                self.deal(gs, hist, new_raises, revealed, reach)
            } else {
                self.node(gs, hist, new_raises, revealed, reach)
            };
            hist.pop();
            gs.undo_action();
            out
        }

        fn deal(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64]) -> f64 {
            let hero_mask = card_mask(&self.hero);
            match revealed {
                0 => {
                    let mut sum = 0.0;
                    for f in self.flops {
                        if card_mask(f) & hero_mask != 0 {
                            continue;
                        }
                        gs.board[..3].copy_from_slice(f);
                        sum += self.after(gs, hist, raises, 3, reach);
                    }
                    sum / self.flop_div
                }
                3 | 4 => {
                    let div = if revealed == 3 { 45.0 } else { 44.0 };
                    let prefix = card_mask(&gs.board[..revealed]);
                    let mut sum = 0.0;
                    for c in 0..52u8 {
                        if (prefix | hero_mask) & (1 << c) != 0 {
                            continue;
                        }
                        gs.board[revealed] = c;
                        sum += self.after(gs, hist, raises, revealed + 1, reach);
                    }
                    sum / div
                }
                _ => unreachable!(),
            }
        }

        fn after(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64]) -> f64 {
            if gs.is_terminal() {
                if revealed == 5 {
                    self.showdown(gs, reach)
                } else {
                    self.deal(gs, hist, raises, revealed, reach)
                }
            } else {
                self.node(gs, hist, raises, revealed, reach)
            }
        }

        fn showdown(&self, gs: &GameState, reach: &[f64]) -> f64 {
            let matched = gs.total_committed[0].min(gs.total_committed[1]) as f64;
            let b = &gs.board;
            let hero_rank = evaluate_7_lut(&[self.hero[0], self.hero[1], b[0], b[1], b[2], b[3], b[4]]);
            let mut sum = 0.0;
            for (j, &r) in reach.iter().enumerate() {
                if r == 0.0 || self.blocked(j, &b[..5]) {
                    continue;
                }
                let [ja, jb] = combo_cards(j);
                let opp_rank = evaluate_7_lut(&[ja, jb, b[0], b[1], b[2], b[3], b[4]]);
                sum += r * matched
                    * if hero_rank > opp_rank {
                        1.0
                    } else if hero_rank < opp_rank {
                        -1.0
                    } else {
                        0.0
                    };
            }
            sum
        }
    }

    /// A non-trivial policy: bias a few of the SB's root info sets so the
    /// σ-weighting path (keys → probs → reach) is exercised, not just the
    /// uniform fallback.
    fn biased_root_policy(game: &BlueprintHoldem) -> CompactPolicy {
        let state = game.play_state([[48, 49], [50, 51]], [NO_CARD; 5]);
        let n = game.actions(&state).len();
        let mut entries = Vec::new();
        for hole in [[48u8, 50], [0u8, 1], [4u8, 9]] {
            let key = game.key_for(0, hole, &[], &[]);
            let mut p = vec![0.05f32; n];
            // Deterministic skew: most mass on one action, different per hand.
            p[(hole[0] as usize) % n] = 1.0 - 0.05 * (n as f32 - 1.0);
            entries.push((key, p));
        }
        CompactPolicy::from_entries(entries)
    }

    /// The gold gate: the vectorized walker equals the fully independent
    /// scalar oracle, per hand, for both BR seats, with biased and uniform
    /// keys in play.  Full turn/river enumeration makes this expensive
    /// (~40 min in the test profile — verified passing 2026-07-15); run it
    /// whenever the walker changes: `cargo test -p poker-ai --release
    /// walker_matches -- --ignored`.
    #[test]
    #[ignore]
    fn walker_matches_the_scalar_oracle() {
        let game = micro();
        let policy = biased_root_policy(&game);
        let flops = vec![[2u8, 17, 33], [5u8, 6, 40]];
        for br in [0usize, 1] {
            // board_samples=0 → exact enumeration, so the walker equals the
            // exact scalar oracle.
            let v = br_value_vector(&game, &policy, br, &flops, 0, 1);
            // AKo-ish and a low suited hand: exercise both biased and uniform keys.
            for hero in [[0u8, 1], [12u8, 16]] {
                let want = Oracle::value(&game, &policy, br, &flops, hero);
                let got = v[combo_index(hero[0], hero[1])];
                let tol = 1e-8 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() < tol,
                    "br={br} hero={hero:?}: walker {got} != oracle {want}"
                );
            }
        }
    }

    /// Cheap always-on smoke: a 1 bb stack collapses betting to shove/fold-
    /// scale trees, but the full deal cascade (flop fan-out, turn/river
    /// enumeration, all-in run-outs, sweeps, masking, the measure) still
    /// runs end-to-end.  Uniform play must read exploitable, the metric
    /// non-negative, and the evaluator deterministic.
    #[test]
    fn smoke_uniform_is_exploitable_and_deterministic() {
        let game = BlueprintHoldem::new(2, 2, 1, 0).with_raise_cap(1);
        let policy = CompactPolicy::from_entries(vec![]);
        let flops = sample_flops(1, 7);
        let r = blueprint_exploitability(&game, &policy, &flops, 0, 1);
        assert!(
            r.exploitability_mbb > 0.0,
            "uniform play must be exploitable, got {} mbb",
            r.exploitability_mbb
        );
        let again = best_response_value(&game, &policy, 0, &flops, 0, 1);
        assert_eq!(again, r.br_value_bb[0], "evaluator must be deterministic");
    }

    /// Board sampling is deterministic for a fixed seed and reduces to the
    /// same shape of answer: uniform play stays exploitable, and two runs
    /// with the same seed agree exactly (the parallel flop fan-out included).
    #[test]
    fn sampled_board_runouts_are_deterministic() {
        let game = BlueprintHoldem::new(2, 2, 1, 0).with_raise_cap(1);
        let policy = CompactPolicy::from_entries(vec![]);
        let flops = sample_flops(2, 7);
        let a = best_response_value(&game, &policy, 0, &flops, 3, 42);
        let b = best_response_value(&game, &policy, 0, &flops, 3, 42);
        assert_eq!(a, b, "fixed seed must be reproducible");
        assert!(a.is_finite());
    }
}
