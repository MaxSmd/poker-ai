//! The best-response tree walk.
//!
//! One [`Ctx`] per flop-parallel worker walks the blueprint's own abstract
//! betting tree carrying a 1326-wide reach vector: at the blueprint's nodes it
//! averages under the policy, at the best responder's it takes the max — the
//! standard vector BR, with exact inclusion–exclusion folds and
//! [`PreparedShowdown`] sweeps at the terminals.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use poker_core::state::GameState;

use super::{COMBOS, FLOP_CONSISTENT_RATE, FLOP_SUBTREES_DONE};
use crate::abstraction::features::{combo_cards, PreparedShowdown};
use crate::games::blueprint::BlueprintHoldem;
use crate::play::CompactPolicy;

/// One best-response pass.  Holds the per-pass policy-probability cache
/// (info keys depend on betting history and bucket, not on raw cards, so the
/// same entries recur across every board — caching turns ~10⁹ binary searches
/// into one per distinct info set).  Flop-parallel workers get their own `Ctx`.
pub(super) struct Ctx<'a> {
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
    pub(super) fn new(
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
    pub(super) fn with_start(
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
    pub(super) fn node(
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
    pub(super) fn descend(
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
    pub(super) fn deal(
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
pub(super) fn card_mask(cards: &[u8]) -> u64 {
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
