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

mod walk;
#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::games::blueprint::BlueprintHoldem;
use crate::play::CompactPolicy;
use crate::util::rng::Rng;

use walk::Ctx;

/// Global progress heartbeat: flop subtrees evaluated in the current BR pass.
/// A flop fan-out happens at *every* pre-flop→flop transition (many per pass),
/// so a per-fan-out counter would reset and repeat — this counts the total.
/// Reset at the start of each [`best_response_value`]; the tool runs one BR at
/// a time so a process-global counter is safe.
static FLOP_SUBTREES_DONE: AtomicU64 = AtomicU64::new(0);

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

