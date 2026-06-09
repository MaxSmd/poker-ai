//! Offline information-abstraction builder (Phase 2).
//!
//! For each post-flop street it samples boards, computes each canonical
//! situation's equity-distribution histogram **once** (deduped by suit
//! isomorphism), clusters them into buckets, and writes the [`EquityCache`] and
//! [`BucketMap`] to `data/`.  A training run loads the maps into
//! [`BlueprintHoldem`](poker_ai::games::blueprint::BlueprintHoldem) rather than
//! re-clustering.
//!
//!  * **River** — exact equity (the board is complete; one showdown
//!    enumeration per situation).
//!  * **Flop / turn** — Monte-Carlo equity (`ehs_histogram_mc`): exact
//!    enumeration is ~10⁶ showdowns per flop hand, so the runout is sampled
//!    while each sampled showdown stays exact.
//!
//! **Scale.** Full postflop *coverage* (~1.3M canonical flop, ~14M turn, ~123M
//! river situations) is the project's compute-bound step and is the intended
//! cloud burst — raise `--boards` and run it on a high-core server.  The modest
//! defaults here build inspectable maps locally so bucket quality can be eyeballed
//! (the Phase 2 "inspect bucket contents manually" deliverable).
//!
//! Usage:
//!   cluster [boards] [seed]
//!     boards  river boards to sample; flop/turn use boards/2  (default 40)
//!     seed    RNG seed for sampling + clustering               (default 1)

use std::path::Path;

use rayon::prelude::*;

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::abstraction::equity_cache::EquityCache;
use poker_ai::abstraction::features::{ehs_histogram_mc, river_equity};

/// Histogram resolution (plan: 50-bin equity histograms).
const BINS: usize = 50;
/// Monte-Carlo runout samples per flop/turn situation.
const MC_SAMPLES: usize = 80;

/// SplitMix64 — derives an independent, well-mixed RNG seed per board so each
/// board's sampling is reproducible regardless of which thread runs it.
fn mix_seed(seed: u64, board_idx: usize) -> u64 {
    let mut z = seed ^ (board_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xorshift64* — deterministic sampling without a `rand` dependency.
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Draw `n` distinct cards by partial Fisher–Yates over the deck.
    fn cards(&mut self, n: usize) -> Vec<u8> {
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        let last = 51;
        for i in 0..n {
            let span = 52 - i;
            let j = (i + (self.unit() * span as f64) as usize).min(last);
            deck.swap(i, j);
        }
        deck[..n].to_vec()
    }
}

/// Sample `boards` boards of `board_len` cards, compute every hole pair's
/// feature once (canonical dedup), and cluster into `k` buckets.
///
/// Feature by street: the **flop/turn** use the multi-bin equity-distribution
/// histogram (the runout makes the distribution genuinely spread, so L2 on the
/// histogram captures draw shape).  The **river** is a *complete* board — the
/// distribution is a point mass — so its histogram would be one-hot, and L2 on
/// one-hot vectors is degenerate (every distinct equity is equidistant, erasing
/// the ordinal structure).  The river therefore clusters on the scalar equity.
fn build_street(name: &str, board_len: usize, boards: usize, k: usize, seed: u64) -> BucketMap {
    let river = board_len == 5;
    let bins = if river { 1 } else { BINS };
    let start = std::time::Instant::now();

    // Equity precompute is the compute-bound, embarrassingly-parallel step (the
    // plan calls it out).  Each board is an independent task with its own seeded
    // RNG (so the result is reproducible regardless of thread scheduling),
    // building a board-local cache deduped by suit isomorphism.  The board-local
    // caches are merged in fixed board order, keeping the merge deterministic.
    let per_board: Vec<EquityCache> = (0..boards)
        .into_par_iter()
        .map(|b| {
            let mut rng = Rng(mix_seed(seed, b) | 1);
            let board = rng.cards(board_len);
            let mut used = 0u64;
            for &c in &board {
                used |= 1 << c;
            }
            let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
            let mut local = EquityCache::new(bins);
            for i in 0..live.len() {
                for j in (i + 1)..live.len() {
                    let hole = [live[i], live[j]];
                    local.compute_if_absent_with(&hole, &board, || {
                        if river {
                            let full: [u8; 5] =
                                board.as_slice().try_into().expect("river board is 5 cards");
                            vec![river_equity(hole, full) as f32]
                        } else {
                            ehs_histogram_mc(&hole, &board, BINS, MC_SAMPLES, || rng.unit())
                                .iter()
                                .map(|&x| x as f32)
                                .collect()
                        }
                    });
                }
            }
            local
        })
        .collect();

    let mut cache = EquityCache::new(bins);
    for local in per_board {
        cache.merge(local);
    }
    println!(
        "  [{name}] {boards} boards -> {} canonical situations ({:.1}s)",
        cache.len(),
        start.elapsed().as_secs_f64()
    );

    let k = k.min(cache.len().max(1));
    let map = BucketMap::from_cache(&cache, k, seed);
    let dir = Path::new("data");
    std::fs::create_dir_all(dir).expect("create data/");
    cache.save(dir.join(format!("{name}_equity.bin"))).expect("save cache");
    map.save(dir.join(format!("{name}_buckets.bin"))).expect("save map");
    println!(
        "  [{name}] {} buckets over {} situations -> data/{name}_buckets.bin ({:.1}s)\n",
        map.num_buckets(),
        map.len(),
        start.elapsed().as_secs_f64()
    );
    map
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let boards: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("Building card abstraction (river {boards} boards, flop/turn {} each)", boards / 2);
    // Plan bucket targets: flop/turn 500–800, river 800–1200.  Capped to the
    // number of sampled situations at small scales.
    build_street("flop", 3, boards / 2, 500, seed);
    build_street("turn", 4, boards / 2, 500, seed);
    build_street("river", 5, boards, 800, seed);
    println!("Done. Load the *_buckets.bin maps into BlueprintHoldem::with_street_bucket.");
}
