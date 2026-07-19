//! Offline information-abstraction builder (Phase 2).
//!
//! Built **per board** via the equity sweep ([`board_equities`] /
//! [`board_histograms`]): one board scores all ~1081 holes at once in
//! O(n log n), so the feature for every situation is **exact and ~1000× cheaper**
//! than the old per-hand O(n²) enumeration / Monte-Carlo rollout (exact, low-noise
//! histograms cluster better — sampling noise is the ceiling).
//!
//! Boards are enumerated by a suit-isomorphic board indexer and processed in
//! parallel; each board emits `(joint slot, feature)` entries that are merged
//! serially into the flat [`EquityCache`] keyed by finding #2's joint
//! `(hole, board)` index.  Writes to a joint slot from different raw boards are
//! idempotent (suit-iso ⇒ identical feature), and iterating canonical boards ×
//! all holes covers every joint class.
//!
//!  * **River** — exact scalar equity (1 bin), clustered by the exact 1-D DP.
//!    With `POKER_AI_RIVER_OCHS=1`, the river instead uses the exact
//!    [`OCHS_K`]-dim opponent-cluster-equity feature, clustered by L2 K-Means++
//!    (strictly better buckets at equal count — see `examples/bench_ochs.rs` —
//!    at a K× larger equity cache; the bucket map is unchanged size).
//!  * **Flop / turn** — exact equity-distribution histograms, clustered by L2
//!    K-Means++.
//!
//! **Scale.** Full coverage is the cloud burst (`cap = 0` on a big machine).
//! Locally, `cap` limits the boards processed, and any street whose flat cache
//! would exceed the memory budget is skipped (turn @ 50 bins = 2.8 GB, so the
//! 1.5 GB default skips it). On a bigger box raise the budget with
//! `POKER_AI_CLUSTER_MEM_GB` so the turn street builds too — e.g. on a 64 GB
//! server `POKER_AI_CLUSTER_MEM_GB=8 cluster 0` builds all three streets full.
//!
//! Usage:
//!   cluster [cap] [seed] [flop_k] [turn_k] [river_k] [--data=DIR]
//!     cap   max canonical boards to process per street; 0 = full  (default 300)
//!     seed  K-Means++ seed for flop/turn clustering                (default 1)
//!     *_k   bucket counts per street (defaults 1500/1500/3000)
//!     --data=DIR  output directory for the caches/maps (default `data/`)
//!   env:
//!     POKER_AI_CLUSTER_MEM_GB   per-street flat-cache budget in GB  (default 1.5)
//!     POKER_AI_RIVER_OCHS       1/true → OCHS river feature (default scalar)

use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::abstraction::equity_cache::EquityCache;
use poker_ai::abstraction::features::{
    board_equities, board_histograms, board_ochs, combo_index, ochs_opponent_clusters, OCHS_K,
};
use poker_ai::abstraction::hand_index::HandIndexer;

/// Histogram resolution for flop/turn (plan: 50-bin equity histograms).
const BINS: usize = 50;
/// Default per-street flat-cache budget; a street whose cache would exceed it is
/// skipped (the cloud burst's job). Override with `POKER_AI_CLUSTER_MEM_GB` —
/// raise it past 2.8 GB on a big box to build the turn street too.
const DEFAULT_MEM_BUDGET_BYTES: usize = 1_500_000_000;
/// Boards per parallel batch (bounds the merge buffer).
const BATCH_BOARDS: usize = 256;

/// Per-street flat-cache byte budget, from `POKER_AI_CLUSTER_MEM_GB` (GB, may be
/// fractional) or the 1.5 GB default.
fn mem_budget_bytes() -> usize {
    match std::env::var("POKER_AI_CLUSTER_MEM_GB").ok().and_then(|s| s.parse::<f64>().ok()) {
        Some(gb) if gb > 0.0 => (gb * 1e9) as usize,
        _ => DEFAULT_MEM_BUDGET_BYTES,
    }
}

/// Whether to build the river street with the OCHS opponent-cluster feature
/// (`OCHS_K`-dim, K-Means++) instead of the default scalar equity-vs-uniform
/// (1-bin, exact 1-D DP).  OCHS gives strictly better river buckets at the same
/// bucket count (see `examples/bench_ochs.rs`), but its equity *cache* is K×
/// larger (the river bucket map the solver loads is unchanged), so its flat
/// cache (≈ 3.9 GB at full coverage) needs `POKER_AI_CLUSTER_MEM_GB` raised —
/// the cloud burst.  Enabled by `POKER_AI_RIVER_OCHS` = `1`/`true`.
fn river_ochs_enabled() -> bool {
    matches!(std::env::var("POKER_AI_RIVER_OCHS").as_deref(), Ok("1") | Ok("true"))
}

/// Per-board features: `(joint slot, feature vector)` for every hole on `board`.
///
/// On the river, `ochs = Some(clusters)` produces the [`OCHS_K`]-dim
/// opponent-cluster-equity feature; `None` produces the scalar equity-vs-uniform
/// (1 bin).  Flop/turn always produce the `bins`-bucket equity histogram.
fn board_features(
    board: &[u8],
    river: bool,
    bins: usize,
    joint_ix: &HandIndexer,
    ochs: Option<&[u8; 169]>,
) -> Vec<(usize, Vec<f32>)> {
    let mut used = 0u64;
    for &c in board {
        used |= 1 << c;
    }
    let live: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();
    let mut out = Vec::with_capacity(live.len() * (live.len().saturating_sub(1)) / 2);

    let mut cards = [0u8; 7];
    cards[2..2 + board.len()].copy_from_slice(board);
    let mut river_buf = [f32::NAN; 1326];
    let mut ochs_buf = [[f32::NAN; OCHS_K]; 1326];
    let hists = if river { Vec::new() } else { board_histograms(board, bins) };
    if river {
        let b5: [u8; 5] = board.try_into().expect("river board is 5 cards");
        match ochs {
            Some(clusters) => board_ochs(b5, clusters, &mut ochs_buf),
            None => board_equities(b5, &mut river_buf),
        }
    }

    for i in 0..live.len() {
        let a = live[i];
        for &b in &live[i + 1..] {
            cards[0] = a;
            cards[1] = b;
            let slot = joint_ix.index(&cards[..2 + board.len()]) as usize;
            let ci = combo_index(a, b);
            let feat = if river {
                match ochs {
                    Some(_) => ochs_buf[ci].to_vec(),
                    None => vec![river_buf[ci]],
                }
            } else {
                hists[ci * bins..][..bins].to_vec()
            };
            out.push((slot, feat));
        }
    }
    out
}

/// Build (a prefix of) a street's flat cache by sweeping canonical boards, then
/// cluster and persist into `dir`.
fn build_street(name: &str, board_round: u8, k: usize, seed: u64, cap: usize, dir: &Path) {
    let river = board_round == 5;
    // River: scalar equity-vs-uniform (1 bin) by default, or the OCHS_K-dim
    // opponent-cluster feature when POKER_AI_RIVER_OCHS is set.
    let ochs = if river && river_ochs_enabled() { Some(ochs_opponent_clusters()) } else { None };
    let bins = match (river, ochs.is_some()) {
        (true, true) => OCHS_K,
        (true, false) => 1,
        (false, _) => BINS,
    };
    let joint_ix = HandIndexer::new(&[2, board_round]);
    let n_joint = joint_ix.size() as usize;

    let need = bins.saturating_mul(n_joint).saturating_mul(4);
    let budget = mem_budget_bytes();
    if need > budget {
        println!(
            "  [{name}] joint N={n_joint} → flat cache {:.1} GB exceeds budget {:.1} GB; \
             skipped (raise POKER_AI_CLUSTER_MEM_GB)",
            need as f64 / 1e9,
            budget as f64 / 1e9
        );
        return;
    }

    let board_ix = HandIndexer::new(&[board_round]);
    let n_boards = board_ix.size() as usize;
    let fill = if cap == 0 { n_boards } else { cap.min(n_boards) };
    let start = Instant::now();

    let mut data = vec![f32::NAN; bins * n_joint];
    let mut bi = 0;
    while bi < fill {
        let end = (bi + BATCH_BOARDS).min(fill);
        // Parallel per board; serial idempotent merge (different boards may map a
        // hole to the same joint slot, so a shared parallel write would race).
        let batch: Vec<Vec<(usize, Vec<f32>)>> = (bi..end)
            .into_par_iter()
            .map(|b| board_features(&board_ix.unindex(b as u64), river, bins, &joint_ix, ochs.as_ref()))
            .collect();
        for board_out in batch {
            for (slot, feat) in board_out {
                data[slot * bins..][..bins].copy_from_slice(&feat);
            }
        }
        bi = end;
    }

    let cache = EquityCache::from_parts(bins, &[2, board_round], data);
    println!(
        "  [{name}] swept {fill}/{n_boards} canonical boards → {} joint situations ({:.1}s)",
        cache.len(),
        start.elapsed().as_secs_f64()
    );

    let k = k.min(cache.len().max(1));
    let map = BucketMap::from_cache(&cache, k, seed);
    std::fs::create_dir_all(dir).expect("create data directory");
    cache.save(dir.join(format!("{name}_equity.bin"))).expect("save cache");
    let map_path = dir.join(format!("{name}_buckets.bin"));
    map.save(&map_path).expect("save map");
    println!(
        "  [{name}] {} buckets over {} situations -> {} ({:.1}s)\n",
        map.num_buckets(),
        map.len(),
        map_path.display(),
        start.elapsed().as_secs_f64()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir: PathBuf = args
        .iter()
        .find_map(|a| a.strip_prefix("--data="))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data"));
    // Positional args are the numeric ones; flags start with `--`.
    let nums: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    let cap: usize = nums.first().and_then(|s| s.parse().ok()).unwrap_or(300);
    let seed: u64 = nums.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    // Bucket targets: [flop] [turn] [river], same positional shape as
    // `memory_estimate` so the two tools stay in lockstep — check the
    // footprint there BEFORE spending the build time here.  Defaults bumped
    // from the original 500/500/800 plan minimum: river dominates node count
    // (~78%, Step 18/29 finding) so it gets the largest multiplier.
    let flop_buckets: usize = nums.get(2).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let turn_buckets: usize = nums.get(3).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let river_buckets: usize = nums.get(4).and_then(|s| s.parse().ok()).unwrap_or(3000);

    let cap_str = if cap == 0 { "full".to_string() } else { cap.to_string() };
    println!(
        "Building card abstraction (exact sweep; cap {cap_str} boards/street, seed {seed}, \
         mem budget {:.1} GB/street, buckets flop={flop_buckets} turn={turn_buckets} river={river_buckets})",
        mem_budget_bytes() as f64 / 1e9
    );
    build_street("flop", 3, flop_buckets, seed, cap, &dir);
    build_street("turn", 4, turn_buckets, seed, cap, &dir);
    build_street("river", 5, river_buckets, seed, cap, &dir);
    println!("Done. Load the *_buckets.bin maps into BlueprintHoldem::with_street_bucket.");
}
