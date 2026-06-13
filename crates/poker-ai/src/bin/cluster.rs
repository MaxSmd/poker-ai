//! Offline information-abstraction builder (Phase 2).
//!
//! Built **per board** via the equity sweep ([`board_equities`] /
//! [`board_histograms`]): one board scores all ~1081 holes at once in
//! O(n log n), so the feature for every situation is **exact and ~1000× cheaper**
//! than the old per-hand O(n²) enumeration / Monte-Carlo rollout (exact, low-noise
//! histograms cluster better — the plan names sampling noise as the ceiling).
//!
//! Boards are enumerated by a suit-isomorphic board indexer and processed in
//! parallel; each board emits `(joint slot, feature)` entries that are merged
//! serially into the flat [`EquityCache`] keyed by finding #2's joint
//! `(hole, board)` index.  Writes to a joint slot from different raw boards are
//! idempotent (suit-iso ⇒ identical feature), and iterating canonical boards ×
//! all holes covers every joint class.
//!
//!  * **River** — exact scalar equity (1 bin), clustered by the exact 1-D DP.
//!  * **Flop / turn** — exact equity-distribution histograms, clustered by L2
//!    K-Means++.
//!
//! **Scale.** Full coverage is the cloud burst (`cap = 0` on a big machine).
//! Locally, `cap` limits the boards processed, and any street whose flat cache
//! would exceed the memory budget is skipped (turn @ 50 bins = 2.8 GB).
//!
//! Usage:
//!   cluster [cap] [seed]
//!     cap   max canonical boards to process per street; 0 = full  (default 300)
//!     seed  K-Means++ seed for flop/turn clustering                (default 1)

use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::abstraction::equity_cache::EquityCache;
use poker_ai::abstraction::features::{board_equities, board_histograms, combo_index};
use poker_ai::abstraction::hand_index::HandIndexer;

/// Histogram resolution for flop/turn (plan: 50-bin equity histograms).
const BINS: usize = 50;
/// Skip a street whose flat cache would exceed this — the cloud burst's job.
const MEM_BUDGET_BYTES: usize = 1_500_000_000;
/// Boards per parallel batch (bounds the merge buffer).
const BATCH_BOARDS: usize = 256;

/// Per-board features: `(joint slot, feature vector)` for every hole on `board`.
fn board_features(
    board: &[u8],
    river: bool,
    bins: usize,
    joint_ix: &HandIndexer,
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
    let hists = if river { Vec::new() } else { board_histograms(board, bins) };
    if river {
        board_equities(board.try_into().expect("river board is 5 cards"), &mut river_buf);
    }

    for i in 0..live.len() {
        let a = live[i];
        for &b in &live[i + 1..] {
            cards[0] = a;
            cards[1] = b;
            let slot = joint_ix.index(&cards[..2 + board.len()]) as usize;
            let ci = combo_index(a, b);
            let feat = if river {
                vec![river_buf[ci]]
            } else {
                hists[ci * bins..][..bins].to_vec()
            };
            out.push((slot, feat));
        }
    }
    out
}

/// Build (a prefix of) a street's flat cache by sweeping canonical boards, then
/// cluster and persist.
fn build_street(name: &str, board_round: u8, k: usize, seed: u64, cap: usize) {
    let river = board_round == 5;
    let bins = if river { 1 } else { BINS };
    let joint_ix = HandIndexer::new(&[2, board_round]);
    let n_joint = joint_ix.size() as usize;

    let need = bins.saturating_mul(n_joint).saturating_mul(4);
    if need > MEM_BUDGET_BYTES {
        println!(
            "  [{name}] joint N={n_joint} → flat cache {:.1} GB exceeds budget; skipped (cloud burst)",
            need as f64 / 1e9
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
            .map(|b| board_features(&board_ix.unindex(b as u64), river, bins, &joint_ix))
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
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cap: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let cap_str = if cap == 0 { "full".to_string() } else { cap.to_string() };
    println!("Building card abstraction (exact sweep; cap {cap_str} boards/street, seed {seed})");
    // Plan bucket targets: flop/turn 500–800, river 800–1200.  Capped to the
    // number of filled situations at small scales.
    build_street("flop", 3, 500, seed, cap);
    build_street("turn", 4, 500, seed, cap);
    build_street("river", 5, 800, seed, cap);
    println!("Done. Load the *_buckets.bin maps into BlueprintHoldem::with_street_bucket.");
}
