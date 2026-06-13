//! Information abstraction: canonical situation → bucket id, as a **flat array**.
//!
//! The bucket map is what the solver indexes: it collapses the billions of
//! `(hole, board)` situations into a few hundred-to-thousand strategically
//! similar buckets per street.  Keyed on the dense [`HandIndexer`], it is a flat
//! `Vec<u16>` — `bucket(hole, board)` is one index computation plus one array
//! read, no hashing and no key storage.  The full river map is a single
//! `Vec<u16>` blob that loads in one read.
//!
//! It is produced by clustering the filled [`EquityCache`] slots
//! ([`super::clustering`]).  Unfilled slots (a partial/prefix build) carry the
//! [`UNASSIGNED`] sentinel, so [`bucket`](BucketMap::bucket) returns `None` there.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::clustering::{cluster_1d, kmeans};
use super::equity_cache::EquityCache;
use super::hand_index::HandIndexer;

/// Sentinel for a slot with no bucket (outside a partial/prefix build).
const UNASSIGNED: u16 = u16::MAX;

/// A flat dense-index → bucket-id map.
pub struct BucketMap {
    num_buckets: u32,
    indexer: HandIndexer,
    /// One bucket id per dense canonical index; [`UNASSIGNED`] where unbuilt.
    buckets: Vec<u16>,
}

/// Serialized form (the indexer is rebuilt from `rounds` on load).
#[derive(Serialize, Deserialize)]
struct Persist {
    num_buckets: u32,
    rounds: Vec<u8>,
    buckets: Vec<u16>,
}

impl BucketMap {
    /// Build a bucket map by clustering every **filled** slot in `cache` into `k`
    /// buckets, writing the assignments into a flat array indexed by the same
    /// dense index.
    ///
    /// A **scalar** feature (`bins == 1`, the river's equity) is clustered with
    /// the exact 1-D DP ([`cluster_1d`]) — deterministic and globally optimal,
    /// the right tool for a one-dimensional point cloud where k-means degenerates.
    /// A **histogram** feature (flop/turn) uses K-Means++ on L2 (`seed`).
    pub fn from_cache(cache: &EquityCache, k: usize, seed: u64) -> Self {
        if cache.bins() == 1 {
            return Self::from_scalar_cache(cache, k);
        }
        let filled: Vec<(usize, Vec<f64>)> =
            cache.iter().map(|(slot, h)| (slot, h.iter().map(|&x| x as f64).collect())).collect();
        let data: Vec<Vec<f64>> = filled.iter().map(|(_, h)| h.clone()).collect();

        let result = kmeans(&data, k, seed, 100);
        let mut buckets = vec![UNASSIGNED; cache.capacity()];
        for ((slot, _), b) in filled.iter().zip(result.assignments) {
            buckets[*slot] = b as u16;
        }
        Self {
            num_buckets: result.centroids.len() as u32,
            indexer: HandIndexer::new(cache.indexer().cards_per_round()),
            buckets,
        }
    }

    /// Scalar (1-bin) clustering via the exact 1-D DP over the distinct equity
    /// values (river equities take ≤ 991 distinct values, so this is cheap).
    fn from_scalar_cache(cache: &EquityCache, k: usize) -> Self {
        let filled: Vec<(usize, f64)> = cache.iter().map(|(slot, h)| (slot, h[0] as f64)).collect();

        // Distinct sorted values + their counts.
        let mut values: Vec<f64> = filled.iter().map(|&(_, e)| e).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        let lookup = |e: f64| values.partition_point(|&v| v < e);
        let mut weights = vec![0u64; values.len()];
        for &(_, e) in &filled {
            weights[lookup(e)] += 1;
        }

        let assign = cluster_1d(&values, &weights, k);
        let num_buckets = assign.iter().copied().max().map_or(0, |m| m + 1) as u32;
        let mut buckets = vec![UNASSIGNED; cache.capacity()];
        for &(slot, e) in &filled {
            buckets[slot] = assign[lookup(e)] as u16;
        }
        Self {
            num_buckets,
            indexer: HandIndexer::new(cache.indexer().cards_per_round()),
            buckets,
        }
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Number of assigned (built) slots.
    pub fn len(&self) -> usize {
        self.buckets.iter().filter(|&&b| b != UNASSIGNED).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bucket id for a `(hole, board)` situation, or `None` if its slot was not
    /// part of the built set.
    pub fn bucket(&self, hole: &[u8], board: &[u8]) -> Option<u32> {
        let mut cards = [0u8; 7];
        let n = hole.len() + board.len();
        cards[..hole.len()].copy_from_slice(hole);
        cards[hole.len()..n].copy_from_slice(board);
        let b = self.buckets[self.indexer.index(&cards[..n]) as usize];
        (b != UNASSIGNED).then_some(b as u32)
    }

    /// Serialize to a bincode file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let persist = Persist {
            num_buckets: self.num_buckets,
            rounds: self.indexer.cards_per_round().to_vec(),
            buckets: self.buckets.clone(),
        };
        let bytes = bincode::serialize(&persist).map_err(io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Load from a bincode file (rebuilds the indexer from the stored rounds).
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let p: Persist = bincode::deserialize(&bytes).map_err(io::Error::other)?;
        Ok(Self {
            num_buckets: p.num_buckets,
            indexer: HandIndexer::new(&p.rounds),
            buckets: p.buckets,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::{make_card, rank_of, suit_of};

    /// Build a small cache of synthetic histograms over real (distinct) hands on
    /// a fixed flop board so tests are fast and deterministic.
    fn synthetic_cache() -> EquityCache {
        let mut cache = EquityCache::new(4, &[2, 3]);
        let board = [make_card(11, 0), make_card(7, 1), make_card(2, 2)];
        // strong cluster
        cache.insert(&[make_card(12, 0), make_card(12, 1)], &board, vec![0.0, 0.0, 0.1, 0.9]);
        cache.insert(&[make_card(12, 2), make_card(12, 3)], &board, vec![0.0, 0.0, 0.15, 0.85]);
        // medium cluster
        cache.insert(&[make_card(9, 0), make_card(9, 1)], &board, vec![0.1, 0.8, 0.1, 0.0]);
        cache.insert(&[make_card(8, 0), make_card(8, 1)], &board, vec![0.15, 0.8, 0.05, 0.0]);
        // weak cluster
        cache.insert(&[make_card(5, 0), make_card(3, 1)], &board, vec![0.9, 0.1, 0.0, 0.0]);
        cache.insert(&[make_card(4, 0), make_card(3, 2)], &board, vec![0.85, 0.15, 0.0, 0.0]);
        cache
    }

    #[test]
    fn groups_similar_histograms_and_separates_different_ones() {
        let cache = synthetic_cache();
        let bm = BucketMap::from_cache(&cache, 3, 1);
        assert_eq!(bm.num_buckets(), 3);

        let board = [make_card(11, 0), make_card(7, 1), make_card(2, 2)];
        let strong_a = bm.bucket(&[make_card(12, 0), make_card(12, 1)], &board).unwrap();
        let strong_b = bm.bucket(&[make_card(12, 2), make_card(12, 3)], &board).unwrap();
        let weak = bm.bucket(&[make_card(5, 0), make_card(3, 1)], &board).unwrap();

        assert_eq!(strong_a, strong_b, "the two strong hands share a bucket");
        assert_ne!(strong_a, weak, "strong and weak hands are in different buckets");
    }

    #[test]
    fn lookup_is_suit_isomorphic() {
        let cache = synthetic_cache();
        let bm = BucketMap::from_cache(&cache, 3, 1);
        let board = [make_card(11, 0), make_card(7, 1), make_card(2, 2)];
        let hole = [make_card(12, 0), make_card(12, 1)];
        let direct = bm.bucket(&hole, &board).unwrap();

        // Rotate every suit; the dense index — and bucket — must be unchanged.
        let rot = |c: u8| make_card(rank_of(c), (suit_of(c) + 1) % 4);
        let hole2 = [rot(hole[0]), rot(hole[1])];
        let board2: Vec<u8> = board.iter().map(|&c| rot(c)).collect();
        assert_eq!(bm.bucket(&hole2, &board2), Some(direct));
    }

    #[test]
    fn unknown_situation_returns_none() {
        let cache = synthetic_cache();
        let bm = BucketMap::from_cache(&cache, 3, 1);
        let other_board = [make_card(0, 0), make_card(1, 1), make_card(3, 2)];
        assert!(bm.bucket(&[make_card(12, 0), make_card(12, 1)], &other_board).is_none());
    }

    #[test]
    fn save_load_round_trips() {
        let cache = synthetic_cache();
        let bm = BucketMap::from_cache(&cache, 3, 1);
        let path = std::env::temp_dir().join(format!("bucket_map_test_{}.bin", std::process::id()));
        bm.save(&path).unwrap();
        let loaded = BucketMap::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.num_buckets(), bm.num_buckets());
        assert_eq!(loaded.len(), bm.len());
        let board = [make_card(11, 0), make_card(7, 1), make_card(2, 2)];
        assert_eq!(
            loaded.bucket(&[make_card(12, 0), make_card(12, 1)], &board),
            bm.bucket(&[make_card(12, 0), make_card(12, 1)], &board)
        );
    }

    /// A real, end-to-end abstraction on a fixed river board: cluster a sample of
    /// hands by their actual equity and confirm a monster and air land in
    /// different buckets.  Ignored (computes real equities); run with:
    ///   cargo test -p poker-ai --release -- --ignored real_river_bucketing
    #[test]
    #[ignore]
    fn real_river_bucketing() {
        use crate::abstraction::features::river_equity;
        // River board: A♣ K♦ 9♥ 4♠ 2♣.
        let board =
            [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        let mut used = 0u64;
        for &c in &board {
            used |= 1 << c;
        }
        let deck: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

        // River clusters on scalar equity (1 bin).
        let mut cache = EquityCache::new(1, &[2, 5]);
        let mut counter = 0;
        for i in 0..deck.len() {
            for j in (i + 1)..deck.len() {
                counter += 1;
                if counter % 7 != 0 {
                    continue; // subsample to keep the test quick
                }
                let hole = [deck[i], deck[j]];
                cache.insert(&hole, &board, vec![river_equity(hole, board) as f32]);
            }
        }
        assert!(cache.len() > 20);

        let bm = BucketMap::from_cache(&cache, 8, 1);
        let strong = [make_card(12, 1), make_card(12, 2)]; // trip aces
        let weak = [make_card(5, 1), make_card(3, 2)];
        if let (Some(s), Some(w)) = (bm.bucket(&strong, &board), bm.bucket(&weak, &board)) {
            assert_ne!(s, w, "a monster and air must not share a river bucket");
        }
    }
}
