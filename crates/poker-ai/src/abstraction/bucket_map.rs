//! Information abstraction: canonical situation → bucket id.
//!
//! The bucket map is what the solver actually indexes: it collapses the
//! billions of `(hole, board)` situations into a few hundred-to-thousand
//! strategically-similar buckets per street.  It is produced by clustering the
//! [`EquityCache`] histograms ([`super::clustering`]) and then keyed by the
//! suit-isomorphic [`canonical_key`], so any dealt situation resolves to the
//! bucket of its canonical form in one lookup.
//!
//! The map is serialized so a training run loads it at startup rather than
//! re-running clustering, and so re-clustering experiments are reproducible.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::canonical::canonical_key;
use super::clustering::kmeans;
use super::equity_cache::EquityCache;

/// A serialized canonical-situation → bucket-id map.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BucketMap {
    num_buckets: u32,
    /// Canonical key bytes → bucket id.
    map: HashMap<Vec<u8>, u32>,
}

impl BucketMap {
    /// Build a bucket map by clustering every histogram in `cache` into `k`
    /// buckets (K-Means++ on L2, reproducible via `seed`).
    pub fn from_cache(cache: &EquityCache, k: usize, seed: u64) -> Self {
        let pairs: Vec<(&Vec<u8>, &Vec<f32>)> = cache.iter().collect();
        let data: Vec<Vec<f64>> =
            pairs.iter().map(|(_, h)| h.iter().map(|&x| x as f64).collect()).collect();

        let result = kmeans(&data, k, seed, 100);
        let num_buckets = result.centroids.len() as u32;
        let map = pairs
            .iter()
            .zip(result.assignments)
            .map(|((key, _), bucket)| ((*key).clone(), bucket as u32))
            .collect();
        Self { num_buckets, map }
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Number of distinct canonical situations mapped.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Bucket id for a `(hole, board)` situation (canonicalized on lookup), or
    /// `None` if the situation was not part of the clustered set.
    pub fn bucket(&self, hole: &[u8], board: &[u8]) -> Option<u32> {
        self.map.get(&canonical_key(hole, board)).copied()
    }

    /// Serialize to a bincode file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = bincode::serialize(self).map_err(io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Load from a bincode file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::{make_card, rank_of, suit_of};

    /// Build a small cache of synthetic histograms over real (distinct) hands on
    /// a fixed board so tests are fast and deterministic.
    fn synthetic_cache() -> EquityCache {
        let mut cache = EquityCache::new(4);
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

        // Rotate every suit; the canonical key — and bucket — must be unchanged.
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

    /// A real, end-to-end abstraction on a fixed river board: cluster a sample
    /// of hands by their actual equity and confirm a monster and air land in
    /// different buckets.  Ignored (computes real equities); run with:
    ///   cargo test -p poker-ai --release -- --ignored real_river_bucketing
    #[test]
    #[ignore]
    fn real_river_bucketing() {
        use crate::abstraction::features::ehs_histogram;
        // River board: A♣ K♦ 9♥ 4♠ 2♣.
        let board =
            [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        let mut used = 0u64;
        for &c in &board {
            used |= 1 << c;
        }
        let deck: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

        let mut cache = EquityCache::new(10);
        let mut counter = 0;
        for i in 0..deck.len() {
            for j in (i + 1)..deck.len() {
                counter += 1;
                if counter % 7 != 0 {
                    continue; // subsample to keep the test quick
                }
                let hole = [deck[i], deck[j]];
                let hist: Vec<f32> =
                    ehs_histogram(&hole, &board, 10).iter().map(|&x| x as f32).collect();
                cache.insert(&hole, &board, hist);
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
