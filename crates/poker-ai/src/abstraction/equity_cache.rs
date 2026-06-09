//! Canonical equity cache (Phase 2).
//!
//! Equity-distribution computation over millions of boards is the most
//! compute-bound offline step in the whole project, and it is embarrassingly
//! parallel and full of redundancy: every dealt situation is suit-isomorphic to
//! many others.  The cache does the work **once per canonical situation**
//! ([`super::canonical`]) and serializes the result, so re-clustering
//! experiments (different bucket counts, turn-conditioned-on-flop passes) read
//! histograms back instead of recomputing rollouts.
//!
//! Histograms are stored as `f32` to halve the on-disk and in-RAM footprint;
//! they are promoted to `f64` only when handed to the clusterer.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::canonical::canonical_key;
use super::features::ehs_histogram;

/// A map from canonical `(hole, board)` key to its equity-distribution
/// histogram.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EquityCache {
    /// Number of histogram bins every entry uses.
    bins: usize,
    /// Canonical key bytes → histogram.
    entries: HashMap<Vec<u8>, Vec<f32>>,
}

impl EquityCache {
    /// Create an empty cache for `bins`-bin histograms.
    pub fn new(bins: usize) -> Self {
        Self { bins, entries: HashMap::new() }
    }

    /// Histogram bin count.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Number of distinct (canonical) situations stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the histogram for a situation (canonicalized on the way in).
    pub fn get(&self, hole: &[u8], board: &[u8]) -> Option<&Vec<f32>> {
        self.entries.get(&canonical_key(hole, board))
    }

    /// Insert a pre-computed histogram under a situation's canonical key.
    pub fn insert(&mut self, hole: &[u8], board: &[u8], histogram: Vec<f32>) {
        debug_assert_eq!(histogram.len(), self.bins, "histogram length must match `bins`");
        self.entries.insert(canonical_key(hole, board), histogram);
    }

    /// Compute the equity histogram for a situation and store it — but only if
    /// its canonical key is not already present.  Returns `true` if work was
    /// done, `false` if the canonical situation was already cached.  This is the
    /// dedup that turns an O(dealt situations) job into O(canonical situations).
    pub fn compute_if_absent(&mut self, hole: &[u8; 2], board: &[u8]) -> bool {
        let bins = self.bins;
        self.compute_if_absent_with(hole, board, || {
            ehs_histogram(hole, board, bins).iter().map(|&x| x as f32).collect()
        })
    }

    /// Like [`compute_if_absent`](Self::compute_if_absent) but the histogram is
    /// produced by `make`, called only on a miss.  Lets the caller choose the
    /// feature computation — e.g. a Monte-Carlo rollout for the flop and turn,
    /// where exact enumeration is too slow — while keeping the canonical dedup.
    pub fn compute_if_absent_with(
        &mut self,
        hole: &[u8; 2],
        board: &[u8],
        make: impl FnOnce() -> Vec<f32>,
    ) -> bool {
        let key = canonical_key(hole, board);
        if self.entries.contains_key(&key) {
            return false;
        }
        let hist = make();
        debug_assert_eq!(hist.len(), self.bins, "histogram length must match `bins`");
        self.entries.insert(key, hist);
        true
    }

    /// Iterate `(canonical key, histogram)` pairs — the input to clustering.
    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &Vec<f32>)> {
        self.entries.iter()
    }

    /// Fold another cache's entries into this one (dedup by canonical key).  Used
    /// to merge per-board caches built in parallel; merging in a fixed board
    /// order keeps the result deterministic regardless of which thread built
    /// which board.
    pub fn merge(&mut self, other: EquityCache) {
        debug_assert_eq!(self.bins, other.bins, "merging caches must share bin count");
        for (key, hist) in other.entries {
            self.entries.entry(key).or_insert(hist);
        }
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

    #[test]
    fn dedups_suit_isomorphic_situations() {
        let mut cache = EquityCache::new(10);
        let hole = [make_card(12, 0), make_card(11, 0)]; // A♠K♠
        let board = [make_card(5, 0), make_card(9, 1), make_card(2, 2)];

        assert!(cache.compute_if_absent(&hole, &board), "first compute does work");

        // The same situation with suits rotated by one is isomorphic.
        let rot = |c: u8| make_card(rank_of(c), (suit_of(c) + 1) % 4);
        let hole2 = [rot(hole[0]), rot(hole[1])];
        let board2: Vec<u8> = board.iter().map(|&c| rot(c)).collect();
        assert!(!cache.compute_if_absent(&hole2, &board2), "isomorphic situation is a cache hit");

        assert_eq!(cache.len(), 1, "both situations share one canonical entry");
        // And the histogram is retrievable through either representation.
        assert!(cache.get(&hole, &board).is_some());
        assert_eq!(cache.get(&hole, &board), cache.get(&hole2, &board2));
    }

    #[test]
    fn save_load_round_trips() {
        let mut cache = EquityCache::new(8);
        cache.insert(
            &[make_card(12, 0), make_card(12, 1)],
            &[make_card(11, 0), make_card(7, 1), make_card(2, 2)],
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0],
        );
        let dir = std::env::temp_dir();
        let path = dir.join(format!("equity_cache_test_{}.bin", std::process::id()));
        cache.save(&path).unwrap();
        let loaded = EquityCache::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.bins(), 8);
        assert_eq!(loaded.len(), 1);
        let h = loaded
            .get(&[make_card(12, 0), make_card(12, 1)], &[make_card(11, 0), make_card(7, 1), make_card(2, 2)])
            .expect("entry survives round-trip");
        assert!((h[2] - 0.2).abs() < 1e-6);
    }
}
