//! Canonical equity cache — a **flat array** over the dense hand
//! index.
//!
//! Equity-distribution computation over millions of boards is the most
//! compute-bound offline step, and it is full of suit-isomorphic redundancy.
//! Keyed on the dense [`HandIndexer`], the cache is a flat `Vec<f32>`: slot
//! `index(hole, board)` holds that canonical situation's `bins`-bin histogram.
//! No hashing, no `Vec<u8>` keys, and the offline build fills the slots by
//! partitioning the index range (see `bin/cluster`) instead of merging per-board
//! maps.
//!
//! Unfilled slots carry a `NaN` sentinel, so a *partial* build (the local
//! flop/prefix pass) and a full one (the cloud burst) share the same structure.
//! Histograms are `f32` to halve the footprint; the clusterer promotes to `f64`.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::hand_index::HandIndexer;

/// A flat equity cache: one `bins`-bin histogram per dense canonical index.
pub struct EquityCache {
    bins: usize,
    indexer: HandIndexer,
    /// `bins × indexer.size()` values; slot `i` occupies `data[i*bins..][..bins]`.
    /// The first bin of an unfilled slot is `NaN`.
    data: Vec<f32>,
}

/// Serialized form (the indexer is rebuilt from `rounds` on load).
#[derive(Serialize, Deserialize)]
struct Persist {
    bins: usize,
    rounds: Vec<u8>,
    data: Vec<f32>,
}

impl EquityCache {
    /// An empty cache (all slots `NaN`) for the given round structure (e.g.
    /// `[2, 3]` for a flop situation).
    pub fn new(bins: usize, cards_per_round: &[u8]) -> Self {
        let indexer = HandIndexer::new(cards_per_round);
        let data = vec![f32::NAN; bins * indexer.size() as usize];
        Self { bins, indexer, data }
    }

    /// Wrap a pre-filled flat `data` array (the parallel build path).
    pub fn from_parts(bins: usize, cards_per_round: &[u8], data: Vec<f32>) -> Self {
        let indexer = HandIndexer::new(cards_per_round);
        debug_assert_eq!(data.len(), bins * indexer.size() as usize, "data length must be bins × size");
        Self { bins, indexer, data }
    }

    /// Histogram bin count.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// The dense indexer this cache is keyed on.
    pub fn indexer(&self) -> &HandIndexer {
        &self.indexer
    }

    /// Total slot capacity (= `indexer.size()`).
    pub fn capacity(&self) -> usize {
        self.indexer.size() as usize
    }

    /// Number of **filled** slots.
    pub fn len(&self) -> usize {
        (0..self.capacity()).filter(|&i| !self.data[i * self.bins].is_nan()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dense slot index for a situation.
    fn slot(&self, hole: &[u8], board: &[u8]) -> usize {
        let mut cards = [0u8; 7];
        let n = hole.len() + board.len();
        cards[..hole.len()].copy_from_slice(hole);
        cards[hole.len()..n].copy_from_slice(board);
        self.indexer.index(&cards[..n]) as usize
    }

    /// Histogram for a situation, or `None` if its slot is unfilled.
    pub fn get(&self, hole: &[u8], board: &[u8]) -> Option<&[f32]> {
        let i = self.slot(hole, board);
        let slot = &self.data[i * self.bins..][..self.bins];
        if slot[0].is_nan() {
            None
        } else {
            Some(slot)
        }
    }

    /// Write a histogram into a situation's slot.
    pub fn insert(&mut self, hole: &[u8], board: &[u8], histogram: Vec<f32>) {
        debug_assert_eq!(histogram.len(), self.bins, "histogram length must match `bins`");
        let i = self.slot(hole, board);
        self.data[i * self.bins..][..self.bins].copy_from_slice(&histogram);
    }

    /// Iterate `(slot index, histogram)` over the **filled** slots — the input to
    /// clustering.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f32])> {
        (0..self.capacity()).filter_map(move |i| {
            let slot = &self.data[i * self.bins..][..self.bins];
            (!slot[0].is_nan()).then_some((i, slot))
        })
    }

    /// Serialize to a bincode file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let persist = Persist {
            bins: self.bins,
            rounds: self.indexer.cards_per_round().to_vec(),
            data: self.data.clone(),
        };
        let bytes = bincode::serialize(&persist).map_err(io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Load from a bincode file (rebuilds the indexer from the stored rounds).
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let p: Persist = bincode::deserialize(&bytes).map_err(io::Error::other)?;
        Ok(Self::from_parts(p.bins, &p.rounds, p.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::{make_card, rank_of, suit_of};

    #[test]
    fn dedups_suit_isomorphic_situations() {
        let mut cache = EquityCache::new(10, &[2, 3]);
        let hole = [make_card(12, 0), make_card(11, 0)]; // A♠K♠
        let board = [make_card(5, 0), make_card(9, 1), make_card(2, 2)];
        cache.insert(&hole, &board, vec![0.1; 10]);

        // The same situation with suits rotated by one is isomorphic: it must
        // land in (and read from) the same canonical slot.
        let rot = |c: u8| make_card(rank_of(c), (suit_of(c) + 1) % 4);
        let hole2 = [rot(hole[0]), rot(hole[1])];
        let board2: Vec<u8> = board.iter().map(|&c| rot(c)).collect();

        assert_eq!(cache.len(), 1, "both situations share one canonical slot");
        assert!(cache.get(&hole2, &board2).is_some(), "isomorphic situation is a cache hit");
        assert_eq!(cache.get(&hole, &board), cache.get(&hole2, &board2));
    }

    #[test]
    fn save_load_round_trips() {
        let mut cache = EquityCache::new(8, &[2, 3]);
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
