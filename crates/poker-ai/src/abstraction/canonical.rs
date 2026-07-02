//! Suit-isomorphic canonicalization.
//!
//! Poker has no inherent suit ordering: a hand and board are strategically
//! identical to any relabeling of their suits.  Mapping every `(hole, board)` to
//! a single canonical form collapses that symmetry so the compute-bound equity
//! work is done once per *distinct* situation.
//!
//! [`canonical_key`] is the lexicographically-smallest packed representation over
//! all 24 suit permutations (hole and board kept as separate sorted groups).  It
//! is **allocation-free** — a `const` permutation table and fixed stack buffers,
//! returning a packed `u64` — and serves as the correctness oracle for the dense
//! [`HandIndexer`](super::hand_index::HandIndexer) that production code keys on.
//!
//! [`preflop_index`] is the direct 169-class map for the pre-flop hot path: a
//! handful of comparisons, no permutation loop.

use poker_core::{make_card, rank_of, suit_of};

/// The 24 permutations of the four suits `{0,1,2,3}` (built at compile time).
const SUIT_PERMS: [[u8; 4]; 24] = build_suit_perms();

const fn build_suit_perms() -> [[u8; 4]; 24] {
    let mut perms = [[0u8; 4]; 24];
    let mut idx = 0;
    let mut a = 0u8;
    while a < 4 {
        let mut b = 0u8;
        while b < 4 {
            if b != a {
                let mut c = 0u8;
                while c < 4 {
                    if c != a && c != b {
                        perms[idx] = [a, b, c, 6 - a - b - c];
                        idx += 1;
                    }
                    c += 1;
                }
            }
            b += 1;
        }
        a += 1;
    }
    perms
}

/// Remap a card's suit through `perm` (rank unchanged).
fn remap(card: u8, perm: &[u8; 4]) -> u8 {
    make_card(rank_of(card), perm[suit_of(card) as usize])
}

/// Pack a sorted hole group (≤ 2 cards) and sorted board group (≤ 5 cards) into a
/// `u64`, hole in the high bits so the two groups stay distinguishable and the
/// packed value orders the same way the byte sequence would.  Absent slots use
/// the `0x3F` sentinel (no real card reaches it).
fn pack(hole: &[u8], board: &[u8]) -> u64 {
    let mut key = 0u64;
    for k in 0..2 {
        key = (key << 6) | hole.get(k).map_or(0x3F, |&c| c as u64);
    }
    for k in 0..5 {
        key = (key << 6) | board.get(k).map_or(0x3F, |&c| c as u64);
    }
    key
}

/// Canonical key for a `(hole, board)` situation under suit isomorphism.
///
/// Two situations that differ only by a relabeling of suits produce the same
/// key; situations differing in rank structure or in how suits align between hole
/// and board produce different keys.  Allocation-free.
pub fn canonical_key(hole: &[u8], board: &[u8]) -> u64 {
    debug_assert!(hole.len() <= 2 && board.len() <= 5, "canonical_key sizes");
    let mut best = u64::MAX;
    let mut hbuf = [0u8; 2];
    let mut bbuf = [0u8; 5];
    for perm in &SUIT_PERMS {
        for (i, &c) in hole.iter().enumerate() {
            hbuf[i] = remap(c, perm);
        }
        hbuf[..hole.len()].sort_unstable();
        for (i, &c) in board.iter().enumerate() {
            bbuf[i] = remap(c, perm);
        }
        bbuf[..board.len()].sort_unstable();
        let key = pack(&hbuf[..hole.len()], &bbuf[..board.len()]);
        if key < best {
            best = key;
        }
    }
    best
}

/// Direct 169-class pre-flop index in `0..169` — three comparisons, no
/// permutation loop.  Layout: `0..13` pocket pairs (by rank), `13..91` suited,
/// `91..169` offsuit, where a non-pair `(hi, lo)` ranks at `hi·(hi−1)/2 + lo`.
pub fn preflop_index(hole: &[u8; 2]) -> u16 {
    let (r0, r1) = (rank_of(hole[0]) as u16, rank_of(hole[1]) as u16);
    let suited = suit_of(hole[0]) == suit_of(hole[1]);
    let (hi, lo) = if r0 >= r1 { (r0, r1) } else { (r1, r0) };
    if hi == lo {
        hi // pocket pair
    } else {
        let combo = hi * (hi - 1) / 2 + lo; // 0..78
        13 + if suited { combo } else { 78 + combo }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::hand_index::HandIndexer;
    use poker_core::make_card;

    fn remap_all(cards: &[u8], perm: &[u8; 4]) -> Vec<u8> {
        cards.iter().map(|&c| remap(c, perm)).collect()
    }

    #[test]
    fn there_are_24_distinct_permutations() {
        let mut seen = std::collections::HashSet::new();
        for p in SUIT_PERMS {
            assert!(seen.insert(p), "permutation {p:?} repeated");
            let mut sorted = p;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2, 3]);
        }
    }

    #[test]
    fn key_is_invariant_under_suit_relabeling() {
        let hole = [make_card(12, 0), make_card(11, 0)];
        let board = [make_card(5, 0), make_card(9, 0), make_card(2, 1)];
        let base = canonical_key(&hole, &board);
        for perm in SUIT_PERMS {
            let h = remap_all(&hole, &perm);
            let b = remap_all(&board, &perm);
            assert_eq!(canonical_key(&h, &b), base, "perm {perm:?} changed the canonical key");
        }
    }

    #[test]
    fn distinct_suit_structures_differ() {
        let suited = [make_card(12, 0), make_card(11, 0)];
        let offsuit = [make_card(12, 0), make_card(11, 1)];
        let board = [make_card(5, 0), make_card(9, 2), make_card(2, 3)];
        assert_ne!(canonical_key(&suited, &board), canonical_key(&offsuit, &board));
    }

    #[test]
    fn preflop_index_is_a_169_class_bijection() {
        // Dense over 0..169, and partitions hands exactly as the indexer does.
        let ix = HandIndexer::new(&[2]);
        let mut seen = [false; 169];
        let mut pairs: Vec<(u16, u64)> = Vec::new();
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let lut = preflop_index(&[a, b]);
                assert!(lut < 169);
                seen[lut as usize] = true;
                pairs.push((lut, ix.index(&[a, b])));
            }
        }
        assert!(seen.iter().all(|&s| s), "all 169 classes reached");
        // Same equivalence classes as the dense indexer: same LUT ⟺ same index.
        for i in 0..pairs.len() {
            for j in (i + 1)..pairs.len() {
                assert_eq!(
                    pairs[i].0 == pairs[j].0,
                    pairs[i].1 == pairs[j].1,
                    "preflop LUT and indexer disagree on a class boundary"
                );
            }
        }
    }

    #[test]
    fn equity_is_invariant_under_suit_relabeling() {
        use crate::abstraction::features::river_equity;
        let hole = [make_card(12, 0), make_card(12, 1)];
        let board = [
            make_card(11, 0),
            make_card(9, 0),
            make_card(2, 1),
            make_card(5, 2),
            make_card(7, 3),
        ];
        let base = river_equity(hole, board);
        for perm in SUIT_PERMS {
            let h = remap_all(&hole, &perm);
            let b: Vec<u8> = remap_all(&board, &perm);
            let hh = [h[0], h[1]];
            let bb = [b[0], b[1], b[2], b[3], b[4]];
            assert!((river_equity(hh, bb) - base).abs() < 1e-12, "equity changed under {perm:?}");
        }
    }
}
