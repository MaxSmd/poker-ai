//! Suit-isomorphic canonicalization for the equity cache.
//!
//! Poker has no inherent suit ordering: a hand and board are strategically
//! identical to any relabeling of their suits (there are no rules that prefer
//! spades to hearts).  Mapping every `(hole, board)` to a single canonical form
//! collapses the four-fold-or-more suit symmetry, so the most compute-bound
//! offline step — equity over millions of boards — is done once per *distinct*
//! situation rather than once per dealt situation.  This is the key that the
//! equity cache (Phase 2) is keyed on.
//!
//! The canonical form is the lexicographically smallest representation over all
//! 24 suit permutations, with hole and board kept distinguishable (which suit a
//! card sits in relative to the board matters; whether that suit is "spades" or
//! "hearts" does not).

use poker_core::{make_card, rank_of, suit_of};

/// The 24 permutations of the four suits `{0,1,2,3}`.
fn suit_permutations() -> [[u8; 4]; 24] {
    let mut perms = [[0u8; 4]; 24];
    let mut idx = 0;
    for a in 0..4u8 {
        for b in 0..4u8 {
            if b == a {
                continue;
            }
            for c in 0..4u8 {
                if c == a || c == b {
                    continue;
                }
                // The fourth suit is whatever's left.
                let d = 6 - a - b - c;
                perms[idx] = [a, b, c, d];
                idx += 1;
            }
        }
    }
    debug_assert_eq!(idx, 24);
    perms
}

/// Remap a card's suit through `perm` (rank unchanged).
fn remap(card: u8, perm: &[u8; 4]) -> u8 {
    make_card(rank_of(card), perm[suit_of(card) as usize])
}

/// Canonical key for a `(hole, board)` situation under suit isomorphism.
///
/// Two situations that differ only by a relabeling of suits produce the same
/// key; situations that differ in rank structure or in how suits align between
/// hole and board produce different keys.  Hole and board are each treated as
/// unordered sets (sorted) but kept in separate groups.
pub fn canonical_key(hole: &[u8], board: &[u8]) -> Vec<u8> {
    let mut best: Option<Vec<u8>> = None;
    for perm in suit_permutations() {
        let mut h: Vec<u8> = hole.iter().map(|&c| remap(c, &perm)).collect();
        h.sort_unstable();
        let mut b: Vec<u8> = board.iter().map(|&c| remap(c, &perm)).collect();
        b.sort_unstable();

        let mut key = Vec::with_capacity(h.len() + b.len() + 1);
        key.extend_from_slice(&h);
        key.push(0xFF); // keep hole and board groups distinguishable
        key.extend_from_slice(&b);

        if best.as_ref().is_none_or(|cur| key < *cur) {
            best = Some(key);
        }
    }
    best.expect("at least one permutation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::make_card;

    fn remap_all(cards: &[u8], perm: &[u8; 4]) -> Vec<u8> {
        cards.iter().map(|&c| remap(c, perm)).collect()
    }

    #[test]
    fn there_are_24_distinct_permutations() {
        let perms = suit_permutations();
        let mut seen = std::collections::HashSet::new();
        for p in perms {
            assert!(seen.insert(p), "permutation {p:?} repeated");
            // Each is a genuine permutation of {0,1,2,3}.
            let mut sorted = p;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2, 3]);
        }
    }

    #[test]
    fn key_is_invariant_under_suit_relabeling() {
        // A♠K♠ on a two-spade board, and the same hand/board with every suit
        // permutation applied, must all map to one canonical key.
        let hole = [make_card(12, 0), make_card(11, 0)];
        let board = [make_card(5, 0), make_card(9, 0), make_card(2, 1)];
        let base = canonical_key(&hole, &board);
        for perm in suit_permutations() {
            let h = remap_all(&hole, &perm);
            let b = remap_all(&board, &perm);
            assert_eq!(canonical_key(&h, &b), base, "perm {perm:?} changed the canonical key");
        }
    }

    #[test]
    fn distinct_suit_structures_differ() {
        // Suited hole cards vs offsuit hole cards on the same ranks must NOT
        // canonicalize together — the suit *relationship* is strategically real.
        let suited = [make_card(12, 0), make_card(11, 0)];
        let offsuit = [make_card(12, 0), make_card(11, 1)];
        let board = [make_card(5, 0), make_card(9, 2), make_card(2, 3)];
        assert_ne!(canonical_key(&suited, &board), canonical_key(&offsuit, &board));
    }

    #[test]
    fn equity_is_invariant_under_suit_relabeling() {
        // Cross-check: poker-core's evaluator is suit-blind, so equity must be
        // identical across suit permutations — the property that justifies
        // caching by canonical key.
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
        for perm in suit_permutations() {
            let h = remap_all(&hole, &perm);
            let b: Vec<u8> = remap_all(&board, &perm);
            let hh = [h[0], h[1]];
            let bb = [b[0], b[1], b[2], b[3], b[4]];
            assert!((river_equity(hh, bb) - base).abs() < 1e-12, "equity changed under {perm:?}");
        }
    }
}
