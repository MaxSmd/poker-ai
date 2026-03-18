//! Hand strength evaluation using a fast algorithmic evaluator.
//!
//! Cards are encoded as `rank * 4 + suit` where:
//!   - rank: 0 = 2, 1 = 3, …, 8 = T, 9 = J, 10 = Q, 11 = K, 12 = A
//!   - suit: 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades
//!
//! Hand strength is returned as a `u32` where **higher is better**.
//! Encoding:  bits 23-20 = category (0–8), bits 19-0 = tiebreaker ranks.
//! Category codes:
//!   8 = Straight Flush, 7 = Four of a Kind, 6 = Full House, 5 = Flush,
//!   4 = Straight,       3 = Three of a Kind, 2 = Two Pair, 1 = One Pair,
//!   0 = High Card.

/// Extract the rank from a card byte (0 = 2, …, 12 = Ace).
#[inline]
pub fn rank_of(card: u8) -> u8 {
    card >> 2
}

/// Extract the suit from a card byte (0–3).
#[inline]
pub fn suit_of(card: u8) -> u8 {
    card & 3
}

/// Build a card byte from rank and suit.
#[inline]
pub fn make_card(rank: u8, suit: u8) -> u8 {
    (rank << 2) | suit
}

/// Pack category and up-to-5 tiebreaker ranks into a comparable `u32`.
/// `cat` is 0–8; each `rN` is a rank 0–12 stored in a 4-bit nibble.
#[inline]
fn make_hand(cat: u8, r1: u8, r2: u8, r3: u8, r4: u8, r5: u8) -> u32 {
    ((cat as u32) << 20)
        | ((r1 as u32) << 16)
        | ((r2 as u32) << 12)
        | ((r3 as u32) << 8)
        | ((r4 as u32) << 4)
        | (r5 as u32)
}

/// Evaluate a 5-card hand.  Returns a rank where higher is better.
///
/// No heap allocation.  Average case: O(1) after constant-time rank counting.
pub fn evaluate_5(cards: &[u8; 5]) -> u32 {
    // Extract ranks and suits into stack arrays.
    let mut ranks = [0u8; 5];
    let mut suits = [0u8; 5];
    for i in 0..5 {
        ranks[i] = rank_of(cards[i]);
        suits[i] = suit_of(cards[i]);
    }

    // --- flush test ---
    let is_flush = suits[0] == suits[1]
        && suits[1] == suits[2]
        && suits[2] == suits[3]
        && suits[3] == suits[4];

    // Sort ranks descending.
    ranks.sort_unstable_by(|a, b| b.cmp(a));

    // --- frequency table ---
    let mut freq = [0u8; 13];
    for &r in &ranks {
        freq[r as usize] += 1;
    }

    // --- straight test ---
    // A normal straight: top - bottom == 4 and all ranks distinct.
    let all_distinct = freq.iter().all(|&f| f <= 1);
    let is_normal_straight = all_distinct && (ranks[0] - ranks[4] == 4);
    // Wheel: A-2-3-4-5 — ranks sorted desc = [12, 3, 2, 1, 0]
    let is_wheel =
        ranks[0] == 12 && ranks[1] == 3 && ranks[2] == 2 && ranks[3] == 1 && ranks[4] == 0;
    let is_straight = is_normal_straight || is_wheel;
    // For the wheel the "effective top" is 3 (five-high).
    let straight_top = if is_wheel { 3 } else { ranks[0] };

    // --- frequency-based hand components ---
    let mut quad_rank = 0u8;
    let mut trips_rank = 0u8;
    let mut pair1 = 0u8; // highest pair rank
    let mut pair2 = 0u8; // second-highest pair rank
    let mut num_pairs: u8 = 0;
    let mut has_quads = false;
    let mut has_trips = false;

    for r in (0..13u8).rev() {
        match freq[r as usize] {
            4 => {
                quad_rank = r;
                has_quads = true;
            }
            3 => {
                trips_rank = r;
                has_trips = true;
            }
            2 => {
                if num_pairs == 0 {
                    pair1 = r;
                } else {
                    pair2 = r;
                }
                num_pairs += 1;
            }
            _ => {}
        }
    }

    // --- classify ---
    if is_straight && is_flush {
        make_hand(8, straight_top, 0, 0, 0, 0)
    } else if has_quads {
        // find kicker: first rank != quad_rank
        let kicker = ranks
            .iter()
            .find(|&&r| r != quad_rank)
            .copied()
            .unwrap_or(0);
        make_hand(7, quad_rank, kicker, 0, 0, 0)
    } else if has_trips && num_pairs >= 1 {
        make_hand(6, trips_rank, pair1, 0, 0, 0)
    } else if is_flush {
        make_hand(5, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4])
    } else if is_straight {
        make_hand(4, straight_top, 0, 0, 0, 0)
    } else if has_trips {
        // two kickers, skipping the trips rank
        let mut k = [0u8; 2];
        let mut ki = 0usize;
        for &r in &ranks {
            if r != trips_rank && ki < 2 {
                k[ki] = r;
                ki += 1;
            }
        }
        make_hand(3, trips_rank, k[0], k[1], 0, 0)
    } else if num_pairs >= 2 {
        let mut kicker = 0u8;
        for &r in &ranks {
            if r != pair1 && r != pair2 {
                kicker = r;
                break;
            }
        }
        make_hand(2, pair1, pair2, kicker, 0, 0)
    } else if num_pairs == 1 {
        let mut k = [0u8; 3];
        let mut ki = 0usize;
        for &r in &ranks {
            if r != pair1 && ki < 3 {
                k[ki] = r;
                ki += 1;
            }
        }
        make_hand(1, pair1, k[0], k[1], k[2], 0)
    } else {
        make_hand(0, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4])
    }
}

/// Evaluate the best 5-card hand from 6 cards (e.g., turn + hole cards).
/// Tries all C(6,5) = 6 subsets.  No heap allocation.
pub fn evaluate_6(cards: &[u8; 6]) -> u32 {
    let mut best = 0u32;
    for skip in 0..6usize {
        let mut hand = [0u8; 5];
        let mut idx = 0;
        for (k, &card) in cards.iter().enumerate() {
            if k != skip {
                hand[idx] = card;
                idx += 1;
            }
        }
        let rank = evaluate_5(&hand);
        if rank > best {
            best = rank;
        }
    }
    best
}

/// Evaluate the strength of a 7-card hand. Returns a rank where higher is better.
/// Tries all C(7,5) = 21 subsets.  No heap allocation.
pub fn evaluate_7(cards: &[u8; 7]) -> u32 {
    let mut best = 0u32;
    for i in 0..7usize {
        for j in (i + 1)..7usize {
            let mut hand = [0u8; 5];
            let mut idx = 0;
            for (k, &card) in cards.iter().enumerate() {
                if k != i && k != j {
                    hand[idx] = card;
                    idx += 1;
                }
            }
            let rank = evaluate_5(&hand);
            if rank > best {
                best = rank;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> u8 {
        make_card(rank, suit)
    }

    #[test]
    fn high_card_less_than_pair() {
        // High card: A K Q J 9
        let hc = evaluate_5(&[c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(7, 0)]);
        // One pair: A A K Q J
        let pair = evaluate_5(&[c(12, 0), c(12, 1), c(11, 2), c(10, 3), c(9, 0)]);
        assert!(hc < pair);
    }

    #[test]
    fn straight_flush_beats_quads() {
        // Straight flush 9-T-J-Q-K spades
        let sf = evaluate_5(&[c(7, 3), c(8, 3), c(9, 3), c(10, 3), c(11, 3)]);
        // Four aces + king
        let quads = evaluate_5(&[c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(11, 0)]);
        assert!(sf > quads);
    }

    #[test]
    fn wheel_straight_is_five_high() {
        // A-2-3-4-5 (wheel) – top rank = 3 (five)
        let wheel = evaluate_5(&[c(12, 0), c(0, 1), c(1, 2), c(2, 3), c(3, 0)]);
        // 2-3-4-5-6 straight – top rank = 4 (six)
        let six_high = evaluate_5(&[c(0, 0), c(1, 1), c(2, 2), c(3, 3), c(4, 0)]);
        assert!(wheel < six_high, "wheel {wheel:#x} should be < 6-high {six_high:#x}");
    }

    #[test]
    fn full_house_beats_flush() {
        // Full house: K-K-K-2-2
        let fh = evaluate_5(&[c(11, 0), c(11, 1), c(11, 2), c(0, 0), c(0, 1)]);
        // Flush: A-K-Q-J-9 hearts
        let flush = evaluate_5(&[c(12, 2), c(11, 2), c(10, 2), c(9, 2), c(7, 2)]);
        assert!(fh > flush);
    }

    #[test]
    fn evaluate_7_finds_best_hand() {
        // 7 cards containing a straight flush in spades (T-J-Q-K-A)
        // plus two unrelated cards
        let cards: [u8; 7] = [
            c(8, 3),  // T spades
            c(9, 3),  // J spades
            c(10, 3), // Q spades
            c(11, 3), // K spades
            c(12, 3), // A spades
            c(0, 0),  // 2 clubs
            c(1, 1),  // 3 diamonds
        ];
        let rank = evaluate_7(&cards);
        // Category should be 8 (straight flush)
        assert_eq!(rank >> 20, 8, "expected straight flush category");
    }

    #[test]
    fn two_pair_vs_two_pair_kicker() {
        // A-A-K-K-Q vs A-A-K-K-J — queen kicker beats jack kicker
        let high_kicker = evaluate_5(&[c(12, 0), c(12, 1), c(11, 0), c(11, 1), c(10, 0)]);
        let low_kicker = evaluate_5(&[c(12, 0), c(12, 1), c(11, 0), c(11, 1), c(9, 0)]);
        assert!(high_kicker > low_kicker);
    }
}
