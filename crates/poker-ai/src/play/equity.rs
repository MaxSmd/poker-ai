//! All-in equity of a concrete hand against a belief range.
//!
//! The playing agent needs this exactly when the abstract tracker has desynced:
//! the opponent raised past the blueprint's raise cap, the abstraction has no
//! node for the spot, and a blueprint lookup is therefore impossible.  Without
//! it the agent has nothing to reason with and must guess.
//!
//! The value returned is *all-in* equity — it assumes the board runs out with
//! no further betting.  That understates the value of calling with a drawing
//! hand that could win more later, but a desync only happens deep in a raise
//! war, where the bet faced is large relative to the stacks behind and the
//! all-in approximation is close.
//!
//! Reach vectors use the crate-wide combo ordering (`util::combos`, shared by
//! [`board_cfvs`] and [`BeliefState`]).

use poker_core::state::NO_CARD;

use crate::abstraction::features::{board_cfvs, combo_cards, combo_index};
use crate::resolving::belief_state::{BeliefState, NUM_COMBOS};

/// Runouts drawn when the remaining board is too wide to enumerate (preflop).
/// Each sample already averages over the opponent's whole range, so the
/// residual noise is in the runout only and 200 draws put it well under 1%.
const PREFLOP_SAMPLES: usize = 200;

/// xorshift64* uniform in `[0, 1)`, matching the agent's sampler.
fn unit(rng: &mut u64) -> f64 {
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
}

/// Equity of `hole` against `opp`'s range on `board` (0, 3, 4, or 5 cards),
/// in `[0, 1]`, ties counted as a half win.
///
/// Card removal is handled by [`board_cfvs`]: opponent combos sharing a card
/// with the completed board or with `hole` are dropped from the denominator.
pub fn equity_vs_range(hole: [u8; 2], board: &[u8], opp: &BeliefState, rng: &mut u64) -> f64 {
    let mut reach = [0.0f64; NUM_COMBOS];
    for (i, slot) in reach.iter_mut().enumerate() {
        let [a, b] = combo_cards(i);
        if a != hole[0] && a != hole[1] && b != hole[0] && b != hole[1] {
            *slot = opp.prob(a, b);
        }
    }

    let mut used = 1u64 << hole[0] | 1u64 << hole[1];
    for &c in board {
        if c != NO_CARD {
            used |= 1 << c;
        }
    }
    let deck: Vec<u8> = (0u8..52).filter(|c| used & (1 << c) == 0).collect();

    let hero = combo_index(hole[0], hole[1]);
    let combos: Vec<[u8; 2]> = (0..NUM_COMBOS).map(combo_cards).collect();
    let mut out = [0.0f64; NUM_COMBOS];
    let mut full = [NO_CARD; 5];
    full[..board.len()].copy_from_slice(board);

    // `board_cfvs` yields a reach-weighted counterfactual value, not a
    // normalized equity: it sums over surviving opponent combos without
    // dividing by their mass.  Dividing here turns it into (P(win) - P(lose))
    // in [-1, 1], which maps to equity as (v + 1) / 2.
    let eval = |full: [u8; 5], out: &mut [f64; NUM_COMBOS]| -> f64 {
        board_cfvs(full, &reach, 1.0, out);
        let mut used = 0u64;
        for &c in &full {
            used |= 1 << c;
        }
        let mut mass = 0.0;
        for (i, &[a, b]) in combos.iter().enumerate() {
            if used & (1 << a) == 0 && used & (1 << b) == 0 {
                mass += reach[i];
            }
        }
        if mass > 0.0 {
            out[hero] / mass
        } else {
            0.0
        }
    };

    let mut sum = 0.0;
    let mut n = 0.0;
    match board.len() {
        5 => {
            sum += eval(full, &mut out);
            n += 1.0;
        }
        4 => {
            for &c in &deck {
                full[4] = c;
                sum += eval(full, &mut out);
                n += 1.0;
            }
        }
        3 => {
            for i in 0..deck.len() {
                for j in i + 1..deck.len() {
                    full[3] = deck[i];
                    full[4] = deck[j];
                    sum += eval(full, &mut out);
                    n += 1.0;
                }
            }
        }
        0 => {
            let mut pick = [0u8; 5];
            for _ in 0..PREFLOP_SAMPLES {
                let mut avail = deck.clone();
                for slot in pick.iter_mut() {
                    let k = (unit(rng) * avail.len() as f64) as usize;
                    *slot = avail.swap_remove(k.min(avail.len() - 1));
                }
                full.copy_from_slice(&pick);
                sum += eval(full, &mut out);
                n += 1.0;
            }
        }
        len => panic!("board must have 0, 3, 4, or 5 cards, got {len}"),
    }

    ((sum / n) + 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::make_card;

    fn uniform_opp(hole: [u8; 2]) -> BeliefState {
        let mut b = BeliefState::uniform();
        let mut mask = vec![1.0; NUM_COMBOS];
        for (i, m) in mask.iter_mut().enumerate() {
            let [a, c] = combo_cards(i);
            if a == hole[0] || a == hole[1] || c == hole[0] || c == hole[1] {
                *m = 0.0;
            }
        }
        b.update(&mask);
        b
    }

    #[test]
    fn river_nuts_has_equity_one() {
        // Quad aces on A A A 7 2; nothing in the deck beats it.
        let hole = [make_card(12, 0), make_card(12, 1)];
        let board = [
            make_card(12, 2),
            make_card(12, 3),
            make_card(5, 0),
            make_card(0, 1),
            make_card(3, 2),
        ];
        let mut rng = 1;
        let eq = equity_vs_range(hole, &board, &uniform_opp(hole), &mut rng);
        assert!(eq > 0.999, "nuts should be ~1.0, got {eq}");
    }

    #[test]
    fn preflop_aces_beat_a_random_range() {
        // AA against a uniform opponent is ~85% preflop.
        let hole = [make_card(12, 0), make_card(12, 1)];
        let mut rng = 12345;
        let eq = equity_vs_range(hole, &[], &uniform_opp(hole), &mut rng);
        assert!((0.80..0.90).contains(&eq), "AA preflop equity {eq} outside [0.80, 0.90]");
    }

    #[test]
    fn preflop_worst_hand_is_a_dog() {
        // 72o against a uniform opponent is ~35%.
        let hole = [make_card(5, 0), make_card(0, 1)];
        let mut rng = 999;
        let eq = equity_vs_range(hole, &[], &uniform_opp(hole), &mut rng);
        assert!((0.28..0.42).contains(&eq), "72o preflop equity {eq} outside [0.28, 0.42]");
    }

    #[test]
    fn bottom_pair_folds_to_a_shoving_range_where_a_set_calls() {
        // The shape of the -200bb Qs3d Slumbot loss: board 9c 3h 2c, facing a
        // near-stack shove.  Against the range that actually shoves there, the
        // old code called with bottom pair; the pot odds of calling a shove into
        // a small pot are around 0.45.
        let board = [make_card(7, 0), make_card(1, 2), make_card(0, 0)]; // 9c 3h 2c
        let shoving = BeliefState::from_hands(&[
            [make_card(7, 1), make_card(7, 3)],   // 99 -- top set
            [make_card(0, 1), make_card(0, 3)],   // 22 -- bottom set
            [make_card(12, 0), make_card(12, 1)], // AA
            [make_card(11, 0), make_card(11, 1)], // KK
            [make_card(10, 0), make_card(10, 1)], // QQ
        ]);
        let mut rng = 7;

        let bottom_pair = [make_card(10, 3), make_card(1, 1)]; // Qs 3d
        let set = [make_card(1, 0), make_card(1, 3)]; // 3c 3s

        let eq_weak = equity_vs_range(bottom_pair, &board, &shoving, &mut rng);
        let eq_set = equity_vs_range(set, &board, &shoving, &mut rng);
        assert!(eq_set > eq_weak, "set {eq_set} must beat bottom pair {eq_weak}");
        assert!(eq_weak < 0.45, "bottom pair {eq_weak} must not call a shove");
        assert!(eq_set > 0.45, "a set {eq_set} must call a shove");
    }

    #[test]
    fn turn_enumerates_every_runout() {
        // 46 unseen cards on a 4-card board; the mean must be a proper average.
        let hole = [make_card(12, 0), make_card(11, 0)];
        let board = [make_card(12, 2), make_card(11, 3), make_card(5, 0), make_card(0, 1)];
        let mut rng = 3;
        let eq = equity_vs_range(hole, &board, &uniform_opp(hole), &mut rng);
        assert!((0.0..=1.0).contains(&eq), "equity {eq} out of range");
        assert!(eq > 0.9, "top two pair vs a random range should be strong, got {eq}");
    }
}
