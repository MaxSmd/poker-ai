//! Pot geometry, bet sizing helpers, and action abstraction mapping.
//!
//! Abstract bet sizes per street (as fractions of pot):
//!
//! | Street  | Sizes (pot fractions)              |
//! |---------|------------------------------------|
//! | Preflop | fixed open/3-bet/4-bet multiples   |
//! | Flop    | 0.33, 0.67, 1.0                    |
//! | Turn    | 0.50, 0.75, 1.0                    |
//! | River   | 0.50, 0.75, 1.0, 1.5 (overbet)    |
//! | Any     | all-in always available (caller)   |

pub const FLOP_BET_FRACS: &[f64] = &[0.33, 0.67, 1.0];
pub const TURN_BET_FRACS: &[f64] = &[0.50, 0.75, 1.0];
pub const RIVER_BET_FRACS: &[f64] = &[0.50, 0.75, 1.0, 1.50];
/// Preflop uses the same pot-fraction logic; fractions chosen to yield
/// roughly 2.5 BB, 3.5 BB, and 6 BB opens from a standard blind structure.
pub const PREFLOP_BET_FRACS: &[f64] = &[0.50, 1.0, 2.0];

/// Total chip amounts committed to the pot so far (across both players / all players).
/// `total_committed` is the sum of all chips put in the pot by all players.
#[inline]
pub fn pot_total(total_committed: &[u32]) -> u32 {
    total_committed.iter().sum()
}

/// Compute abstract raise amounts (as total bet levels, not increments) for
/// the current state.
///
/// * `pot`        — total chips in the pot before this action
/// * `current_bet`— the current bet level that must be at least called
/// * `street`     — 0 = preflop, 1 = flop, 2 = turn, 3 = river
/// * `big_blind`  — BB size (used as minimum raise unit)
///
/// Returns a stack-allocated array of up to 6 absolute bet amounts plus the
/// count of valid entries.  Each entry is a total-bet level: the raiser's
/// `street_bet` would be set to this value.
pub fn abstract_raise_amounts(
    pot: u32,
    current_bet: u32,
    street: u8,
    big_blind: u32,
) -> ([u32; 6], usize) {
    let mut amounts = [0u32; 6];
    let mut count = 0usize;

    let fracs: &[f64] = match street {
        0 => PREFLOP_BET_FRACS,
        1 => FLOP_BET_FRACS,
        2 => TURN_BET_FRACS,
        3 => RIVER_BET_FRACS,
        _ => &[],
    };

    for &frac in fracs {
        // `raise_size` = how many extra chips the raiser adds above `current_bet`.
        // We size relative to `pot + current_bet` (i.e. the pot after the caller
        // puts in `current_bet`), so a "1× pot bet" means the raiser adds chips
        // equal to the current pot + the call amount.
        let raise_size = ((pot as f64 + current_bet as f64) * frac).round() as u32;
        let new_bet = current_bet + raise_size.max(big_blind);
        if count == 0 || amounts[count - 1] != new_bet {
            amounts[count] = new_bet;
            count += 1;
        }
    }

    (amounts, count)
}

/// Map a raw bet size to the nearest abstract bet size defined by the action
/// abstraction.  Returns the abstract total bet level.
///
/// * `pot`        — total pot before this raise
/// * `current_bet`— current highest bet to call
/// * `raw_bet`    — the raw total-bet proposed by the player
/// * `street`     — 0..=3
/// * `big_blind`  — BB size
pub fn abstract_bet_size(pot: u32, current_bet: u32, raw_bet: u32, street: u8, big_blind: u32) -> u32 {
    let (amounts, n) = abstract_raise_amounts(pot, current_bet, street, big_blind);
    if n == 0 {
        return raw_bet;
    }
    // Find the closest abstract size.
    let mut best = amounts[0];
    let mut best_dist = (raw_bet as i64 - best as i64).unsigned_abs();
    for &a in &amounts[1..n] {
        let d = (raw_bet as i64 - a as i64).unsigned_abs();
        if d < best_dist {
            best_dist = d;
            best = a;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flop_raises_increase_with_fraction() {
        let (amounts, n) = abstract_raise_amounts(100, 0, 1, 2);
        assert_eq!(n, 3);
        // Amounts should be strictly increasing.
        for i in 1..n {
            assert!(amounts[i] > amounts[i - 1]);
        }
    }

    #[test]
    fn river_has_four_sizes() {
        let (_, n) = abstract_raise_amounts(200, 0, 3, 2);
        assert_eq!(n, 4);
    }

    #[test]
    fn abstract_bet_maps_to_closest() {
        // pot = 100, current_bet = 0, street = flop (1), BB = 2
        // fractions: 0.33→33, 0.67→67, 1.0→100 chips
        let a = abstract_bet_size(100, 0, 35, 1, 2);
        // 35 is closer to 33 than to 67
        assert_eq!(a, 33);
        let b = abstract_bet_size(100, 0, 70, 1, 2);
        // 70 is closer to 67 than to 100
        assert_eq!(b, 67);
    }
}
