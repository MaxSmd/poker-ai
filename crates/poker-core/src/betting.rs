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
//!
//! All sizing arithmetic uses integer-only math (numerator/denominator pairs)
//! to guarantee deterministic results across platforms.  No floating-point
//! operations are performed.

/// Pot fractions stored as `(numerator, denominator)` pairs for integer arithmetic.
/// The fraction value is `numerator / denominator`.
pub const FLOP_BET_FRACS: &[(u32, u32)] = &[(33, 100), (67, 100), (1, 1)];
pub const TURN_BET_FRACS: &[(u32, u32)] = &[(1, 2), (3, 4), (1, 1)];
pub const RIVER_BET_FRACS: &[(u32, u32)] = &[(1, 2), (3, 4), (1, 1), (3, 2)];
/// Preflop uses the same pot-fraction logic; fractions chosen to yield
/// roughly 2.5 BB, 3.5 BB, and 6 BB opens from a standard blind structure.
pub const PREFLOP_BET_FRACS: &[(u32, u32)] = &[(1, 2), (1, 1), (2, 1)];

/// Compute abstract raise amounts (as total bet levels, not increments) for
/// the current state.
///
/// * `pot`        — total chips in the pot before this action
/// * `current_bet`— the current bet level that must be at least called
/// * `min_raise`  — minimum raise increment (last raise size or 1 BB)
/// * `street`     — 0 = preflop, 1 = flop, 2 = turn, 3 = river
///
/// Returns a stack-allocated array of up to 6 absolute bet amounts plus the
/// count of valid entries.  Each entry is a total-bet level: the raiser's
/// `street_bet` would be set to this value.
pub fn abstract_raise_amounts(
    pot: u32,
    current_bet: u32,
    min_raise: u32,
    street: u8,
) -> ([u32; 6], usize) {
    let mut amounts = [0u32; 6];
    let mut count = 0usize;

    let fracs: &[(u32, u32)] = match street {
        0 => PREFLOP_BET_FRACS,
        1 => FLOP_BET_FRACS,
        2 => TURN_BET_FRACS,
        3 => RIVER_BET_FRACS,
        _ => &[],
    };

    for &(num, den) in fracs {
        // `raise_size` = extra chips the raiser adds above `current_bet`.
        // Sized relative to `pot + current_bet` (the pot after the call).
        // Integer rounding: (base * num + den/2) / den  (round-half-up).
        let base = pot as u64 + current_bet as u64;
        let raise_size = ((base * num as u64 + den as u64 / 2) / den as u64) as u32;
        // The raise must be at least the minimum raise increment (which tracks
        // the last raise size, not just the big blind).
        let new_bet = current_bet + raise_size.max(min_raise);
        // Deduplicate: skip if this rounds to the same total as the previous entry.
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
/// * `min_raise`  — minimum raise increment
/// * `street`     — 0..=3
pub fn abstract_bet_size(pot: u32, current_bet: u32, raw_bet: u32, min_raise: u32, street: u8) -> u32 {
    let (amounts, n) = abstract_raise_amounts(pot, current_bet, min_raise, street);
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
        let (amounts, n) = abstract_raise_amounts(100, 0, 2, 1);
        assert_eq!(n, 3);
        // Amounts should be strictly increasing.
        for i in 1..n {
            assert!(amounts[i] > amounts[i - 1]);
        }
    }

    #[test]
    fn river_has_four_sizes() {
        let (_, n) = abstract_raise_amounts(200, 0, 2, 3);
        assert_eq!(n, 4);
    }

    #[test]
    fn abstract_bet_maps_to_closest() {
        // pot = 100, current_bet = 0, min_raise = 2, street = flop (1)
        // fractions: 33/100→33, 67/100→67, 1/1→100 chips
        let a = abstract_bet_size(100, 0, 35, 2, 1);
        // 35 is closer to 33 than to 67
        assert_eq!(a, 33);
        let b = abstract_bet_size(100, 0, 70, 2, 1);
        // 70 is closer to 67 than to 100
        assert_eq!(b, 67);
    }

    #[test]
    fn min_raise_used_as_floor_not_big_blind() {
        // If min_raise is 20 (from a prior raise), even a small pot fraction
        // must produce at least current_bet + 20.
        let (amounts, n) = abstract_raise_amounts(40, 20, 20, 1);
        assert!(n > 0);
        for &a in &amounts[..n] {
            assert!(a >= 40, "raise total {a} must be >= current_bet(20) + min_raise(20)");
        }
    }

    #[test]
    fn integer_arithmetic_determinism() {
        // Verify that repeated calls produce identical results (no float drift).
        let (a1, n1) = abstract_raise_amounts(137, 23, 10, 2);
        let (a2, n2) = abstract_raise_amounts(137, 23, 10, 2);
        assert_eq!(n1, n2);
        assert_eq!(a1[..n1], a2[..n2]);
    }
}
