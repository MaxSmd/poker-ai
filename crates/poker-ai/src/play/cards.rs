//! Card codec between Slumbot's strings ("Ac", "Td") and the engine's `u8`
//! encoding (`rank << 2 | suit`, ranks 0..=12 for 2..=A).

use poker_core::{make_card, rank_of, suit_of};

const RANKS: &[u8; 13] = b"23456789TJQKA";
const SUITS: &[u8; 4] = b"cdhs";

/// Parse a two-character card like `"Ac"` or `"Td"`.
pub fn parse_card(s: &str) -> Result<u8, String> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return Err(format!("bad card {s:?}"));
    }
    let rank = RANKS.iter().position(|&r| r == b[0]).ok_or_else(|| format!("bad rank in {s:?}"))?;
    let suit = SUITS.iter().position(|&c| c == b[1]).ok_or_else(|| format!("bad suit in {s:?}"))?;
    Ok(make_card(rank as u8, suit as u8))
}

/// Format a card back to Slumbot's notation.
pub fn card_str(card: u8) -> String {
    format!("{}{}", RANKS[rank_of(card) as usize] as char, SUITS[suit_of(card) as usize] as char)
}

/// Parse a hand or board list.
pub fn parse_cards(strs: &[String]) -> Result<Vec<u8>, String> {
    strs.iter().map(|s| parse_card(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_card() {
        for c in 0u8..52 {
            assert_eq!(parse_card(&card_str(c)).unwrap(), c);
        }
    }

    #[test]
    fn known_cards() {
        assert_eq!(parse_card("2c").unwrap(), 0, "deuce of clubs is card 0");
        assert_eq!(parse_card("As").unwrap(), make_card(12, 3));
        assert!(parse_card("1c").is_err());
        assert!(parse_card("Ax").is_err());
        assert!(parse_card("Ac2").is_err());
    }
}
