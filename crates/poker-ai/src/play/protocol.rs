//! Slumbot wire protocol: action-string parsing and chip accounting.
//!
//! Slumbot (slumbot.com) encodes a hand's betting as a single string that both
//! players' actions append to, e.g. `b200c/kb400c/kk/b1000f`:
//!
//! * `k` check, `c` call, `f` fold, `b<N>` bet/raise **to** `N`
//! * `N` counts the chips the bettor has put in **on that street only**
//! * `/` separates streets; an all-in can leave empty streets (`b20000c///`)
//!
//! Positions use Slumbot's convention: **pos 0 = big blind** (second to act
//! preflop, first postflop), **pos 1 = small blind / button** (first to act
//! preflop).  Blinds 50/100, stacks 20 000 (200 bb), reset every hand.
//!
//! [`parse_action`] is a faithful port of the validation logic in Slumbot's
//! published `sample_api.py` (same legality rules, same street bookkeeping),
//! extended to also emit the per-player chip state and the ordered [`Event`]
//! list the bot replays into its abstract game.

/// Slumbot's fixed stakes.
pub const SMALL_BLIND: u32 = 50;
pub const BIG_BLIND: u32 = 100;
pub const STACK_SIZE: u32 = 20_000;
const NUM_STREETS: u8 = 4;

/// One observed action. `pos` is in Slumbot convention (0 = BB, 1 = SB).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub street: u8,
    pub pos: u8,
    pub kind: EventKind,
    /// Total pot before this action (both players, current street included).
    pub pot_before: u32,
    /// The street's outstanding bet level before this action.
    pub bet_before: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Check,
    Call,
    Fold,
    /// Bet/raise **to** this street-cumulative level for the actor.
    BetTo(u32),
}

/// The full public state encoded by an action string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    /// Current street, `0..=3` (preflop..river).
    pub street: u8,
    /// Next player to act (Slumbot pos), `-1` if the hand needs no more actions.
    pub next_pos: i8,
    /// Current street's bet level (the amount to match).
    pub street_last_bet_to: u32,
    /// The bettor lineage's total commitment across streets.
    pub total_last_bet_to: u32,
    /// Size of the outstanding bet/raise (0 = nothing to call).
    pub last_bet_size: u32,
    /// Position of the outstanding bettor, `-1` if none.
    pub last_bettor: i8,
    /// Chips each position has put in on the current street (index = pos).
    pub street_committed: [u32; 2],
    /// Chips each position has put in over the whole hand (index = pos).
    pub total_committed: [u32; 2],
    /// Every action in order — the replay feed for the abstract tracker.
    pub events: Vec<Event>,
}

impl Parsed {
    /// Total pot (both players' commitments, current street included).
    pub fn pot(&self) -> u32 {
        self.total_committed[0] + self.total_committed[1]
    }

    /// Remaining stack of `pos`.
    pub fn stack(&self, pos: usize) -> u32 {
        STACK_SIZE - self.total_committed[pos]
    }
}

/// Parse a Slumbot action string. Returns an error string on any protocol
/// violation (same conditions `sample_api.py` rejects, plus an action after
/// the hand has ended, which the reference lets through).
pub fn parse_action(action: &str) -> Result<Parsed, String> {
    let bytes = action.as_bytes();
    // Blinds are posted: the small blind opens, facing half a big blind.
    let mut p = Parsed {
        street: 0,
        next_pos: 1,
        street_last_bet_to: BIG_BLIND,
        total_last_bet_to: BIG_BLIND,
        last_bet_size: BIG_BLIND - SMALL_BLIND,
        last_bettor: 0,
        street_committed: [BIG_BLIND, SMALL_BLIND], // pos 0 = BB, pos 1 = SB
        total_committed: [BIG_BLIND, SMALL_BLIND],
        events: Vec::new(),
    };
    // Whether the other player has already acted on this street, so that a
    // check or call now closes it.
    let mut closes_street = false;

    let mut i = 0;
    while i < bytes.len() {
        if p.next_pos < 0 {
            return Err("unexpected action after the hand ended".into());
        }
        let c = bytes[i];
        i += 1;
        let actor = p.next_pos as usize;
        let event = |kind| Event {
            street: p.street,
            pos: actor as u8,
            kind,
            pot_before: p.pot(),
            bet_before: p.street_last_bet_to,
        };
        match c {
            b'k' => {
                if p.last_bet_size > 0 {
                    return Err("illegal check facing a bet".into());
                }
                let ev = event(EventKind::Check);
                p.events.push(ev);
                pass_or_close(&mut p, &mut closes_street, bytes, &mut i)?;
            }
            b'c' => {
                if p.last_bet_size == 0 {
                    return Err("illegal call with no outstanding bet".into());
                }
                let ev = event(EventKind::Call);
                p.events.push(ev);
                p.total_committed[actor] += p.street_last_bet_to - p.street_committed[actor];
                p.street_committed[actor] = p.street_last_bet_to;
                if p.total_last_bet_to == STACK_SIZE {
                    // Call of an all-in: optionally the slashes closing every
                    // pre-river street, then nothing else.
                    if i != bytes.len() {
                        for _ in p.street..NUM_STREETS - 1 {
                            if i == bytes.len() {
                                return Err("missing street separator at end".into());
                            }
                            if bytes[i] != b'/' {
                                return Err("missing street separator".into());
                            }
                            i += 1;
                        }
                    }
                    if i != bytes.len() {
                        return Err("extra characters after an all-in call".into());
                    }
                    p.street = NUM_STREETS - 1; // the board runs out
                    p.next_pos = -1;
                    p.last_bet_size = 0;
                    return Ok(p);
                }
                pass_or_close(&mut p, &mut closes_street, bytes, &mut i)?;
                p.last_bet_size = 0;
                p.last_bettor = -1;
            }
            b'f' => {
                if p.last_bet_size == 0 {
                    return Err("illegal fold with no outstanding bet".into());
                }
                if i != bytes.len() {
                    return Err("extra characters after a fold".into());
                }
                let ev = event(EventKind::Fold);
                p.events.push(ev);
                p.next_pos = -1;
                return Ok(p);
            }
            b'b' => {
                let digits = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i == digits {
                    return Err("missing bet size".into());
                }
                let bet_to: u32 =
                    action[digits..i].parse().map_err(|_| "bet size not an integer".to_string())?;
                let bet_size = bet_to.saturating_sub(p.street_last_bet_to);
                // Legality (Slumbot's own rules): minimum is the last bet size,
                // floored at the big blind, capped at the remaining stack; an
                // all-in for less is always allowed.
                let remaining = STACK_SIZE - p.total_last_bet_to;
                let min_bet_size = p.last_bet_size.max(BIG_BLIND).min(remaining);
                if bet_size < min_bet_size {
                    return Err(format!("bet too small ({bet_size} < {min_bet_size})"));
                }
                if bet_size > remaining {
                    return Err(format!("bet too big ({bet_size} > {remaining})"));
                }
                let ev = event(EventKind::BetTo(bet_to));
                p.events.push(ev);
                p.total_committed[actor] += bet_to - p.street_committed[actor];
                p.street_committed[actor] = bet_to;
                p.last_bet_size = bet_size;
                p.street_last_bet_to = bet_to;
                p.total_last_bet_to += bet_size;
                p.last_bettor = actor as i8;
                p.next_pos = (p.next_pos + 1) % 2;
                closes_street = true;
            }
            _ => return Err(format!("unexpected character {:?} in action", c as char)),
        }
    }
    Ok(p)
}

/// After a check or call: pass the action across, or — when the other player
/// has already acted this street (`closes_street`) — close the street,
/// consuming the `/` that follows a closed pre-river street when the string
/// continues.  The river closing is the showdown.
fn pass_or_close(p: &mut Parsed, closes_street: &mut bool, bytes: &[u8], i: &mut usize) -> Result<(), String> {
    if !*closes_street {
        p.next_pos = (p.next_pos + 1) % 2;
        *closes_street = true;
        return Ok(());
    }
    let river = p.street == NUM_STREETS - 1;
    if !river && *i < bytes.len() {
        if bytes[*i] != b'/' {
            return Err("missing street separator".into());
        }
        *i += 1;
    }
    if river {
        p.next_pos = -1; // showdown
    } else {
        p.next_pos = 0; // the big blind opens every postflop street
        p.street += 1;
        p.street_committed = [0, 0];
    }
    p.street_last_bet_to = 0;
    *closes_street = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_action_is_sb_to_act_preflop() {
        let p = parse_action("").unwrap();
        assert_eq!(p.street, 0);
        assert_eq!(p.next_pos, 1, "small blind acts first preflop");
        assert_eq!(p.street_last_bet_to, BIG_BLIND);
        assert_eq!(p.last_bet_size, BIG_BLIND - SMALL_BLIND);
        assert_eq!(p.total_committed, [BIG_BLIND, SMALL_BLIND]);
        assert_eq!(p.pot(), 150);
    }

    #[test]
    fn docstring_example_pot_size_flop_bet() {
        // "b200c/kb400": preflop raise to 200 called (pot 400), flop check then
        // a pot-size bet of 400 — from the API documentation.
        let p = parse_action("b200c/kb400").unwrap();
        assert_eq!(p.street, 1);
        assert_eq!(p.next_pos, 0, "BB faces the flop bet");
        assert_eq!(p.street_last_bet_to, 400);
        assert_eq!(p.last_bet_size, 400);
        assert_eq!(p.last_bettor, 1);
        assert_eq!(p.total_committed, [200, 600]);
        assert_eq!(p.pot(), 800);
        assert_eq!(
            p.events,
            vec![
                Event { street: 0, pos: 1, kind: EventKind::BetTo(200), pot_before: 150, bet_before: 100 },
                Event { street: 0, pos: 0, kind: EventKind::Call, pot_before: 300, bet_before: 200 },
                Event { street: 1, pos: 0, kind: EventKind::Check, pot_before: 400, bet_before: 0 },
                Event { street: 1, pos: 1, kind: EventKind::BetTo(400), pot_before: 400, bet_before: 0 },
            ]
        );
    }

    #[test]
    fn checked_down_hand_reaches_showdown() {
        let p = parse_action("b200c/kk/kk/kk").unwrap();
        assert_eq!(p.street, 3);
        assert_eq!(p.next_pos, -1, "river checks through = showdown");
        assert_eq!(p.total_committed, [200, 200]);
    }

    #[test]
    fn preflop_limp_check_advances_to_flop() {
        let p = parse_action("ck/").unwrap();
        assert_eq!(p.street, 1);
        assert_eq!(p.next_pos, 0, "BB first to act postflop");
        assert_eq!(p.total_committed, [100, 100]);
        assert_eq!(p.street_committed, [0, 0]);
        // Same string without the trailing slash parses identically.
        assert_eq!(parse_action("ck").unwrap(), p);
    }

    #[test]
    fn allin_call_short_circuits_to_river() {
        for s in ["b20000c", "b20000c///"] {
            let p = parse_action(s).unwrap();
            assert_eq!(p.street, 3, "runout to the river");
            assert_eq!(p.next_pos, -1);
            assert_eq!(p.total_committed, [20_000, 20_000]);
            assert_eq!(p.last_bet_size, 0);
        }
    }

    #[test]
    fn fold_ends_the_hand() {
        let p = parse_action("b300f").unwrap();
        assert_eq!(p.next_pos, -1);
        assert_eq!(p.total_committed, [100, 300]);
        assert_eq!(p.events.last().unwrap().kind, EventKind::Fold);
    }

    #[test]
    fn raise_accounting_tracks_street_levels() {
        // SB to 300, BB reraises to 900, SB calls; turn: BB bets 500, SB raises
        // to 1500, BB calls; river pending.
        let p = parse_action("b300b900c/kk/b500b1500c/").unwrap();
        assert_eq!(p.street, 3);
        assert_eq!(p.next_pos, 0);
        assert_eq!(p.total_committed, [2400, 2400]);
        assert_eq!(p.street_committed, [0, 0]);
        assert_eq!(p.last_bet_size, 0);
    }

    #[test]
    fn illegal_actions_are_rejected() {
        assert!(parse_action("k").is_err(), "check facing the blind");
        assert!(parse_action("cc").is_err(), "call with no outstanding bet");
        assert!(parse_action("ff").is_err(), "fold with nothing to fold to");
        assert!(parse_action("b120b130").is_err(), "raise below min raise");
        assert!(parse_action("b90").is_err(), "open below the big blind");
        assert!(parse_action("b30000").is_err(), "bet beyond the stack");
        assert!(parse_action("b200x").is_err(), "unknown character");
    }

    #[test]
    fn min_raise_is_the_last_bet_size() {
        // Raise to 300 (size 200); the minimum reraise is to 500 (another 200).
        assert!(parse_action("b300b499").is_err());
        assert!(parse_action("b300b500").is_ok());
    }

    /// A move after the hand is over has no actor.  The reference parser lets
    /// a check through with a nonsense position; a bet there indexed the chip
    /// arrays with position −1 and panicked, taking the whole match down on
    /// one malformed server string.  It must be an ordinary error.
    #[test]
    fn a_move_after_the_hand_ended_is_an_error_not_a_panic() {
        for over in ["b200c/kk/kk/kk", "b300f", "b20000c///"] {
            assert_eq!(parse_action(over).unwrap().next_pos, -1, "{over:?} ends the hand");
        }
        for late in ["b200c/kk/kk/kkk", "b200c/kk/kk/kkb300", "b200c/kk/kk/kkc", "b200c/kk/kk/kk/"] {
            assert!(parse_action(late).is_err(), "{late:?} acts after the showdown");
        }
    }
}
