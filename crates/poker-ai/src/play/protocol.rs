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
/// violation (same conditions `sample_api.py` rejects).
pub fn parse_action(action: &str) -> Result<Parsed, String> {
    let bytes = action.as_bytes();
    let sz = bytes.len();

    let mut st: u8 = 0;
    let mut street_last_bet_to = BIG_BLIND;
    let mut total_last_bet_to = BIG_BLIND;
    let mut last_bet_size = BIG_BLIND - SMALL_BLIND;
    let mut last_bettor: i8 = 0;
    let mut pos: i8 = 1; // SB acts first preflop
    let mut street_committed = [BIG_BLIND, SMALL_BLIND]; // pos 0 = BB, pos 1 = SB
    let mut total_committed = [BIG_BLIND, SMALL_BLIND];
    let mut events = Vec::new();

    let done = |st: u8,
                pos: i8,
                street_last_bet_to: u32,
                total_last_bet_to: u32,
                last_bet_size: u32,
                last_bettor: i8,
                street_committed: [u32; 2],
                total_committed: [u32; 2],
                events: Vec<Event>| Parsed {
        street: st,
        next_pos: pos,
        street_last_bet_to,
        total_last_bet_to,
        last_bet_size,
        last_bettor,
        street_committed,
        total_committed,
        events,
    };

    if sz == 0 {
        return Ok(done(
            st,
            pos,
            street_last_bet_to,
            total_last_bet_to,
            last_bet_size,
            last_bettor,
            street_committed,
            total_committed,
            events,
        ));
    }

    let mut check_or_call_ends_street = false;
    let mut i = 0usize;
    while i < sz {
        if st >= NUM_STREETS {
            return Err("unexpected action after the river".into());
        }
        let c = bytes[i];
        i += 1;
        match c {
            b'k' => {
                if last_bet_size > 0 {
                    return Err("illegal check facing a bet".into());
                }
                events.push(Event {
                    street: st,
                    pos: pos as u8,
                    kind: EventKind::Check,
                    pot_before: total_committed[0] + total_committed[1],
                    bet_before: street_last_bet_to,
                });
                if check_or_call_ends_street {
                    if st < NUM_STREETS - 1 && i < sz {
                        if bytes[i] != b'/' {
                            return Err("missing street separator".into());
                        }
                        i += 1;
                    }
                    if st == NUM_STREETS - 1 {
                        pos = -1; // showdown
                    } else {
                        pos = 0;
                        st += 1;
                        street_committed = [0, 0];
                    }
                    street_last_bet_to = 0;
                    check_or_call_ends_street = false;
                } else {
                    pos = (pos + 1) % 2;
                    check_or_call_ends_street = true;
                }
            }
            b'c' => {
                if last_bet_size == 0 {
                    return Err("illegal call with no outstanding bet".into());
                }
                let caller = pos as usize;
                events.push(Event {
                    street: st,
                    pos: pos as u8,
                    kind: EventKind::Call,
                    pot_before: total_committed[0] + total_committed[1],
                    bet_before: street_last_bet_to,
                });
                total_committed[caller] += street_last_bet_to - street_committed[caller];
                street_committed[caller] = street_last_bet_to;
                if total_last_bet_to == STACK_SIZE {
                    // Call of an all-in: optionally slashes closing every
                    // pre-river street, then nothing else.
                    if i != sz {
                        for _ in st..NUM_STREETS - 1 {
                            if i == sz {
                                return Err("missing street separator at end".into());
                            }
                            if bytes[i] != b'/' {
                                return Err("missing street separator".into());
                            }
                            i += 1;
                        }
                    }
                    if i != sz {
                        return Err("extra characters after an all-in call".into());
                    }
                    return Ok(done(
                        NUM_STREETS - 1,
                        -1,
                        street_last_bet_to,
                        total_last_bet_to,
                        0,
                        last_bettor,
                        street_committed,
                        total_committed,
                        events,
                    ));
                }
                if check_or_call_ends_street {
                    if st < NUM_STREETS - 1 && i < sz {
                        if bytes[i] != b'/' {
                            return Err("missing street separator".into());
                        }
                        i += 1;
                    }
                    if st == NUM_STREETS - 1 {
                        pos = -1; // showdown
                    } else {
                        pos = 0;
                        st += 1;
                        street_committed = [0, 0];
                    }
                    street_last_bet_to = 0;
                    check_or_call_ends_street = false;
                } else {
                    pos = (pos + 1) % 2;
                    check_or_call_ends_street = true;
                }
                last_bet_size = 0;
                last_bettor = -1;
            }
            b'f' => {
                if last_bet_size == 0 {
                    return Err("illegal fold with no outstanding bet".into());
                }
                if i != sz {
                    return Err("extra characters after a fold".into());
                }
                events.push(Event {
                    street: st,
                    pos: pos as u8,
                    kind: EventKind::Fold,
                    pot_before: total_committed[0] + total_committed[1],
                    bet_before: street_last_bet_to,
                });
                return Ok(done(
                    st,
                    -1,
                    street_last_bet_to,
                    total_last_bet_to,
                    last_bet_size,
                    last_bettor,
                    street_committed,
                    total_committed,
                    events,
                ));
            }
            b'b' => {
                let j = i;
                while i < sz && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i == j {
                    return Err("missing bet size".into());
                }
                let new_street_last_bet_to: u32 =
                    action[j..i].parse().map_err(|_| "bet size not an integer".to_string())?;
                let new_last_bet_size = new_street_last_bet_to.saturating_sub(street_last_bet_to);
                // Legality (Slumbot's own rules): minimum is the last bet size,
                // floored at the big blind, capped at the remaining stack; an
                // all-in for less is always allowed.
                let remaining = STACK_SIZE - total_last_bet_to;
                let min_bet_size = last_bet_size.max(BIG_BLIND).min(remaining);
                if new_last_bet_size < min_bet_size {
                    return Err(format!("bet too small ({new_last_bet_size} < {min_bet_size})"));
                }
                if new_last_bet_size > remaining {
                    return Err(format!("bet too big ({new_last_bet_size} > {remaining})"));
                }
                let bettor = pos as usize;
                events.push(Event {
                    street: st,
                    pos: pos as u8,
                    kind: EventKind::BetTo(new_street_last_bet_to),
                    pot_before: total_committed[0] + total_committed[1],
                    bet_before: street_last_bet_to,
                });
                total_committed[bettor] += new_street_last_bet_to - street_committed[bettor];
                street_committed[bettor] = new_street_last_bet_to;
                last_bet_size = new_last_bet_size;
                street_last_bet_to = new_street_last_bet_to;
                total_last_bet_to += last_bet_size;
                last_bettor = pos;
                pos = (pos + 1) % 2;
                check_or_call_ends_street = true;
            }
            _ => return Err(format!("unexpected character {:?} in action", c as char)),
        }
    }

    Ok(done(
        st,
        pos,
        street_last_bet_to,
        total_last_bet_to,
        last_bet_size,
        last_bettor,
        street_committed,
        total_committed,
        events,
    ))
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
}
