//! Tests for the playing agent.
//!
//! The load-bearing ones drive whole hands against a scripted opponent and
//! assert every emitted move is a protocol-legal increment, plus live resolves
//! (river / turn / flop / gadget carry) returning legal root actions.

use super::*;
use crate::play::protocol::{BIG_BLIND, STACK_SIZE};
use crate::play::cards::parse_card;

fn bot(resolve: bool) -> Bot {
    let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
    let policy = CompactPolicy::from_entries(vec![]); // uniform everywhere
    Bot::new(
        game,
        policy,
        BotConfig {
            resolve_river: resolve,
            river_iters: 120,
            river_cap: 2,
            purify: 0.0,
            seed: 42,
            ..Default::default()
        },
    )
}

fn cards(list: &[&str]) -> Vec<u8> {
    list.iter().map(|s| parse_card(s).unwrap()).collect()
}

/// Drive the bot through whole hands against a scripted/random opponent,
/// verifying every emitted increment is legal per the protocol parser.
#[test]
fn emits_legal_increments_over_random_hands() {
    let mut b = bot(false);
    let mut rng = 0xBADC_0FFEu64;
    let mut unit = move || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };

    let full_board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
    for hand_no in 0..40u32 {
        let client_pos = (hand_no % 2) as u8;
        let hole = if hand_no % 3 == 0 {
            [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()]
        } else {
            [parse_card("7h").unwrap(), parse_card("2c").unwrap()]
        };
        let mut hs = b.start_hand(client_pos, hole);
        let mut action = String::new();

        loop {
            let parsed = parse_action(&action).expect("running action string stays legal");
            if parsed.next_pos < 0 {
                break; // hand over
            }
            let street = parsed.street as usize;
            let board = &full_board[..[0usize, 3, 4, 5][street]];
            let before_street = parsed.street;
            if parsed.next_pos == hs.my_pos as i8 {
                let incr = b.act(&mut hs, &action, board).expect("bot acts");
                action.push_str(&incr);
            } else {
                // Random legal opponent move.
                let facing = parsed.last_bet_size > 0;
                let remaining = STACK_SIZE - parsed.total_last_bet_to;
                let choice = unit();
                let mv = if facing {
                    if choice < 0.55 {
                        "c".to_string()
                    } else if choice < 0.7 && remaining > 0 {
                        let min = parsed.last_bet_size.max(BIG_BLIND).min(remaining);
                        let to = parsed.street_last_bet_to + min.max((remaining as f64 * unit() * 0.4) as u32).min(remaining);
                        format!("b{to}")
                    } else {
                        "f".to_string()
                    }
                } else if choice < 0.6 || remaining == 0 {
                    "k".to_string()
                } else {
                    let min = BIG_BLIND.min(remaining);
                    let to = parsed.street_last_bet_to
                        + min.max((remaining as f64 * unit() * 0.3) as u32).min(remaining);
                    format!("b{to}")
                };
                action.push_str(&mv);
            }
            // Close a finished pre-river street with the slash the server
            // inserts (mid-string separators are mandatory).
            let reparsed = parse_action(&action).expect("every appended move stays legal");
            if reparsed.next_pos >= 0 && reparsed.street > before_street && !action.ends_with('/')
            {
                action.push('/');
            }
        }
    }
}

#[test]
fn bucketed_flop_updates_never_index_out_of_bounds() {
    // Regression: with a real (bounds-checked) bucket map loaded, the
    // belief-update loop used to feed board-overlapping combos into the
    // hand indexer — a duplicated card yields an index past the canonical
    // table and panicked in BucketMap::bucket (seen live on the first
    // flop decision against Slumbot).  Card removal at board reveal plus
    // the defensive mask must keep every queried combo valid.
    use crate::abstraction::bucket_map::BucketMap;
    let game = BlueprintHoldem::new(400, 2, 1, 0)
        .with_raise_cap(3)
        .with_street_bucket(0, BucketMap::full_coverage_mod(&[2, 3], 40));
    let policy = CompactPolicy::from_entries(vec![]);
    let mut b = Bot::new(
        game,
        policy,
        BotConfig {
            resolve_river: false,
            river_iters: 0,
            river_cap: 2,
            purify: 0.0,
            seed: 7,
            ..Default::default()
        },
    );
    // We are the BB (first to act postflop): SB opens, we call, flop comes,
    // our decision triggers a bucketed range update over the full board.
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole);
    let board = cards(&["Qs", "4s", "3h"]);
    let incr = b.act(&mut hs, "b300c/", &board).expect("flop decision with bucket map");
    assert!(parse_action(&format!("b300c/{incr}")).is_ok(), "legal move, got {incr:?}");

    // Board-overlapping combos carry zero mass in both ranges.
    for r in &hs.ranges {
        for (i, &p) in r.probs.iter().enumerate() {
            let [a, c] = combo_cards(i);
            if board.contains(&a) || board.contains(&c) {
                assert_eq!(p, 0.0, "board-overlap combo must be dead");
            }
        }
        assert!((r.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    // And the opponent can never hold our exact cards.
    let opp = &hs.ranges[1 - hs.my_seat];
    assert_eq!(opp.prob(hole[0], hole[1]), 0.0);
}

#[test]
fn river_resolve_returns_a_root_action() {
    let mut b = bot(true);
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole); // we are BB, first to act postflop
    // SB open to 200, we call; flop checks; turn checks; river to us.
    let action = "b200c/kk/kk/";
    let board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
    let incr = b.act(&mut hs, action, &board).expect("river decision");
    assert!(
        incr == "k" || incr.starts_with('b'),
        "unopened river allows check or bet, got {incr:?}"
    );
    if let Some(n) = incr.strip_prefix('b') {
        let to: u32 = n.parse().unwrap();
        assert!((BIG_BLIND..=19_800).contains(&to), "legal river bet size, got {to}");
    }
    // The full string with our move must still parse.
    assert!(parse_action(&format!("{action}{incr}")).is_ok());
}

/// Continual re-solving end-to-end: the hand's first resolve bootstraps
/// and stores the opponent's CFVs; a second decision in the same hand
/// runs the gadget-constrained resolve off that carry and still produces
/// a legal move.
#[test]
fn second_river_decision_carries_cfvs_through_the_gadget() {
    let mut b = bot(true);
    assert!(b.cfg.continual, "production default");
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole); // we are BB, first to act postflop
    let action = "b200c/kk/kk/";
    let board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
    let incr = b.act(&mut hs, action, &board).expect("first river decision");
    let carried = hs.carried_cfvs.clone().expect("bootstrap resolve must store CFVs");
    assert!(carried.iter().all(|v| v.is_finite()), "CFVs must be finite");

    // The opponent raises us back into a second decision on the same street.
    let action2 = if incr == "k" {
        format!("{action}kb1000")
    } else {
        let to: u32 = incr.strip_prefix('b').expect("check or bet").parse().unwrap();
        format!("{action}{incr}b{}", (to * 3).min(19_000))
    };
    let incr2 = b.act(&mut hs, &action2, &board).expect("gadget-constrained decision");
    assert!(
        incr2 == "f" || incr2 == "c" || incr2.starts_with('b'),
        "facing a bet allows fold/call/raise, got {incr2:?}"
    );
    assert!(parse_action(&format!("{action2}{incr2}")).is_ok());
    // The carry was refreshed by the second resolve.
    assert!(hs.carried_cfvs.is_some());
}

/// A preflop raise war no longer desyncs: `AllIn` survives the cap, so the
/// opponent's 5-bet maps onto it and the tracker keeps up.
#[test]
fn a_post_cap_five_bet_maps_onto_the_all_in() {
    let mut b = bot(false);
    let h = [parse_card("Ac").unwrap(), parse_card("Ad").unwrap()];
    let mut hs = b.start_hand(1, h);

    let action = "b250b750b1657b12000";
    let parsed = parse_action(action).expect("legal action string");
    b.sync(&mut hs, &parsed, &[]);
    assert!(
        hs.hand.expects(&b.game, hs.my_seat, 0),
        "the 5-bet must translate onto the abstract all-in, not desync"
    );
}

/// A desync is still reachable: once the abstraction is all-in but the real
/// stacks are not, later streets have no node.  The agent used to answer
/// every such spot with an unconditional call -- which is how it called off
/// 200bb with bottom pair against Slumbot.
#[test]
fn a_desynced_bot_folds_trash_and_calls_a_monster() {
    // Preflop war ends with a call; the abstraction is all-in.  On the flop
    // the opponent shoves 13250 into 26750, so calling needs 0.33 equity.
    let action = "b250b750b1657b6750c/b13250";
    let board = cards(&["2c", "7h", "9s"]);
    for (hole, expected) in [(["3d", "4d"], "f"), (["Ac", "Ad"], "c")] {
        let mut b = bot(false);
        let h = [parse_card(hole[0]).unwrap(), parse_card(hole[1]).unwrap()];
        let mut hs = b.start_hand(1, h);

        let parsed = parse_action(action).expect("legal action string");
        b.sync(&mut hs, &parsed, &board);
        assert!(
            !hs.hand.expects(&b.game, hs.my_seat, 1),
            "an all-in abstraction over live real stacks must desync"
        );

        let incr = b.act(&mut hs, action, &board).expect("bot acts");
        assert_eq!(incr, expected, "holding {hole:?} facing a flop shove");
    }
}

/// Turn re-solving is wired end-to-end: `decide_resolve` synthesizes the
/// turn root, runs the K-continuation runout solver, and plays a legal root
/// action.  Small cap + K keep the deep-stack turn tree fast for a unit test.
fn resolve_bot(turn: bool, flop: bool, continuations: Vec<f64>) -> Bot {
    let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
    let policy = CompactPolicy::from_entries(vec![]);
    Bot::new(
        game,
        policy,
        BotConfig {
            resolve_river: false,
            resolve_turn: turn,
            resolve_flop: flop,
            river_iters: 0,
            turn_iters: 40,
            river_cap: 1,
            continuations,
            // These tests pin the K-continuation leaf path; the full-river
            // path has its own test below.
            turn_full_river: false,
            // Exact runout sweeps: these tests pin leaf wiring and legality,
            // so they should not also depend on the sampling schedule.
            runout_sample: 0,
            continual: true,
            purify: 0.0,
            seed: 42,
        },
    )
}

#[test]
fn turn_full_river_resolve_returns_a_root_action() {
    // The default turn mode: river dealt as chance, real river betting
    // solved below.  Same scenario as the K-continuation test.
    let mut b = resolve_bot(true, false, vec![0.0]);
    b.cfg.turn_full_river = true;
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole);
    let action = "b200c/kk/";
    let board = cards(&["Qs", "4s", "3h", "Th"]);
    let incr = b.act(&mut hs, action, &board).expect("turn decision");
    assert!(
        incr == "k" || incr.starts_with('b'),
        "unopened turn allows check or bet, got {incr:?}"
    );
    assert!(parse_action(&format!("{action}{incr}")).is_ok());
}

#[test]
fn turn_resolve_returns_a_root_action() {
    // K=2 exercises the continuation-chooser path through the bot as well.
    let mut b = resolve_bot(true, false, vec![0.0, 1.0]);
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole); // BB, first to act postflop
    // Preflop SB open + call; flop checks through; turn is on us, unopened.
    let action = "b200c/kk/";
    let board = cards(&["Qs", "4s", "3h", "Th"]);
    let incr = b.act(&mut hs, action, &board).expect("turn decision");
    assert!(
        incr == "k" || incr.starts_with('b'),
        "unopened turn allows check or bet, got {incr:?}"
    );
    if let Some(n) = incr.strip_prefix('b') {
        let to: u32 = n.parse().unwrap();
        assert!((BIG_BLIND..=19_800).contains(&to), "legal turn bet size, got {to}");
    }
    assert!(parse_action(&format!("{action}{incr}")).is_ok());
}

#[test]
#[ignore = "flop resolve enumerates C(45,2)=990 runouts per leaf — seconds-slow; \
            the wiring is exercised fast by turn_resolve_returns_a_root_action"]
fn flop_resolve_returns_a_root_action() {
    let mut b = resolve_bot(false, true, vec![0.0, 1.0]);
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole); // BB, first to act postflop
    let action = "b200c/"; // preflop done; flop is on us, unopened
    let board = cards(&["Qs", "4s", "3h"]);
    let incr = b.act(&mut hs, action, &board).expect("flop decision");
    assert!(
        incr == "k" || incr.starts_with('b'),
        "unopened flop allows check or bet, got {incr:?}"
    );
    assert!(parse_action(&format!("{action}{incr}")).is_ok());
}

#[test]
fn facing_a_river_shove_resolves_to_a_legal_response() {
    let mut b = bot(true);
    let hole = [parse_card("Ac").unwrap(), parse_card("Ad").unwrap()];
    let mut hs = b.start_hand(0, hole);
    let action = "b200c/kk/kk/kb19800";
    let board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
    let incr = b.act(&mut hs, action, &board).expect("shove response");
    assert!(incr == "c" || incr == "f", "call or fold vs a shove, got {incr:?}");
    assert!(parse_action(&format!("{action}{incr}")).is_ok());
}
