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
            search: false,
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
    let mut unit = move || crate::util::rng::xorshift_next_unit(&mut rng);

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
            search: false,
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
        // Pin the pot-odds fallback itself; by default a desynced postflop
        // spot is re-solved instead (next test).
        b.cfg.resolve_desynced = false;
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
            resolve_desynced: true,
            purify: 0.0,
            opp_range_floor: 0.0,
            search: false,
            search_iters: 0,
            search_continuations: 4,
            search_bias: 5.0,
            search_searcher_chooses: true,
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

/// After a re-solved decision our range is conditioned on the strategy the
/// resolve played, not the blueprint's.  The blueprint here is uniform, so
/// the old blueprint-likelihood update left the range exactly as it was; a
/// river resolve mixes differently across hands, so the new one reshapes it.
/// The carry is then taken at the chosen child and must exist for any
/// non-fold action.
#[test]
fn resolved_decision_updates_our_range_from_the_resolve() {
    let mut b = bot(true);
    b.cfg.continual = true;
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole); // BB, first to act postflop
    let action = "b200c/kk/kk/";
    let board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
    b.sync(&mut hs, &parse_action(action).unwrap(), &board);
    let before = hs.ranges[hs.my_seat].probs.clone();

    let incr = b.act(&mut hs, action, &board).expect("river decision");
    assert!(incr == "k" || incr.starts_with('b'), "unopened river: check or bet, got {incr:?}");

    let after = &hs.ranges[hs.my_seat].probs;
    let total: f64 = after.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "range stays normalized, sums to {total}");
    let reshaped = before
        .iter()
        .zip(after)
        .filter(|(&p, _)| p > 0.0)
        .any(|(&p, &q)| (q / p - 1.0).abs() > 1e-3);
    assert!(reshaped, "the resolve's strategy must reshape our range");
    assert!(hs.carried_cfvs.is_some(), "a check or bet leaves a carry for the next resolve");
    assert!(hs.action_luck_chips().is_finite(), "a resolved decision records an action-luck term");
}

/// A desynced postflop spot is re-solved even when that street's re-solving
/// is off: the abstraction has no node there, so the blueprint cannot play
/// it and the pot-odds fallback never raises.  The carry the resolve leaves
/// behind is the observable that the resolver, not the fallback, decided.
#[test]
fn a_desynced_flop_spot_is_resolved_even_with_flop_resolving_off() {
    let action = "b250b750b1657b6750c/b13250";
    let board = cards(&["2c", "7h", "9s"]);
    let mut b = resolve_bot(false, false, vec![0.0]);
    b.cfg.turn_iters = 20;
    b.cfg.runout_sample = 8;
    let h = [parse_card("Ac").unwrap(), parse_card("Ad").unwrap()];
    let mut hs = b.start_hand(1, h);
    let incr = b.act(&mut hs, action, &board).expect("bot acts");
    assert!(parse_action(&format!("{action}{incr}")).is_ok(), "legal response, got {incr:?}");
    assert!(hs.carried_cfvs.is_some(), "the desynced spot went through the resolver");
    let t = b.tracking_counts();
    assert_eq!(t.desynced_decisions, 1);
    // The flop shove arrives after the abstract hand ended: skipped, and
    // counted — the case the counters used to report as 0%.
    assert!(t.unmapped_events >= 1, "skipped events must be counted, got {t}");

    // The blueprint-only ablation keeps the fallback.
    let mut b = resolve_bot(false, false, vec![0.0]);
    b.cfg.resolve_desynced = false;
    let mut hs = b.start_hand(1, h);
    b.act(&mut hs, action, &board).expect("bot acts");
    assert!(hs.carried_cfvs.is_none(), "--no-resolve must not re-solve");
    assert_eq!(b.tracking_counts().desynced_decisions, 1);
}

/// The carry lives within a street.  After a turn decision it exists (the
/// opponent acts next, same street); the river card drops it, because it
/// averaged over that card and bounds no specific one; the river's first
/// resolve then bootstraps a fresh carry.
#[test]
fn the_carry_is_dropped_at_a_street_change() {
    let mut b = resolve_bot(true, false, vec![0.0]);
    b.cfg.resolve_river = true;
    b.cfg.river_iters = 60;
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole); // BB, first to act postflop
    let turn = "b200c/kk/";
    let incr = b.act(&mut hs, turn, &cards(&["Qs", "4s", "3h", "Th"])).expect("turn decision");
    assert!(hs.carried_cfvs.is_some(), "the opponent acts next on the turn: carry kept");

    // The opponent closes the turn (checks behind, or calls our bet).
    let river = if incr == "k" { format!("{turn}kk/") } else { format!("{turn}{incr}c/") };
    let river_board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
    b.sync(&mut hs, &parse_action(&river).unwrap(), &river_board);
    assert!(hs.carried_cfvs.is_none(), "the river card must drop the turn's carry");

    b.act(&mut hs, &river, &river_board).expect("river decision");
    assert!(hs.carried_cfvs.is_some(), "the river's bootstrap resolve leaves a new carry");
}

/// Forced mode plays the move given — even one the agent would rarely draw —
/// records the distribution it would have drawn from, matches a bet to the
/// nearest root action, and counts a move no root action matches.
#[test]
fn act_forced_plays_the_logged_move_and_records_the_distribution() {
    let mut b = bot(false);
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];

    let mut hs = b.start_hand(1, hole); // SB, first in preflop
    assert_eq!(b.act_forced(&mut hs, "", &[], "f").unwrap(), "f");
    let t = b.last_trace().expect("trace recorded");
    assert_eq!(t.path, DecisionPath::Blueprint);
    assert!((t.prob(|_| true) - 1.0).abs() < 1e-9, "trace is a distribution");
    assert!(t.prob(|m| m == RealMove::Fold) > 0.0);

    let mut hs = b.start_hand(1, hole);
    let incr = b.act_forced(&mut hs, "", &[], "b210").unwrap();
    assert!(incr.starts_with('b'), "a bet maps to the nearest raise, got {incr:?}");
    assert_eq!(b.forced_mismatches(), 0);

    // Facing the blind there is no check: counted, and the likeliest move plays.
    let mut hs = b.start_hand(1, hole);
    b.act_forced(&mut hs, "", &[], "k").unwrap();
    assert_eq!(b.forced_mismatches(), 1);
}

/// The floor bounds how far one postflop opponent action can narrow its
/// range.  Hands holding the 2♣ never bet this flop in the policy; after a
/// bet, a plain Bayes update (floor 0) kills them, floor 1 leaves the range
/// exactly as it was, and anything between keeps them alive.
#[test]
fn opp_range_floor_limits_how_far_a_postflop_action_narrows_the_range() {
    let board = cards(&["Qs", "4s", "3h"]);
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let deuce = parse_card("2c").unwrap();
    let game = || BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
    let cfg = |floor| BotConfig { opp_range_floor: floor, resolve_flop: false, search: false, ..Default::default() };

    // Keys at the big blind's flop decision, for every live hand with the 2♣.
    let mut probe = Bot::new(game(), CompactPolicy::from_entries(vec![]), cfg(0.0));
    let mut hs = probe.start_hand(1, hole);
    probe.sync(&mut hs, &parse_action("b200c/").unwrap(), &board);
    let n = hs.hand.actions(&probe.game).len();
    let blocked = |c: u8| board.contains(&c) || hole.contains(&c);
    let deuces: Vec<usize> = (0..NUM_COMBOS)
        .filter(|&i| {
            let [a, c] = combo_cards(i);
            (a == deuce || c == deuce) && !blocked(a) && !blocked(c)
        })
        .collect();
    let mut entries = std::collections::BTreeMap::new();
    for &i in &deuces {
        let mut p = vec![0.0f32; n];
        p[0] = 1.0; // always check
        entries.insert(hs.hand.key_with_hole(&probe.game, combo_cards(i)), p);
    }
    let entries: Vec<(u64, Vec<f32>)> = entries.into_iter().collect();

    let run = |floor: f64| {
        let mut b = Bot::new(game(), CompactPolicy::from_entries(entries.clone()), cfg(floor));
        let mut hs = b.start_hand(1, hole);
        b.sync(&mut hs, &parse_action("b200c/").unwrap(), &board);
        let before = hs.ranges[1 - hs.my_seat].probs.clone();
        b.sync(&mut hs, &parse_action("b200c/b100").unwrap(), &board);
        (before, hs.ranges[1 - hs.my_seat].probs.clone())
    };

    let (before, after) = run(0.0);
    for &i in &deuces {
        assert!(before[i] > 0.0 && after[i] == 0.0, "plain Bayes: a never-bet hand dies");
    }
    let (before, after) = run(1.0);
    for (x, y) in before.iter().zip(&after) {
        assert!((x - y).abs() < 1e-12, "floor 1: the bet must not move the range");
    }
    let (_, after) = run(0.3);
    assert!(deuces.iter().all(|&i| after[i] > 0.0), "floor 0.3 keeps them alive");
}

/// Search end to end: a flop decision goes through the depth-limited
/// search, plays a legal move, conditions our range on what the search
/// played, and advances the mirror like any other decision.
#[test]
fn a_searched_flop_decision_is_legal_and_updates_our_range() {
    let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(2);
    let cfg = BotConfig { search: true, search_iters: 200, resolve_river: false, purify: 0.0, seed: 3, ..Default::default() };
    let mut b = Bot::new(game, CompactPolicy::from_entries(vec![]), cfg);
    let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
    let mut hs = b.start_hand(0, hole); // BB, first to act postflop
    let action = "b200c/";
    let board = cards(&["Qs", "4s", "3h"]);
    b.sync(&mut hs, &parse_action(action).unwrap(), &board);
    let before = hs.ranges[hs.my_seat].probs.clone();

    let incr = b.act(&mut hs, action, &board).expect("flop decision");
    assert!(incr == "k" || incr.starts_with('b'), "unopened flop: check or bet, got {incr:?}");
    assert!(parse_action(&format!("{action}{incr}")).is_ok());
    let t = b.last_trace().unwrap();
    assert_eq!(t.path, DecisionPath::Search);
    assert!((t.prob(|_| true) - 1.0).abs() < 1e-9);

    let after = &hs.ranges[hs.my_seat].probs;
    assert!((after.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(
        before.iter().zip(after).filter(|(&p, _)| p > 0.0).any(|(&p, &q)| (q / p - 1.0).abs() > 1e-3),
        "the search's strategy must reshape our range"
    );
    assert_eq!(hs.pending_self.map(|p| p.is_some()), Some(true), "mirror advanced");
}

/// A spot the mirror has lost has no blueprint node to search from: it
/// takes the usual desynced path instead.
#[test]
fn a_desynced_spot_is_not_searched() {
    let action = "b250b750b1657b6750c/b13250";
    let board = cards(&["2c", "7h", "9s"]);
    let mut b = resolve_bot(false, false, vec![0.0]);
    b.cfg.search = true;
    b.cfg.search_iters = 50;
    b.cfg.runout_sample = 8;
    let h = [parse_card("Ac").unwrap(), parse_card("Ad").unwrap()];
    let mut hs = b.start_hand(1, h);
    b.act(&mut hs, action, &board).expect("bot acts");
    assert_ne!(b.last_trace().unwrap().path, DecisionPath::Search);
}
