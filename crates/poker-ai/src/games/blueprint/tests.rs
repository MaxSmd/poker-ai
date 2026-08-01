//! Tests for the abstracted heads-up blueprint game.

use super::*;
use poker_core::{make_card, rank_of, suit_of};

// NOTE: `Game` and `CursorGame` share method names, so importing both makes
// every `game.method()` call below ambiguous — import only what these tests
// call directly (the cursor path is exercised through `Mccfr::train_fast`).
use crate::games::Game;

use crate::solver::cfr::Variant;
use crate::solver::dcfr::Discount;
use crate::solver::mccfr::Mccfr;

/// Suit-rotate a card by `+1 (mod 4)` — for asserting suit isomorphism.
fn rotate_suit(c: u8) -> u8 {
    make_card(rank_of(c), (suit_of(c) + 1) % 4)
}

/// A tiny deterministic unit source for driving `sample_chance` directly.
fn unit_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed | 1;
    move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn sampled_deal_uses_nine_distinct_real_cards() {
    let game = BlueprintHoldem::new(100, 2, 1, 0);
    let root = game.root();
    assert!(game.is_chance(&root));
    assert!(!game.is_chance_enumerable(&root));

    for seed in 0..200u64 {
        let st = game.sample_chance(&root, unit_stream(seed));
        let gs = st.gs.as_ref().unwrap();
        let mut cards = Vec::new();
        cards.extend_from_slice(&gs.hole_cards[0]);
        cards.extend_from_slice(&gs.hole_cards[1]);
        cards.extend_from_slice(&gs.board);
        assert_eq!(cards.len(), DEAL_CARDS);
        assert!(cards.iter().all(|&c| c < 52), "every dealt card is a real card");
        cards.sort_unstable();
        cards.dedup();
        assert_eq!(cards.len(), DEAL_CARDS, "no card is dealt twice (seed {seed})");
    }
}

#[test]
fn preflop_key_collapses_suit_isomorphic_hands() {
    // Two pre-flop situations that differ only by a global suit rotation must
    // share an information key (same 169-class), and they must differ from a
    // genuinely different starting hand.
    let game = BlueprintHoldem::new(100, 2, 1, 0);
    let mk = |holes: [[u8; 2]; 2]| {
        let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
        h[0] = holes[0];
        h[1] = holes[1];
        let board = [NO_CARD; 5];
        let gs = GameState::new(2, 2, 1, game.stacks, h, board, 0);
        BlueprintState { gs: Some(gs), history: Vec::new(), street_raises: 0 }
    };
    // A♠K♠ vs 7♦7♣  →  rotate every suit  →  A♥K♥ vs 7♣7♠.
    let base = mk([[make_card(12, 0), make_card(11, 0)], [make_card(5, 1), make_card(5, 2)]]);
    let rot = mk([
        [rotate_suit(make_card(12, 0)), rotate_suit(make_card(11, 0))],
        [rotate_suit(make_card(5, 1)), rotate_suit(make_card(5, 2))],
    ]);
    // The acting pre-flop player is the same in both; keys must match.
    assert_eq!(game.info_key(&base), game.info_key(&rot));

    // A different starting hand (Q♠J♠) keys differently.
    let other = mk([[make_card(10, 0), make_card(9, 0)], [make_card(5, 1), make_card(5, 2)]]);
    assert_ne!(game.info_key(&base), game.info_key(&other));
}

#[test]
fn mccfr_runs_over_sampled_blueprint() {
    // The keystone smoke test: external sampling drives the real engine
    // through sampled deals + bucketed keys, completes, and produces valid
    // probability distributions at every discovered info set.
    let game = BlueprintHoldem::new(40, 2, 1, 0);
    let mut solver = Mccfr::new(game, Variant::Dcfr(Discount::RECOMMENDED));
    solver.train(2_000);
    assert!(solver.num_info_sets() > 0, "should discover info sets");
    for (_key, probs) in solver.average_strategy() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
        assert!(probs.iter().all(|&p| p >= 0.0));
    }
}

#[test]
fn baseline_mccfr_runs_over_sampled_blueprint() {
    // The VR-MCCFR chance baseline must gracefully no-op on a non-enumerable
    // chance node (no outcome list to index) yet still train cleanly.
    let game = BlueprintHoldem::new(40, 2, 1, 0);
    let mut solver = Mccfr::new(game, Variant::Vanilla).with_baseline();
    solver.train(1_000);
    assert!(solver.num_info_sets() > 0);
}

#[test]
fn is_deterministic_for_fixed_seed() {
    let run = || {
        let game = BlueprintHoldem::new(40, 2, 1, 0);
        let mut s = Mccfr::with_seed(game, Variant::Vanilla, 99);
        s.train(1_000);
        s.num_info_sets()
    };
    assert_eq!(run(), run(), "same seed must visit the same info sets");
}

#[test]
fn raise_cap_removes_sized_raises_but_never_the_all_in() {
    let game = BlueprintHoldem::new(200, 2, 1, 0).with_raise_cap(1);
    // Deep-stacked heads-up preflop: SB to act faces a raise/all-in menu.
    let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
    h[0] = [make_card(12, 0), make_card(11, 0)];
    h[1] = [make_card(5, 1), make_card(5, 2)];
    let gs = GameState::new(2, 2, 1, game.stacks, h, [NO_CARD; 5], 0);

    // Below the cap (0 raises so far) the opening raise is still offered.
    let under = game.capped_legal(&gs, 0);
    assert!(
        under.iter().any(|a| matches!(a, Action::Raise(_))),
        "opening raise must be legal below the cap, got {under:?}"
    );

    // At the cap, sized raises are gone but all-in remains: the abstraction
    // must stay closed under aggression, so a raise war always has a
    // terminating action for the tracker to map an opponent's shove onto.
    let at = game.capped_legal(&gs, 1);
    assert!(
        at.iter().all(|a| !matches!(a, Action::Raise(_))),
        "no sized reraise at the cap, got {at:?}"
    );
    assert!(
        at.iter().any(|a| matches!(a, Action::AllIn)),
        "all-in must survive the cap, got {at:?}"
    );
    assert!(
        at.iter().any(|a| matches!(a, Action::Fold | Action::Call | Action::Check)),
        "a passive action must remain, got {at:?}"
    );

    // The uncapped default never filters, however many raises have happened.
    let uncapped = BlueprintHoldem::new(200, 2, 1, 0);
    assert!(uncapped.capped_legal(&gs, 9).iter().any(|a| matches!(a, Action::Raise(_))));
}

/// The property the fix exists for: from any node, however many raises have
/// already gone in, the acting player can still put the rest of the stack
/// in.  Nothing an opponent does can leave the abstraction without an
/// aggressive action to translate their bet onto.
#[test]
fn aggression_always_has_a_landing_spot_at_the_cap() {
    let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
    let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
    h[0] = [make_card(12, 0), make_card(11, 0)];
    h[1] = [make_card(5, 1), make_card(5, 2)];
    let gs = GameState::new(2, 2, 1, game.stacks, h, [NO_CARD; 5], 0);

    for raises in 0..8u8 {
        let acts = game.capped_legal(&gs, raises);
        assert!(
            acts.iter().any(|a| matches!(a, Action::Raise(_) | Action::AllIn)),
            "no aggressive action after {raises} raises: {acts:?}"
        );
    }
}

#[test]
fn capped_clone_and_cursor_paths_agree() {
    // The cursor path maintains `street_raises` in place via apply/undo; it
    // must visit exactly the same (capped) info sets as the clone path.
    use crate::games::CursorGame;
    let mk = || BlueprintHoldem::new(40, 2, 1, 0).with_raise_cap(1);
    let _ = CursorGame::root(&mk()); // ensure the capped game is a CursorGame

    let mut clone_path = Mccfr::with_seed(mk(), Variant::Dcfr(Discount::RECOMMENDED), 5);
    clone_path.train(500);
    let mut cursor_path = Mccfr::with_seed(mk(), Variant::Dcfr(Discount::RECOMMENDED), 5);
    cursor_path.train_fast(500);
    assert_eq!(
        clone_path.num_info_sets(),
        cursor_path.num_info_sets(),
        "capped legal lists must match between the clone and cursor paths"
    );
}

#[test]
fn indexed_preflop_only_partition_and_key_round_trip() {
    use crate::games::{CursorGame, IndexedGame};
    // stack == big blind: the BB is all-in from its blind, so the SB faces a
    // single fold/all-in decision and no post-flop node is ever created.  The
    // placeholder maps are therefore never queried — this keeps the test O(1)
    // while exercising the full IndexedGame plumbing (capacity, index,
    // actions_at, and the info_key_at export inverse).
    let game = BlueprintHoldem::new(2, 2, 1, 0)
        .with_raise_cap(1)
        .with_street_bucket(0, BucketMap::placeholder(&[2, 3], 50))
        .with_street_bucket(1, BucketMap::placeholder(&[2, 4], 50))
        .with_street_bucket(2, BucketMap::placeholder(&[2, 5], 50))
        .with_indexing();

    let cap = game.info_set_capacity();
    assert!(cap >= 169 && cap.is_multiple_of(169), "preflop-only capacity is a multiple of 169, got {cap}");

    // The dense index and the HashMap info key must induce the SAME partition,
    // and info_key_at must invert the index back to that key.
    let mut by_key: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut by_idx: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
    for seed in 0..500u64 {
        let mut c = CursorGame::root(&game);
        CursorGame::sample_chance(&game, &mut c, unit_stream(seed));
        let key = CursorGame::info_key(&game, &c);
        let idx = game.info_set_index(&c);
        assert!(idx < cap, "index in range");
        assert_eq!(game.actions_at(idx), 2, "fold/all-in menu");
        assert_eq!(game.info_key_at(idx), key, "info_key_at inverts the dense index");
        assert_eq!(*by_key.entry(key).or_insert(idx), idx, "key -> one index");
        assert_eq!(*by_idx.entry(idx).or_insert(key), key, "index -> one key");
    }
    assert!(by_key.len() > 100, "should see many distinct starting-hand classes");
}

/// Full post-flop coverage: builds the turn/river full-coverage maps (~280 MB)
/// so the dense index has no out-of-set situation.  Confirms the dense index
/// partitions information sets identically to the `HashMap` key on every
/// street, that `info_key_at` inverts it, and that the SoA solver trains over
/// the indexed full game to valid distributions.
///   cargo test -p poker-ai --release -- --ignored indexed_blueprint_postflop_and_soa
/// Throughput comparison of the three SoA training paths on a realistic
/// indexed blueprint tree (20 bb, cap-2) — the parallel-scaling
/// deliverable.  Prints nodes/s per configuration; the assertions are a
/// loose sanity ordering so a busy machine cannot flake the test.
///   cargo test -p poker-ai --release -- --ignored --nocapture atomic_scaling
#[test]
#[ignore]
fn atomic_scaling_benchmark() {
    use crate::solver::cfr::Variant;
    use crate::solver::dcfr::Discount;
    use crate::solver::mccfr::SoaMccfr;
    use std::time::Instant;

    let mk = || {
        BlueprintHoldem::new(40, 2, 1, 0)
            .with_raise_cap(2)
            .with_street_bucket(0, BucketMap::full_coverage_mod(&[2, 3], 40))
            .with_street_bucket(1, BucketMap::full_coverage_mod(&[2, 4], 40))
            .with_street_bucket(2, BucketMap::full_coverage_mod(&[2, 5], 40))
            .with_indexing()
    };
    let iters = 200_000u64;
    let bench = |name: &str, f: &mut dyn FnMut(&mut SoaMccfr<BlueprintHoldem>)| -> f64 {
        let mut s =
            SoaMccfr::with_seed(mk(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
        let t0 = Instant::now();
        f(&mut s);
        let secs = t0.elapsed().as_secs_f64();
        let nps = s.nodes_visited() as f64 / secs;
        println!("{name:>16}: {secs:6.2}s  {nps:>12.0} nodes/s");
        nps
    };

    let serial = bench("serial", &mut |s| s.train(iters));
    let parallel = bench("parallel(512)", &mut |s| s.train_parallel(iters, 512));
    let mut atomic_best = 0.0f64;
    for threads in [1usize, 2, 4, 8] {
        let name = format!("atomic({threads})");
        let nps = bench(&name, &mut |s| s.train_atomic(iters, threads));
        atomic_best = atomic_best.max(nps);
    }
    assert!(atomic_best > serial, "atomic best {atomic_best} should beat serial {serial}");
    assert!(
        atomic_best > parallel,
        "atomic best {atomic_best} should beat batched parallel {parallel}"
    );
}

#[test]
#[ignore]
fn indexed_blueprint_postflop_and_soa() {
    use crate::games::{CursorGame, IndexedGame};
    use crate::solver::mccfr::SoaMccfr;

    let mk = || {
        BlueprintHoldem::new(12, 2, 1, 0) // 6bb: check lines reach every street under cap 1
            .with_raise_cap(1)
            .with_street_bucket(0, BucketMap::full_coverage_mod(&[2, 3], 40))
            .with_street_bucket(1, BucketMap::full_coverage_mod(&[2, 4], 40))
            .with_street_bucket(2, BucketMap::full_coverage_mod(&[2, 5], 40))
            .with_indexing()
    };
    let game = mk();
    let cap = game.info_set_capacity();
    assert!(cap > 0, "non-empty index");

    // Roll out full hands, checking the partition + round-trip at every
    // decision node on every street.
    let mut by_key: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut rng = 0x00C0_FFEEu64;
    let mut next = || {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..3000 {
        let mut c = CursorGame::root(&game);
        CursorGame::sample_chance(&game, &mut c, &mut next);
        while !CursorGame::is_terminal(&game, &c) {
            let key = CursorGame::info_key(&game, &c);
            let idx = game.info_set_index(&c);
            assert!(idx < cap, "index in range");
            assert_eq!(game.info_key_at(idx), key, "info_key_at inverts the dense index");
            assert_eq!(*by_key.entry(key).or_insert(idx), idx, "key -> one index (same partition)");
            let acts = CursorGame::legal(&game, &c);
            let n = acts.as_ref().len();
            let a = ((next() * n as f64) as usize).min(n - 1);
            CursorGame::apply(&game, &mut c, a, acts.as_ref()[a]);
        }
    }
    assert!(by_key.len() > 200, "should exercise many post-flop info sets, got {}", by_key.len());

    // The SoA solver trains over the indexed full game and yields valid
    // probability distributions at every visited info set.
    let mut soa: SoaMccfr<BlueprintHoldem> =
        SoaMccfr::with_seed(mk(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
    soa.train(20_000);
    let mut visited = 0;
    for i in 0..soa.capacity() {
        if soa.is_visited(i) {
            visited += 1;
            let p = soa.average_strategy_at(i);
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "distribution at {i} sums to {sum}");
            assert!(p.iter().all(|&x| x >= 0.0));
        }
    }
    assert!(visited > 100, "SoA training should visit many info sets, got {visited}");
}
