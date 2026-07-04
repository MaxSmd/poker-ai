//! Heads-up push/fold NLHE — the first *converging* blueprint over real
//! mechanics (Phase 1.5).
//!
//! [`super::blueprint::BlueprintHoldem`] is the full game, but it cannot
//! converge locally: without a complete postflop card abstraction (the
//! cloud-scale equity precompute), every randomly-dealt board mints fresh,
//! once-visited postflop information sets and the tree never plateaus.  Push/fold
//! removes the problem at the root: at every decision the only choices are
//! **fold** or **commit all-in**.  The flat-call / limp line — the sole source
//! of postflop play — is gone, so the tree is two levels deep at *any* stack
//! depth:
//!
//! ```text
//!   SB:  fold  | shove
//!   BB (vs shove):  fold | call
//! ```
//!
//! That is ~169 + 169 information sets (one decision per suit-canonical starting
//! hand per player), it plateaus, and it has a well-known Nash solution to
//! validate against — exactly the "prove it on a known-solution game
//! first" discipline, now over the real `poker-core` engine: the all-in runout
//! and showdown come from the real evaluator, and payoffs are real chip deltas.
//!
//! Chance (the full deal) is sampled, not enumerated, so the game reuses the
//! same [`Game::sample_chance`] path the blueprint introduced.

use poker_core::action::Action;
use poker_core::legal_actions;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use super::Game;
use crate::util::hash::Fnv1a;
use crate::abstraction::canonical::preflop_index;

/// Cards consumed by a heads-up deal: 2 hole cards each + 5 board.
const DEAL_CARDS: usize = 9;

/// Maximum decisions in a push/fold hand (SB then BB) — the cursor's inline
/// history is sized to this with generous headroom.
const MAX_DEPTH: usize = 4;

/// A heads-up push/fold NLHE game with sampled deals.
pub struct PushFoldHoldem {
    stacks: [u32; MAX_PLAYERS],
    big_blind: u32,
    small_blind: u32,
    button: u8,
}

/// A node: the pre-deal chance root (`gs == None`) or a play node.
#[derive(Clone, Debug)]
pub struct PushFoldState {
    gs: Option<GameState>,
    /// Perfect-recall action history (`0 = fold, 1 = commit`).
    history: Vec<u8>,
}

impl PushFoldHoldem {
    /// A game with equal starting stacks (`stack` chips each).  A realistic
    /// short-stack scenario uses, e.g., `stack = 25 * big_blind`.
    pub fn new(stack: u32, big_blind: u32, small_blind: u32, button: u8) -> Self {
        let mut stacks = [0u32; MAX_PLAYERS];
        stacks[0] = stack;
        stacks[1] = stack;
        Self { stacks, big_blind, small_blind, button }
    }

    /// Deal both hands + the full board from a freshly shuffled deck.
    fn deal(&self, mut next_unit: impl FnMut() -> f64) -> GameState {
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        let last = 51;
        for i in 0..DEAL_CARDS {
            let span = 52 - i;
            let j = (i + (next_unit() * span as f64) as usize).min(last);
            deck.swap(i, j);
        }
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [deck[0], deck[1]];
        holes[1] = [deck[2], deck[3]];
        let board = [deck[4], deck[5], deck[6], deck[7], deck[8]];
        GameState::new(2, self.big_blind, self.small_blind, self.stacks, holes, board, self.button)
    }

    /// The two-action push/fold menu at a decision node: `[Fold, commit]`, where
    /// *commit* is `AllIn` if available, else `Call` (calling a shove is itself
    /// all-in).  Restricting to this menu is what removes postflop play.
    fn menu(gs: &GameState) -> [Action; 2] {
        let acts = legal_actions(gs);
        let mut commit = None;
        let mut has_fold = false;
        for &a in acts.iter() {
            match a {
                Action::Fold => has_fold = true,
                Action::AllIn => commit = Some(Action::AllIn),
                Action::Call if commit.is_none() => commit = Some(Action::Call),
                _ => {}
            }
        }
        debug_assert!(has_fold, "push/fold decision nodes always allow folding");
        [Action::Fold, commit.expect("a chips-committing action is always available")]
    }

    /// Fold the SB-open / BB-vs-shove information-set key, streamed straight into
    /// FNV-1a so neither the clone-based nor the cursor-based path allocates.
    /// Pre-flop only: the 169-class pre-flop index plus the perfect-recall
    /// history.
    fn info_key_for(gs: &GameState, history: &[u8]) -> u64 {
        let player = gs.current_player();
        let hole = gs.hole_cards[player];
        Self::preflop_key(player, &hole, history)
    }

    /// The information-set key for a pre-flop `(player, hole, history)`.  Public
    /// so the trainer's shove-chart reconstruction stays in lock-step with the
    /// solver (a wrong reconstruction silently reads the wrong strategy).
    pub fn preflop_key(player: usize, hole: &[u8; 2], history: &[u8]) -> u64 {
        let mut sorted = *hole;
        sorted.sort_unstable();
        let class = preflop_index(&sorted);

        let mut h = Fnv1a::new();
        h.write(player as u8);
        h.write_all(&class.to_le_bytes());
        h.write(0xFF);
        h.write_all(history);
        h.finish()
    }
}

impl Game for PushFoldHoldem {
    type State = PushFoldState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> PushFoldState {
        PushFoldState { gs: None, history: Vec::new() }
    }

    fn is_terminal(&self, state: &PushFoldState) -> bool {
        state.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, state: &PushFoldState) -> bool {
        state.gs.is_none()
    }

    fn is_chance_enumerable(&self, _state: &PushFoldState) -> bool {
        false
    }

    fn utility(&self, state: &PushFoldState, player: usize) -> f64 {
        let gs = state.gs.as_ref().expect("utility at a play node");
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    fn chance_outcomes(&self, _state: &PushFoldState) -> Vec<(PushFoldState, f64)> {
        unimplemented!("PushFoldHoldem chance is not enumerable; use sample_chance")
    }

    fn sample_chance(
        &self,
        _state: &PushFoldState,
        next_unit: impl FnMut() -> f64,
    ) -> PushFoldState {
        PushFoldState { gs: Some(self.deal(next_unit)), history: Vec::new() }
    }

    fn current_player(&self, state: &PushFoldState) -> usize {
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, _state: &PushFoldState) -> usize {
        2
    }

    fn apply(&self, state: &PushFoldState, action: usize) -> PushFoldState {
        let gs = state.gs.as_ref().expect("apply at a play node");
        let act = Self::menu(gs)[action];
        let mut next_gs = gs.clone();
        next_gs.apply_action(act);
        let mut history = state.history.clone();
        history.push(action as u8);
        PushFoldState { gs: Some(next_gs), history }
    }

    fn info_key(&self, state: &PushFoldState) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        Self::info_key_for(gs, &state.history)
    }
}

/// A zero-allocation traversal cursor for [`PushFoldHoldem`]: one `GameState`
/// walked in place, plus an inline perfect-recall history.
pub struct PushFoldCursor {
    /// `None` at the pre-deal chance root; `Some` once a deal has been sampled.
    gs: Option<GameState>,
    /// Action indices taken from the root (`0 = fold, 1 = commit`).
    history: [u8; MAX_DEPTH],
    /// Current depth — number of valid entries in `history`.
    depth: usize,
}

impl super::CursorGame for PushFoldHoldem {
    type Cursor = PushFoldCursor;
    type Action = Action;
    type Actions = [Action; 2];

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> PushFoldCursor {
        PushFoldCursor { gs: None, history: [0; MAX_DEPTH], depth: 0 }
    }

    fn is_terminal(&self, c: &PushFoldCursor) -> bool {
        c.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, c: &PushFoldCursor) -> bool {
        c.gs.is_none()
    }

    fn utility(&self, c: &PushFoldCursor, player: usize) -> f64 {
        let gs = c.gs.as_ref().expect("utility at a play node");
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    fn current_player(&self, c: &PushFoldCursor) -> usize {
        c.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn legal(&self, c: &PushFoldCursor) -> [Action; 2] {
        Self::menu(c.gs.as_ref().expect("legal at a play node"))
    }

    fn info_key(&self, c: &PushFoldCursor) -> u64 {
        let gs = c.gs.as_ref().expect("info_key at a play node");
        Self::info_key_for(gs, &c.history[..c.depth])
    }

    fn apply(&self, c: &mut PushFoldCursor, a: usize, action: Action) {
        c.gs.as_mut().expect("apply at a play node").apply_action(action);
        c.history[c.depth] = a as u8;
        c.depth += 1;
    }

    fn undo(&self, c: &mut PushFoldCursor) {
        c.depth -= 1;
        c.gs.as_mut().expect("undo at a play node").undo_action();
    }

    fn sample_chance(&self, c: &mut PushFoldCursor, next_unit: impl FnMut() -> f64) {
        c.gs = Some(self.deal(next_unit));
        c.depth = 0;
    }

    fn undo_chance(&self, c: &mut PushFoldCursor) {
        c.gs = None;
        c.depth = 0;
    }
}

/// The 169 suit-canonical pre-flop starting-hand classes.
const PREFLOP_CLASSES: usize = 169;

impl super::IndexedGame for PushFoldHoldem {
    /// Two betting sequences (SB open, BB vs shove) × 169 pre-flop classes.
    fn info_set_capacity(&self) -> usize {
        2 * PREFLOP_CLASSES
    }

    /// `sequence · 169 + preflop_class`, where the sequence is the decision depth
    /// (0 = SB open with empty history, 1 = BB facing the shove).  This is a
    /// bijection onto `0..338` and partitions exactly as the `HashMap` info key.
    fn info_set_index(&self, c: &PushFoldCursor) -> usize {
        debug_assert!(c.depth < 2, "push/fold has at most two decisions per hand");
        let gs = c.gs.as_ref().expect("info_set_index at a play node");
        let player = gs.current_player();
        let mut hole = gs.hole_cards[player];
        hole.sort_unstable();
        c.depth * PREFLOP_CLASSES + preflop_index(&hole) as usize
    }

    fn actions_at(&self, _index: usize) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::canonical::preflop_index;
    use crate::solver::cfr::Variant;
    use crate::solver::dcfr::Discount;
    use crate::solver::mccfr::{LeanMccfr, Mccfr, SoaMccfr};
    use crate::solver::regret_table::RegretStore;
    use poker_core::make_card;
    use std::collections::HashMap;

    #[test]
    fn tree_is_two_levels_and_bounded() {
        // The defining property: info-set count plateaus (no postflop).
        let game = PushFoldHoldem::new(50, 2, 1, 0);
        let mut s = Mccfr::with_seed(game, Variant::Vanilla, 1);
        s.train(40_000);
        let a = s.num_info_sets();
        s.train(60_000);
        let b = s.num_info_sets();
        assert_eq!(a, b, "info-set count must plateau ({a} -> {b})");
        // Two players × 169 classes, minus hands never reached; comfortably < 400.
        assert!(b <= 400, "push/fold has ~338 info sets, got {b}");
        assert!(b > 100, "should discover most starting-hand classes, got {b}");
    }

    #[test]
    fn payoffs_are_zero_sum() {
        // Drive a few sampled hands to terminal and check chips are conserved.
        let game = PushFoldHoldem::new(50, 2, 1, 0);
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..50 {
            let mut st = game.sample_chance(&game.root(), &mut next);
            while !game.is_terminal(&st) {
                // Always take the "commit" branch so we reach showdowns too.
                let a = if game.num_actions(&st) == 2 { 1 } else { 0 };
                st = game.apply(&st, a);
            }
            let (u0, u1) = (game.utility(&st, 0), game.utility(&st, 1));
            assert!((u0 + u1).abs() < 1e-9, "payoffs must sum to zero: {u0} + {u1}");
        }
    }

    #[test]
    fn strategy_is_monotone_premium_shoves_more_than_trash() {
        // A sanity check against the known solution shape: the SB shoves a strong
        // hand more often than a weak one.  We read the opening (no-history) SB
        // node for AA vs 72o.
        let game = PushFoldHoldem::new(40, 2, 1, 0);
        let mut s = Mccfr::with_seed(game, Variant::Dcfr(Discount::RECOMMENDED), 1);
        s.train(150_000);
        let avg = s.average_strategy();

        let game = PushFoldHoldem::new(40, 2, 1, 0);
        let shove_prob = |hole: [u8; 2]| -> f64 {
            // Reconstruct the SB opening info key for this exact hand (player 0,
            // empty history) via the same helper the solver keys on.
            let key = PushFoldHoldem::preflop_key(0, &hole, &[]);
            avg.get(&key).map(|p| p[1]).unwrap_or(0.0) // p[1] = commit/shove
        };
        let _ = &game;
        let aces = shove_prob([make_card(12, 0), make_card(12, 1)]);
        let trash = shove_prob([make_card(5, 0), make_card(0, 1)]); // 7-2 offsuit
        assert!(aces > trash, "AA shove {aces} should exceed 72o shove {trash}");
        assert!(aces > 0.9, "AA should shove almost always, got {aces}");
    }

    #[test]
    fn soa_table_footprint_is_tiny() {
        // The flat store holds 3 f32 accumulators × 2 actions = 24 bytes/info set,
        // versus the HashMap Node's five heap vecs (~350 B) — the ~10× the memory
        // budget is about.
        let soa: SoaMccfr<PushFoldHoldem> = SoaMccfr::new(PushFoldHoldem::new(40, 2, 1, 0), Variant::Vanilla);
        assert_eq!(soa.bytes_per_info_set(), 24);
    }

    /// Export a trained SoA solver back to the `HashMap` strategy artifact
    /// (`preflop_key`-keyed), so the exploitability estimator can score it.
    fn export_soa<S: RegretStore>(
        soa: &SoaMccfr<PushFoldHoldem, S>,
    ) -> std::collections::HashMap<u64, Vec<f64>> {
        let mut out = std::collections::HashMap::new();
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let hole = [a, b];
                let class = preflop_index(&hole) as usize;
                for (depth, history) in [(0usize, &[][..]), (1, &[1u8][..])] {
                    let idx = depth * 169 + class;
                    if soa.is_visited(idx) {
                        let key = PushFoldHoldem::preflop_key(depth, &hole, history);
                        out.entry(key).or_insert_with(|| soa.average_strategy_at(idx));
                    }
                }
            }
        }
        out
    }

    /// The lean-store experiment's verdict gate: the i16/u16 quantized table
    /// under Linear CFR must train to a solution as close to Nash as the f32
    /// table under DCFR, at half the accumulator bytes.  Exploitability-gated
    /// for the same reason as the atomic test below (independent runs differ
    /// per-hand by Monte-Carlo noise).
    #[test]
    fn lean_lcfr_store_converges_like_the_f32_dcfr_store() {
        use crate::evaluation::exploitability::push_fold_exploitability;
        use std::time::Instant;
        let iters = 1_000_000;
        let cfg = || PushFoldHoldem::new(40, 2, 1, 0);

        let t0 = Instant::now();
        let mut f32s: SoaMccfr<PushFoldHoldem> =
            SoaMccfr::with_seed(cfg(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
        f32s.train(iters);
        let f32_secs = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let mut lean: LeanMccfr<PushFoldHoldem> =
            LeanMccfr::with_seed(cfg(), Variant::Dcfr(Discount::LINEAR), 1).with_baseline();
        lean.train(iters);
        let lean_secs = t0.elapsed().as_secs_f64();

        let game = cfg();
        let expl_f32 = push_fold_exploitability(&game, &export_soa(&f32s), 200_000, 9);
        let expl_lean = push_fold_exploitability(&game, &export_soa(&lean), 200_000, 9);
        println!(
            "f32+DCFR: {} B/infoset, {f32_secs:.2}s, expl {expl_f32:.4} bb | \
             lean+LCFR: {} B/infoset, {lean_secs:.2}s, expl {expl_lean:.4} bb",
            f32s.bytes_per_info_set(),
            lean.bytes_per_info_set(),
        );
        assert_eq!(lean.bytes_per_info_set() * 2, f32s.bytes_per_info_set(), "half the accumulator bytes");
        assert!(expl_lean < 0.12, "lean-trained strategy exploitability {expl_lean} bb too high");
        assert!(
            expl_lean < expl_f32 + 0.04,
            "lean+LCFR ({expl_lean} bb) materially worse than f32+DCFR ({expl_f32} bb)"
        );

        let aa = lean.average_strategy_at(preflop_index(&[make_card(12, 0), make_card(12, 1)]) as usize)[1];
        let trash = lean.average_strategy_at(preflop_index(&[make_card(5, 0), make_card(0, 1)]) as usize)[1];
        assert!(aa > 0.9 && aa > trash, "lean chart shape: AA {aa} vs 72o {trash}");
    }

    #[test]
    fn atomic_converges_like_the_serial_soa() {
        // The lock-free atomic trainer must reach a solution as close to Nash
        // as the serial SoA reference.  It is NOT bit-deterministic (thread
        // interleaving races float updates — the documented trade), and its
        // per-iteration RNG streams differ from the serial chain, so per-hand
        // probabilities are only comparable up to Monte-Carlo noise (~0.05 mean
        // at this budget even between two SERIAL seeds).  The principled gate
        // is therefore exploitability: both strategies must be near-Nash, and
        // the atomic one no worse than the serial one beyond estimator noise.
        use crate::evaluation::exploitability::push_fold_exploitability;
        let iters = 1_000_000;
        let cfg = || PushFoldHoldem::new(40, 2, 1, 0);

        let mut serial: SoaMccfr<PushFoldHoldem> =
            SoaMccfr::with_seed(cfg(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
        serial.train(iters);

        let mut atomic = SoaMccfr::with_seed(cfg(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
        atomic.train_atomic(iters, 4);

        let game = cfg();
        let expl_serial = push_fold_exploitability(&game, &export_soa(&serial), 200_000, 9);
        let expl_atomic = push_fold_exploitability(&game, &export_soa(&atomic), 200_000, 9);
        assert!(expl_atomic < 0.12, "atomic-trained strategy exploitability {expl_atomic} bb too high");
        assert!(
            expl_atomic < expl_serial + 0.03,
            "atomic ({expl_atomic} bb) materially worse than serial ({expl_serial} bb)"
        );

        let aa = atomic.average_strategy_at(preflop_index(&[make_card(12, 0), make_card(12, 1)]) as usize)[1];
        let trash = atomic.average_strategy_at(preflop_index(&[make_card(5, 0), make_card(0, 1)]) as usize)[1];
        assert!(aa > 0.9 && aa > trash, "atomic chart shape: AA {aa} vs 72o {trash}");
    }

    #[test]
    fn soa_converges_like_the_hashmap_solver() {
        // The flat f32 SoA store reaches the same push/fold solution as the f64
        // HashMap reference (within tolerance — f32 storage and a separate code
        // path mean it is not bit-identical, see the module note).
        let iters = 150_000;
        let cfg = || PushFoldHoldem::new(40, 2, 1, 0);

        let mut hash = Mccfr::with_seed(cfg(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
        hash.train_fast(iters);
        let havg = hash.average_strategy();

        let mut soa: SoaMccfr<PushFoldHoldem> =
            SoaMccfr::with_seed(cfg(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
        soa.train(iters);

        // Compare the SB opening shove probability for every starting hand.
        let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0u32);
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let hole = [a, b];
                let h = havg.get(&PushFoldHoldem::preflop_key(0, &hole, &[])).map(|p| p[1]).unwrap_or(0.0);
                let s = soa.average_strategy_at(preflop_index(&hole) as usize)[1];
                let d = (h - s).abs();
                sum += d;
                max = max.max(d);
                n += 1;
            }
        }
        let mean = sum / n as f64;
        assert!(mean < 0.03, "SoA vs HashMap mean SB-shove diff {mean} too large");
        assert!(max < 0.15, "SoA vs HashMap max SB-shove diff {max} too large");

        let aa = soa.average_strategy_at(preflop_index(&[make_card(12, 0), make_card(12, 1)]) as usize)[1];
        let trash = soa.average_strategy_at(preflop_index(&[make_card(5, 0), make_card(0, 1)]) as usize)[1];
        assert!(aa > 0.9 && aa > trash, "SoA chart shape: AA {aa} vs 72o {trash}");
    }

    #[test]
    fn soa_checkpoint_round_trips() {
        // A flat-table checkpoint resumes bit-identically (f32 arrays + RNG +
        // counters round-trip exactly through bincode).
        let cfg = || PushFoldHoldem::new(40, 2, 1, 0);
        let mut whole: SoaMccfr<PushFoldHoldem> =
            SoaMccfr::with_seed(cfg(), Variant::Dcfr(Discount::RECOMMENDED), 11).with_baseline();
        whole.train(40_000);

        let mut part = SoaMccfr::with_seed(cfg(), Variant::Dcfr(Discount::RECOMMENDED), 11).with_baseline();
        part.train(20_000);
        let path = std::env::temp_dir().join(format!("soa_ckpt_{}.bin", std::process::id()));
        part.save_checkpoint(&path).unwrap();
        drop(part);

        let mut resumed = SoaMccfr::load_checkpoint(&path, cfg()).unwrap();
        assert_eq!(resumed.iterations(), 20_000);
        resumed.train(20_000);
        std::fs::remove_file(&path).ok();

        for idx in 0..2 * 169 {
            assert_eq!(
                whole.average_strategy_at(idx),
                resumed.average_strategy_at(idx),
                "SoA resume must be bit-identical at info set {idx}"
            );
        }
    }

    /// The push/fold equilibrium is computed without the curated-deal trick, so
    /// it doubles as a reusable fixture for later checks.
    #[test]
    fn average_strategy_is_valid_distribution() {
        let game = PushFoldHoldem::new(30, 2, 1, 0);
        let mut s = Mccfr::with_seed(game, Variant::Vanilla, 1);
        s.train(10_000);
        let avg: HashMap<u64, Vec<f64>> = s.average_strategy();
        assert!(!avg.is_empty());
        for probs in avg.values() {
            assert_eq!(probs.len(), 2);
            assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }
}
