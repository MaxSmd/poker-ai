//! Lock-free atomic training for the SoA blueprint solver — the many-core path.
//!
//! ## Why this exists
//!
//! The deterministic mini-batch scheme ([`SoaMccfr::train_parallel`]) tops out
//! around ~7 effective cores on a 64-core server: each external-sampling
//! iteration is only microseconds of work on these betting trees, so per-batch
//! dispatch plus the **serial** in-iteration-order delta merge dominate
//! (measured on the first real server run).  This module removes both: worker
//! threads claim iteration numbers from a shared counter and update the flat
//! regret table **in place** with per-slot compare-and-swap — no deltas, no
//! merge, no barriers.  This is the standard large-scale CFR design
//! (Pluribus-style "Hogwild" updates): regret updates commute approximately,
//! and CFR is robust to the slightly stale values racing threads read.
//!
//! ## What is traded away
//!
//! **Bit-determinism.**  Thread interleaving changes float update order, so two
//! runs with the same seed differ in the last bits (a single-threaded run is
//! still reproducible).  The deterministic paths (`train`, `train_parallel`)
//! are untouched and remain the correctness reference; this path is validated
//! the same way the SoA store itself was — by converging to the same solution
//! within tolerance (`atomic_converges_like_the_serial_soa`), not by
//! bit-equality.  Checkpoints remain exact snapshots of whatever state the run
//! reached.
//!
//! ## Memory ordering
//!
//! All atomics are `Relaxed`: every slot is an independent accumulator with no
//! cross-location invariant, and the algorithm tolerates stale reads by
//! design.  Exclusive `&mut RegretTable` is held for the whole run (see
//! [`RegretTable::atomic_parts`]), so no non-atomic access can race.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::parallel::splitmix;
use crate::games::{CursorGame, IndexedGame};
use crate::solver::cfr::Variant;
use crate::solver::regret_table::RegretTable;
use crate::util::rng::{sample_index, xorshift_next_unit};

use super::BASELINE_RATE;

/// Widest action fan-out the stack buffers support (the engine's `ActionList`
/// caps at 8; asserted against the table layout before training starts).
pub(super) const MAX_ACTIONS: usize = 8;

/// Shared atomic view over the three flat accumulator arrays.  Constructed
/// from an exclusive borrow of the table; all access goes through per-slot
/// `AtomicU32` (f32 bit-cast) operations.
struct AtomicTable<'a> {
    regret: *mut f32,
    strategy_sum: *mut f32,
    baseline: *mut f32,
    offsets: &'a [u32],
    num_actions: &'a [u8],
}

// Soundness: the raw pointers target arrays borrowed exclusively for the whole
// run, and every access goes through atomics — concurrent use is data-race-free.
unsafe impl Send for AtomicTable<'_> {}
unsafe impl Sync for AtomicTable<'_> {}

#[inline]
fn atomic_at(ptr: *mut f32, slot: usize) -> &'static AtomicU32 {
    // Safety: `slot` is within the array (layout-derived), f32 is 4-aligned,
    // and the exclusive table borrow outlives every thread in the scope.
    unsafe { AtomicU32::from_ptr(ptr.add(slot).cast()) }
}

#[inline]
fn load(ptr: *mut f32, slot: usize) -> f32 {
    f32::from_bits(atomic_at(ptr, slot).load(Ordering::Relaxed))
}

/// Per-slot read-modify-write via a compare-and-swap loop.
#[inline]
fn rmw(ptr: *mut f32, slot: usize, f: impl Fn(f32) -> f32) {
    let a = atomic_at(ptr, slot);
    let mut cur = a.load(Ordering::Relaxed);
    loop {
        let new = f(f32::from_bits(cur)).to_bits();
        match a.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(seen) => cur = seen,
        }
    }
}

impl AtomicTable<'_> {
    #[inline]
    fn span(&self, index: usize) -> (usize, usize) {
        (self.offsets[index] as usize, self.num_actions[index] as usize)
    }

    /// Regret-matching current strategy from atomic regret reads, written into
    /// a stack buffer.  Returns the action count.
    #[inline]
    fn strategy(&self, index: usize, out: &mut [f64; MAX_ACTIONS]) -> usize {
        let (base, n) = self.span(index);
        let mut total = 0.0;
        for (a, o) in out.iter_mut().enumerate().take(n) {
            let p = (load(self.regret, base + a) as f64).max(0.0);
            *o = p;
            total += p;
        }
        if total > 0.0 {
            for o in out.iter_mut().take(n) {
                *o /= total;
            }
        } else {
            out[..n].fill(1.0 / n as f64);
        }
        n
    }

    /// The traverser's regret update: per-action lazy DCFR discount + add of
    /// the instantaneous regret, each slot one CAS (mirrors the serial
    /// `SoaMccfr::update_regret` arithmetic: f64 math, f32 store).
    #[inline]
    fn add_regret(&self, index: usize, util: &[f64], node_value: f64, pos: f64, neg: f64, discount: bool) {
        let (base, n) = self.span(index);
        for (a, &u) in util.iter().enumerate().take(n) {
            let inst = u - node_value;
            rmw(self.regret, base + a, |r32| {
                let mut r = r32 as f64;
                if discount {
                    r *= if r > 0.0 { pos } else { neg };
                }
                (r + inst) as f32
            });
        }
    }

    /// The opponent's average-strategy accumulation (`weight · σ(a)` per slot).
    #[inline]
    fn add_strategy(&self, index: usize, weight: f64, strategy: &[f64], n: usize) {
        let (base, _) = self.span(index);
        for (a, &p) in strategy.iter().enumerate().take(n) {
            rmw(self.strategy_sum, base + a, |s| (s as f64 + weight * p) as f32);
        }
    }

    /// One EMA step of a baseline slot toward `target` (player-0 perspective).
    #[inline]
    fn baseline_ema(&self, index: usize, a: usize, target: f64) {
        let (base, _) = self.span(index);
        rmw(self.baseline, base + a, |b| (b as f64 + BASELINE_RATE * (target - b as f64)) as f32);
    }
}

/// Per-iteration constants: the game, the shared view, and iteration `t`'s
/// discount/averaging factors (fixed for the whole traversal).
struct Ctx<'a, G: IndexedGame> {
    game: &'a G,
    table: &'a AtomicTable<'a>,
    use_baseline: bool,
    pos: f64,
    neg: f64,
    discount: bool,
    strategy_weight: f64,
}

fn sign(traverser: usize) -> f64 {
    if traverser == 0 {
        1.0
    } else {
        -1.0
    }
}

/// External-sampling traversal updating the shared table atomically — the
/// lock-free counterpart of the serial `SoaMccfr::traverse`, with stack
/// buffers instead of per-node `Vec`s.
fn traverse<G: IndexedGame>(
    ctx: &Ctx<'_, G>,
    cursor: &mut G::Cursor,
    traverser: usize,
    rng: &mut u64,
    visited: &mut u64,
) -> f64 {
    *visited += 1;
    let game = ctx.game;
    if CursorGame::is_terminal(game, cursor) {
        return CursorGame::utility(game, cursor, traverser);
    }
    if CursorGame::is_chance(game, cursor) {
        CursorGame::sample_chance(game, cursor, || xorshift_next_unit(rng));
        let v = traverse(ctx, cursor, traverser, rng, visited);
        CursorGame::undo_chance(game, cursor);
        return v;
    }

    let player = CursorGame::current_player(game, cursor);
    let index = game.info_set_index(cursor);
    let actions = CursorGame::legal(game, cursor);
    let acts = actions.as_ref();
    let num_actions = acts.len();
    let mut strategy = [0.0f64; MAX_ACTIONS];
    let n = ctx.table.strategy(index, &mut strategy);
    debug_assert_eq!(n, num_actions, "table layout disagrees with legal actions");

    if player == traverser {
        let mut util = [0.0f64; MAX_ACTIONS];
        let mut node_value = 0.0;
        for a in 0..num_actions {
            CursorGame::apply(game, cursor, a, acts[a]);
            util[a] = traverse(ctx, cursor, traverser, rng, visited);
            CursorGame::undo(game, cursor);
            node_value += strategy[a] * util[a];
        }
        ctx.table.add_regret(index, &util[..num_actions], node_value, ctx.pos, ctx.neg, ctx.discount);
        if ctx.use_baseline {
            let sgn = sign(traverser);
            for (a, &u) in util.iter().enumerate().take(num_actions) {
                ctx.table.baseline_ema(index, a, sgn * u);
            }
        }
        node_value
    } else {
        ctx.table.add_strategy(index, ctx.strategy_weight, &strategy, num_actions);
        let a = sample_index(strategy[..num_actions].iter().copied(), xorshift_next_unit(rng));
        CursorGame::apply(game, cursor, a, acts[a]);
        let v_child = traverse(ctx, cursor, traverser, rng, visited);
        CursorGame::undo(game, cursor);
        if !ctx.use_baseline {
            return v_child;
        }
        // VR-MCCFR control variate against the (racy but current) baseline,
        // then one EMA step toward the realized value — same arithmetic as the
        // serial path, per-slot atomic.
        let sgn = sign(traverser);
        let v0 = sgn * v_child;
        let (base, _) = ctx.table.span(index);
        let mut baseline_exp = 0.0;
        for (i, &p) in strategy.iter().enumerate().take(num_actions) {
            baseline_exp += p * load(ctx.table.baseline, base + i) as f64;
        }
        let corrected0 = baseline_exp + (v0 - load(ctx.table.baseline, base + a) as f64);
        ctx.table.baseline_ema(index, a, v0);
        sgn * corrected0
    }
}

/// Scalar parameters of one atomic training run (bundled so the entry point
/// stays readable).
pub(super) struct AtomicRun {
    pub variant: Variant,
    pub use_baseline: bool,
    pub seed: u64,
    /// Iterations already completed; numbering continues at `base_iter + 1` so
    /// discounting/averaging weights resume correctly.
    pub base_iter: u64,
    pub iters: u64,
    pub threads: usize,
}

/// Run `run.iters` lock-free iterations over `run.threads` OS threads,
/// mutating `table` in place.  Returns the total node visits.  Each
/// iteration's RNG stream is `splitmix(seed, t)` — the same scheme as the
/// deterministic parallel path.
pub(super) fn run_atomic<G: IndexedGame + Sync>(
    game: &G,
    table: &mut RegretTable,
    run: AtomicRun,
) -> u64 {
    let AtomicRun { variant, use_baseline, seed, base_iter, iters, threads } = run;
    let threads = threads.max(1);
    let (regret, strategy_sum, baseline, offsets, num_actions) = table.atomic_parts();
    assert!(
        num_actions.iter().all(|&n| n as usize <= MAX_ACTIONS),
        "table has an info set wider than {MAX_ACTIONS} actions"
    );
    let table = AtomicTable { regret, strategy_sum, baseline, offsets, num_actions };
    let counter = AtomicU64::new(0);
    let visited_total = AtomicU64::new(0);
    let players = CursorGame::num_players(game);

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut cursor = CursorGame::root(game);
                let mut visited = 0u64;
                loop {
                    let i = counter.fetch_add(1, Ordering::Relaxed);
                    if i >= iters {
                        break;
                    }
                    let t = base_iter + i + 1;
                    let (pos, neg, strategy_weight) = match variant {
                        Variant::Vanilla => (1.0, 1.0, 1.0),
                        Variant::Dcfr(d) => {
                            (d.positive_factor(t), d.negative_factor(t), d.strategy_weight(t))
                        }
                    };
                    let ctx = Ctx {
                        game,
                        table: &table,
                        use_baseline,
                        pos,
                        neg,
                        discount: matches!(variant, Variant::Dcfr(_)),
                        strategy_weight,
                    };
                    let mut rng = splitmix(seed, t);
                    for traverser in 0..players {
                        traverse(&ctx, &mut cursor, traverser, &mut rng, &mut visited);
                    }
                }
                visited_total.fetch_add(visited, Ordering::Relaxed);
            });
        }
    });

    visited_total.into_inner()
}
