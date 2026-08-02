//! Throughput of every MCCFR training path, on a realistic blueprint tree.
//!
//! These are **measurements, not gates**.  They lived as `#[ignore]`d tests
//! until it became clear that was a category error: "is path A faster than path
//! B" has no correct answer, only a machine-dependent one.  On a thermally
//! limited laptop the assertion fails when the chip throttles — which is not a
//! regression, and worse, it fires while competing for cores with whatever else
//! the test harness is running.  A benchmark belongs here, run deliberately, on
//! an idle machine.
//!
//! Two comparisons:
//!
//!  1. **Cursor vs clone** on the `HashMap` solver.  `Game::apply` clones a
//!     child per node, dragging `poker_core`'s pre-allocated `UndoStack` with it
//!     — a heap allocation at every tree node.  `CursorGame` walks one state in
//!     place with apply/undo.  The two are proven *bit-identical* by the test
//!     suite, so nodes/sec is a fair ratio; this measures what that buys.
//!  2. **Serial vs batched-parallel vs atomic** on the flat SoA solver, over a
//!     fully-indexed tree — the parallel-scaling result.
//!
//! The abstraction is [`BucketMap::full_coverage_mod`], a synthetic
//! total-coverage fixture: strategically meaningless, but it gives the dense
//! index no holes, which is all a throughput measurement needs.  Turn/river
//! coverage maps are ~280 MB, so this wants release and some RAM headroom.
//!
//!   cargo run --release --example bench_train_paths

use std::time::Instant;

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::games::blueprint::BlueprintHoldem;
use poker_ai::solver::dcfr::Discount;
use poker_ai::solver::mccfr::{Mccfr, SoaMccfr};
use poker_ai::solver::variant::Variant;

/// Iterations per configuration.  Large enough that startup cost is noise,
/// small enough that the whole run is a couple of minutes.  Override with a
/// numeric argument: on a thermally limited machine a long run measures the
/// cooling system, so prefer several short runs and read the *fastest* one.
const ITERS: u64 = 200_000;

fn iters() -> u64 {
    std::env::args().skip(1).find_map(|a| a.parse().ok()).unwrap_or(ITERS)
}

/// Cursor fast path vs the clone-based path, on deep trees where the
/// clone-per-node undo-stack allocation hurts most.
fn cursor_vs_clone() {
    let iters = iters();
    println!("== HashMap solver: clone-based vs cursor traversal ({iters} iters) ==");

    let mut clone = Mccfr::with_seed(
        BlueprintHoldem::new(40, 2, 1, 0),
        Variant::Dcfr(Discount::RECOMMENDED),
        1,
    );
    let t0 = Instant::now();
    clone.train(iters);
    let clone_s = t0.elapsed().as_secs_f64();

    let mut fast = Mccfr::with_seed(
        BlueprintHoldem::new(40, 2, 1, 0),
        Variant::Dcfr(Discount::RECOMMENDED),
        1,
    );
    let t0 = Instant::now();
    fast.train_fast(iters);
    let fast_s = t0.elapsed().as_secs_f64();

    // Bit-identical work either way, so the node counts must agree — if they
    // don't, the paths have diverged and the timing comparison is meaningless.
    assert_eq!(
        clone.nodes_visited(),
        fast.nodes_visited(),
        "paths must visit identical nodes for the ratio to mean anything"
    );
    let nodes = clone.nodes_visited() as f64;
    println!("           clone: {clone_s:6.2}s  {:>12.0} nodes/s", nodes / clone_s);
    println!("          cursor: {fast_s:6.2}s  {:>12.0} nodes/s", nodes / fast_s);
    println!("          => {:.2}x\n", clone_s / fast_s);
}

/// Serial vs batched-parallel vs lock-free atomic, on an indexed cap-2 tree.
fn soa_scaling() {
    let iters = iters();
    println!("== SoA solver: parallel scaling on an indexed 20bb cap-2 tree ({iters} iters) ==");
    println!("   (building ~280 MB of full-coverage maps per config — this is the slow part)");

    let mk = || {
        BlueprintHoldem::new(40, 2, 1, 0)
            .with_raise_cap(2)
            .with_street_bucket(0, BucketMap::full_coverage_mod(&[2, 3], 40))
            .with_street_bucket(1, BucketMap::full_coverage_mod(&[2, 4], 40))
            .with_street_bucket(2, BucketMap::full_coverage_mod(&[2, 5], 40))
            .with_indexing()
    };
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
        atomic_best = atomic_best.max(bench(&name, &mut |s| s.train_atomic(iters, threads)));
    }

    // Reported, not asserted: on a throttled or busy machine the ordering can
    // legitimately invert, and that is information about the machine rather
    // than about the code.
    println!(
        "\n  atomic best / serial   = {:.2}x\n  atomic best / parallel = {:.2}x",
        atomic_best / serial,
        atomic_best / parallel
    );
    if atomic_best <= serial {
        println!("  NOTE: atomic did not beat serial — expected only on a busy/throttled machine.");
    }
}

fn main() {
    cursor_vs_clone();
    // The SoA arm builds ~280 MB of coverage maps per configuration, six
    // configurations deep — worth having, but not something to run by accident
    // on a small machine, so it is opt-in like `bench_resolve_cost`'s heavy modes.
    if std::env::args().any(|a| a == "all") {
        soa_scaling();
    } else {
        println!("Cursor arm only; pass `all` for the SoA scaling arm (~280 MB of maps per config).");
    }
}
