    use super::*;
    use crate::games::blueprint::BlueprintHoldem;
    use crate::games::kuhn::{Kuhn, GAME_VALUE_P0};
    use crate::games::leduc::Leduc;
    use crate::games::push_fold::PushFoldHoldem;
    use crate::solver::best_response::{exploitability, profile_value};
    use crate::solver::dcfr::Discount;

    #[test]
    fn external_sampling_converges_on_kuhn() {
        let mut solver = Mccfr::new(Kuhn, Variant::Vanilla);
        solver.train(100_000);
        let avg = solver.average_strategy();
        let expl = exploitability(&Kuhn, &avg);
        assert!(expl < 0.01, "MCCFR exploitability {expl} should be < 0.01 after 100k iters");

        let value = profile_value(&Kuhn, &avg, 0);
        assert!(
            (value - GAME_VALUE_P0).abs() < 0.02,
            "MCCFR value {value} should be near -1/18 = {GAME_VALUE_P0}"
        );
    }

    #[test]
    fn dcfr_variant_also_converges_on_kuhn() {
        // DCFR doesn't beat vanilla on a game this small, but it must still
        // converge — the discount schedule should not break the solver.
        let mut solver = Mccfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED));
        solver.train(100_000);
        let expl = exploitability(&Kuhn, &solver.average_strategy());
        assert!(expl < 0.02, "DCFR MCCFR exploitability {expl} should be < 0.02");
    }

    #[test]
    fn discovers_all_kuhn_info_sets() {
        let mut solver = Mccfr::new(Kuhn, Variant::Vanilla);
        solver.train(5_000);
        assert_eq!(solver.num_info_sets(), 12, "should sample all 12 Kuhn info sets");
    }

    #[test]
    fn is_deterministic_for_fixed_seed() {
        let mut a = Mccfr::with_seed(Kuhn, Variant::Vanilla, 42);
        let mut b = Mccfr::with_seed(Kuhn, Variant::Vanilla, 42);
        a.train(2_000);
        b.train(2_000);
        let (sa, sb) = (a.average_strategy(), b.average_strategy());
        assert_eq!(sa.len(), sb.len());
        for (key, va) in &sa {
            let vb = &sb[key];
            for (x, y) in va.iter().zip(vb) {
                assert!((x - y).abs() < 1e-12, "same seed must give identical strategies");
            }
        }
    }

    #[test]
    fn baseline_version_still_converges() {
        // The VR-MCCFR baseline is a control variate: it must not bias the
        // result, so the baseline solver must still reach equilibrium.
        let mut solver = Mccfr::new(Kuhn, Variant::Vanilla).with_baseline();
        solver.train(100_000);
        let expl = exploitability(&Kuhn, &solver.average_strategy());
        assert!(expl < 0.01, "baseline MCCFR exploitability {expl} should be < 0.01");
    }

    #[test]
    fn baseline_is_deterministic_for_fixed_seed() {
        let mut a = Mccfr::with_seed(Kuhn, Variant::Vanilla, 7).with_baseline();
        let mut b = Mccfr::with_seed(Kuhn, Variant::Vanilla, 7).with_baseline();
        a.train(2_000);
        b.train(2_000);
        for (key, va) in &a.average_strategy() {
            for (x, y) in va.iter().zip(&b.average_strategy()[key]) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn baseline_reduces_exploitability_on_kuhn() {
        // Averaged over a fixed set of seeds (so this is deterministic, not
        // flaky), the baseline's control-variate variance reduction lowers the
        // mean exploitability at a fixed iteration budget.
        let seeds = 1..=12u64;
        let iters = 20_000;
        let mean = |with_baseline: bool| -> f64 {
            let xs: Vec<f64> = seeds
                .clone()
                .map(|s| {
                    let mut m = Mccfr::with_seed(Kuhn, Variant::Vanilla, s);
                    if with_baseline {
                        m = m.with_baseline();
                    }
                    m.train(iters);
                    exploitability(&Kuhn, &m.average_strategy())
                })
                .collect();
            xs.iter().sum::<f64>() / xs.len() as f64
        };
        let with = mean(true);
        let without = mean(false);
        assert!(
            with < without,
            "baseline mean exploitability ({with}) should beat no-baseline ({without})"
        );
    }

    #[test]
    fn optimistic_still_converges_on_kuhn() {
        // Optimistic (predictive) updates must not bias the solver: the deployed
        // average strategy still reaches equilibrium.
        let mut s = Mccfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED)).with_optimistic();
        s.train(100_000);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "optimistic MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn optimistic_is_deterministic_for_fixed_seed() {
        let run = || {
            let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 9).with_optimistic();
            s.train(2_000);
            s.average_strategy()
        };
        let (a, b) = (run(), run());
        for (key, va) in &a {
            for (x, y) in va.iter().zip(&b[key]) {
                assert!((x - y).abs() < 1e-12, "optimistic updates must be deterministic");
            }
        }
    }

    #[test]
    fn pruning_preserves_convergence_on_kuhn() {
        // RBP freezes provably-bad branches; the deployed strategy must still
        // converge (a refresh traversal re-checks frozen branches).
        let total = 100_000;
        let cfg = PruningConfig { theta: -2000.0, k: 80, start_fraction: 0.2, refresh_interval: 5_000 };
        let mut s = Mccfr::new(Kuhn, Variant::Vanilla).with_pruning(cfg, total);
        s.train(total);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "pruned MCCFR exploitability {expl} should still converge");
    }

    #[test]
    fn pruning_visits_fewer_nodes_than_plain() {
        // The point of RBP: fewer node visits at an equal iteration budget.  Same
        // seed so the only difference is pruning.
        let total = 100_000;
        let cfg = PruningConfig { theta: -2000.0, k: 80, start_fraction: 0.2, refresh_interval: 5_000 };
        let mut pruned = Mccfr::with_seed(Kuhn, Variant::Vanilla, 1).with_pruning(cfg, total);
        let mut plain = Mccfr::with_seed(Kuhn, Variant::Vanilla, 1);
        pruned.train(total);
        plain.train(total);
        assert!(
            pruned.nodes_visited() < plain.nodes_visited(),
            "RBP should visit fewer nodes: pruned={} plain={}",
            pruned.nodes_visited(),
            plain.nodes_visited()
        );
    }

    #[test]
    fn parallel_converges_on_kuhn() {
        // Mini-batch (parallel) MCCFR is the plain external-sampling estimator,
        // so the deployed average must still reach equilibrium.
        let mut s = Mccfr::new(Kuhn, Variant::Vanilla);
        s.train_parallel(100_000, 64);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "parallel MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn parallel_dcfr_converges_on_kuhn() {
        let mut s = Mccfr::new(Kuhn, Variant::Dcfr(Discount::RECOMMENDED));
        s.train_parallel(100_000, 64);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "parallel DCFR MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn parallel_is_deterministic_for_fixed_seed_and_batch() {
        // Workers merge in iteration order, so a fixed seed + batch gives a
        // bit-identical result no matter how the threads were scheduled.
        let run = || {
            let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 5);
            s.train_parallel(4_000, 32);
            s.average_strategy()
        };
        let (a, b) = (run(), run());
        assert_eq!(a.len(), b.len());
        for (key, va) in &a {
            for (x, y) in va.iter().zip(&b[key]) {
                assert!((x - y).abs() < 1e-12, "parallel training must be deterministic");
            }
        }
    }

    #[test]
    fn parallel_baseline_is_deterministic() {
        // The parallel baseline reads a read-only snapshot and merges target
        // deltas in iteration order, so a fixed seed + batch is reproducible.
        let run = || {
            let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 5).with_baseline();
            s.train_parallel(4_000, 32);
            s.average_strategy()
        };
        let (a, b) = (run(), run());
        assert_eq!(a.len(), b.len());
        for (key, va) in &a {
            for (x, y) in va.iter().zip(&b[key]) {
                assert!((x - y).abs() < 1e-12, "parallel baseline must be deterministic");
            }
        }
    }

    #[test]
    fn parallel_baseline_converges_on_kuhn() {
        // The control variate must not bias the parallel estimator.
        let mut s = Mccfr::new(Kuhn, Variant::Vanilla).with_baseline();
        s.train_parallel(100_000, 32);
        let expl = exploitability(&Kuhn, &s.average_strategy());
        assert!(expl < 0.02, "parallel baseline MCCFR exploitability {expl} should converge");
    }

    #[test]
    fn parallel_baseline_reduces_variance_on_kuhn() {
        // Over a fixed seed set, the parallel baseline's control variate lowers
        // the mean exploitability at a fixed budget — the cloud-burst variance
        // lever, now available on the parallel path (mirrors the serial test).
        let seeds = 1..=12u64;
        let (iters, batch) = (20_000, 32);
        let mean = |with_baseline: bool| -> f64 {
            let xs: Vec<f64> = seeds
                .clone()
                .map(|s| {
                    let mut m = Mccfr::with_seed(Kuhn, Variant::Vanilla, s);
                    if with_baseline {
                        m = m.with_baseline();
                    }
                    m.train_parallel(iters, batch);
                    exploitability(&Kuhn, &m.average_strategy())
                })
                .collect();
            xs.iter().sum::<f64>() / xs.len() as f64
        };
        let (with, without) = (mean(true), mean(false));
        assert!(with < without, "parallel baseline mean ({with}) should beat no-baseline ({without})");
    }

    fn strategies_equal(a: &HashMap<u64, Vec<f64>>, b: &HashMap<u64, Vec<f64>>) {
        assert_eq!(a.len(), b.len(), "same info sets");
        for (key, va) in a {
            let vb = &b[key];
            for (x, y) in va.iter().zip(vb) {
                assert!((x - y).abs() < 1e-12, "checkpoint resume must be bit-identical");
            }
        }
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mccfr_ckpt_{tag}_{}.bin", std::process::id()))
    }

    #[test]
    fn resume_from_checkpoint_is_bit_identical() {
        // A run interrupted at the half-way point and resumed from a checkpoint
        // must produce exactly the same strategy as one that never stopped — the
        // proof the full resumable state (regrets, sums, baseline, RNG, iteration
        // counter) round-trips correctly.
        let mut whole = Mccfr::with_seed(Kuhn, Variant::Dcfr(Discount::RECOMMENDED), 11).with_baseline();
        whole.train(100_000);

        let mut part = Mccfr::with_seed(Kuhn, Variant::Dcfr(Discount::RECOMMENDED), 11).with_baseline();
        part.train(50_000);
        let path = temp_path("resume");
        part.save_checkpoint(&path).unwrap();
        drop(part);

        let mut resumed = Mccfr::load_checkpoint(&path, Kuhn).unwrap();
        assert_eq!(resumed.iterations(), 50_000, "iteration counter restored");
        resumed.train(50_000);
        std::fs::remove_file(&path).ok();

        strategies_equal(&whole.average_strategy(), &resumed.average_strategy());
    }

    #[test]
    fn checkpoint_restores_config_and_counters() {
        // The configuration (variant, baseline/optimistic/pruning) and counters
        // must survive the round-trip, not just the regret table.
        let cfg = PruningConfig { theta: -2000.0, k: 80, start_fraction: 0.2, refresh_interval: 5_000 };
        let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 3)
            .with_optimistic()
            .with_pruning(cfg, 100_000);
        s.train(10_000);
        let (it, nv) = (s.iterations(), s.nodes_visited());
        let path = temp_path("config");
        s.save_checkpoint(&path).unwrap();

        let resumed = Mccfr::load_checkpoint(&path, Kuhn).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(resumed.iterations(), it);
        assert_eq!(resumed.nodes_visited(), nv);
        // Continuing must stay deterministic against an uninterrupted twin.
        let mut twin = Mccfr::with_seed(Kuhn, Variant::Vanilla, 3)
            .with_optimistic()
            .with_pruning(cfg, 100_000);
        twin.train(10_000);
        let mut a = resumed;
        a.train(5_000);
        twin.train(5_000);
        strategies_equal(&a.average_strategy(), &twin.average_strategy());
    }

    #[test]
    fn save_is_atomic_no_leftover_temp() {
        // The atomic write renames a temp file into place; afterwards only the
        // checkpoint exists, never a stray `.ckpt.tmp`.
        let mut s = Mccfr::with_seed(Kuhn, Variant::Vanilla, 1);
        s.train(100);
        let path = temp_path("atomic");
        s.save_checkpoint(&path).unwrap();
        let tmp = path.with_extension("ckpt.tmp");
        assert!(path.exists(), "checkpoint written");
        assert!(!tmp.exists(), "no leftover temp file after atomic rename");
        std::fs::remove_file(&path).ok();
    }

    // ── Cursor fast path: bit-identical to the clone-based path ──────────────
    //
    // The whole point of the cursor path is that it changes *nothing* observable
    // — same RNG consumption, same info keys, same updates — only the allocation
    // behavior.  These tests pin that: a fixed seed must yield identical info-set
    // counts, identical strategies, and identical node-visit counts across the
    // two paths.  (Mirrors the bit-identical checkpoint-resume tests above.)

    #[test]
    fn train_fast_matches_clone_on_push_fold() {
        let make = || PushFoldHoldem::new(40, 2, 1, 0);
        let mut clone = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 7).with_baseline();
        clone.train(3_000);
        let mut fast = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 7).with_baseline();
        fast.train_fast(3_000);
        assert_eq!(clone.num_info_sets(), fast.num_info_sets());
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    #[test]
    fn train_fast_matches_clone_with_optimistic_and_rbp() {
        // Exercises the cursor traverser's prune/optimistic branches.
        let total = 5_000;
        let cfg = PruningConfig { theta: -5_000.0, k: 50, start_fraction: 0.2, refresh_interval: 1_000 };
        let make = || PushFoldHoldem::new(40, 2, 1, 0);
        let build = || {
            Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 9)
                .with_baseline()
                .with_optimistic()
                .with_pruning(cfg, total)
        };
        let mut clone = build();
        clone.train(total);
        let mut fast = build();
        fast.train_fast(total);
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    #[test]
    fn train_parallel_fast_matches_clone_parallel_on_push_fold() {
        let make = || PushFoldHoldem::new(40, 2, 1, 0);
        let mut clone = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 5);
        clone.train_parallel(4_000, 32);
        let mut fast = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 5);
        fast.train_parallel_fast(4_000, 32);
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    #[test]
    fn train_fast_matches_clone_on_blueprint() {
        // The blueprint mints fresh postflop info sets on every deal, so this
        // also checks the two paths *discover* the same information sets.
        let make = || BlueprintHoldem::new(40, 2, 1, 0);
        let mut clone = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 3);
        clone.train(1_000);
        let mut fast = Mccfr::with_seed(make(), Variant::Dcfr(Discount::RECOMMENDED), 3);
        fast.train_fast(1_000);
        assert_eq!(clone.num_info_sets(), fast.num_info_sets());
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        strategies_equal(&clone.average_strategy(), &fast.average_strategy());
    }

    /// Throughput of the cursor fast path vs the clone-based path on the
    /// blueprint (deep trees ⇒ the clone-per-node undo-stack allocation hurts
    /// most).  Prints nodes/sec for both; the fast path must not be slower.  Run
    /// in release to get a meaningful number:
    ///   cargo test -p poker-ai --release -- --ignored cursor_fast_path_is_faster --nocapture
    #[test]
    #[ignore]
    fn cursor_fast_path_is_faster() {
        use std::time::Instant;
        let iters = 200_000;

        let mut clone = Mccfr::with_seed(BlueprintHoldem::new(40, 2, 1, 0), Variant::Dcfr(Discount::RECOMMENDED), 1);
        let t0 = Instant::now();
        clone.train(iters);
        let clone_s = t0.elapsed().as_secs_f64();

        let mut fast = Mccfr::with_seed(BlueprintHoldem::new(40, 2, 1, 0), Variant::Dcfr(Discount::RECOMMENDED), 1);
        let t0 = Instant::now();
        fast.train_fast(iters);
        let fast_s = t0.elapsed().as_secs_f64();

        // Same work either way (bit-identical), so nodes/sec is a fair ratio.
        assert_eq!(clone.nodes_visited(), fast.nodes_visited());
        let nodes = clone.nodes_visited() as f64;
        println!(
            "blueprint {iters} iters: clone {:.2}s ({:.0} nodes/s) vs cursor {:.2}s ({:.0} nodes/s) — {:.2}x",
            clone_s, nodes / clone_s, fast_s, nodes / fast_s, clone_s / fast_s
        );
        assert!(fast_s <= clone_s * 1.05, "cursor path should not be slower (clone {clone_s:.2}s, fast {fast_s:.2}s)");
    }

    /// MCCFR converges on Leduc too — slower and noisier than full traversal, so
    /// run on demand:  cargo test -p poker-ai --release -- --ignored mccfr
    #[test]
    #[ignore]
    fn external_sampling_converges_on_leduc() {
        let mut solver = Mccfr::new(Leduc, Variant::Vanilla);
        solver.train(300_000);
        let expl = exploitability(&Leduc, &solver.average_strategy());
        assert!(expl < 0.05, "Leduc MCCFR exploitability {expl} should be < 0.05");
    }

    /// RBP θ/K sensitivity on Leduc: across a small sweep, pruning should keep
    /// exploitability low while cutting node visits.  On demand (slow):
    ///   cargo test -p poker-ai --release -- --ignored rbp_sensitivity
    #[test]
    #[ignore]
    fn rbp_sensitivity_on_leduc() {
        let total = 400_000;
        let mut plain = Mccfr::with_seed(Leduc, Variant::Dcfr(Discount::RECOMMENDED), 3);
        plain.train(total);
        let plain_expl = exploitability(&Leduc, &plain.average_strategy());
        let plain_nodes = plain.nodes_visited();
        println!("plain: expl={plain_expl:.5}, nodes={plain_nodes}");

        for &(theta, k) in &[(-50.0, 50u32), (-100.0, 100), (-300.0, 200)] {
            let cfg = PruningConfig { theta, k, start_fraction: 0.2, refresh_interval: 10_000 };
            let mut s = Mccfr::with_seed(Leduc, Variant::Dcfr(Discount::RECOMMENDED), 3)
                .with_pruning(cfg, total);
            s.train(total);
            let expl = exploitability(&Leduc, &s.average_strategy());
            let nodes = s.nodes_visited();
            println!("θ={theta} K={k}: expl={expl:.5}, nodes={nodes} ({:.1}% of plain)",
                100.0 * nodes as f64 / plain_nodes as f64);
            assert!(expl < 0.05, "θ={theta} K={k} stayed converged ({expl})");
            assert!(nodes <= plain_nodes, "pruning should not increase node visits");
        }
    }
