//! Which discount schedule should *sampled* training use — and does the
//! VR-MCCFR baseline earn its memory?
//!
//! Both questions are about the blueprint trainer, which runs external-sampling
//! MCCFR with DCFR `(1.5, 0, 2)` and the baseline.  That schedule is the DCFR
//! paper's recommendation for **full-traversal** CFR; for Monte-Carlo CFR the
//! paper evaluates only Linear CFR.  And this solver applies the discount
//! *lazily* — when an info set is visited, not every iteration — which the
//! paper does not cover at all.  So neither default had been measured here.
//!
//! The measurement: exact exploitability of the *average* strategy (the
//! deployed object) at equal iterations, several seeds, on Leduc and on real
//! hold'em betting over nine curated deals at increasing stack depth.
//!
//! ## What it found (2026-11, means over seeds, bb or chips)
//!
//! **Schedules** — 6 seeds, iterations 25k / 100k / 400k:
//!
//! ```text
//!                      Leduc                  NLHE 4 bb              NLHE 12 bb
//!   vanilla          .091  .047  .022       .073  .032  .015       .212  .063  .0127
//!   DCFR 1.5/0/2     .101  .056  .030       .122  .062  .029       .248  .066  .0093
//!   DCFR 1.5/0.5/2   .092  .047  .023       .084  .039  .019       .419  .109  .0025
//!   DCFR 1.5/1/2     .294  .208  .140       .167  .133  .089       .632  .429  .223
//!   linear 1/1/1     .179  .140  .072       .119  .063  .070       .425  .217  .107
//! ```
//!
//! and the `deep` mode — 3 seeds, 400k / 1.6M:
//!
//! ```text
//!                      NLHE 12 bb           NLHE 20 bb
//!   vanilla          .0106  .0043         .0301  .0051
//!   DCFR 1.5/0/2     .0091  .0037         .0112  .0045
//!   DCFR 1.5/0.5/2   .0025  .0014         .0129  .00034
//! ```
//!
//! * No schedule wins everywhere.  The production `1.5/0/2` is never the best
//!   and never far off it at small budgets; plain vanilla is at least as good
//!   until the budget is large.
//! * `β = 0.5` is slower to start and then overtakes hard — 2.6× lower
//!   exploitability than production at 12 bb and 13× at 20 bb by 1.6M
//!   iterations.  The crossover moves out as the game grows — by a rough
//!   count it sits around ten iterations per info set in both — and the
//!   200 bb blueprint is trained to about that many, so which side it lands
//!   on is not something these games can say.  Worth one A/B by `play expl` if a run ever gets
//!   several times longer; not grounds to change the default.
//! * `(1.5, 1, 2)` and Linear, as implemented with the lazy discount, are
//!   3–25× worse.  With `β = 1` the lazy factor is ~1 per visit, so regrets
//!   are effectively undiscounted while the average still leans on late
//!   iterates — the mismatch costs more than either choice alone.
//!
//! **Baseline** (`baseline` vs `baseline no-baseline`, production schedule,
//! 6 seeds, 25k / 100k / 400k):
//!
//! ```text
//!                      Leduc                  NLHE 4 bb              NLHE 12 bb
//!   with             .101  .056  .030       .122  .062  .029       .248  .066  .0093
//!   without          .099  .053  .032       .116  .060  .030       .249  .075  .0096
//! ```
//!
//! Between 6% worse and 11% better, about 1% better on average.  The baseline
//! is a third of the f32 table, and the atomic trainer runs ~20% more nodes
//! per second without it (`bench_train_paths big` vs `big no-baseline`), so at equal memory and time
//! it does not pay on these games — hence `train blueprint --no-baseline`.
//! The caveat is the same as above: these are ≤ 20 bb games, and payoff
//! variance grows with depth.
//!
//! An experiment, not a gate: it answers "which", which has no pass/fail.
//!
//! The full modes take tens of minutes each (exact best responses over the
//! deeper trees dominate); `quick` is a smoke run.
//!
//!   cargo run --release --example bench_discount                        # schedules
//!   cargo run --release --example bench_discount -- quick
//!   cargo run --release --example bench_discount -- deep                # larger budgets, 12 and 20 bb
//!   cargo run --release --example bench_discount -- baseline            # production schedule, with
//!   cargo run --release --example bench_discount -- baseline no-baseline

use poker_ai::games::Game;
use poker_ai::solver::dcfr::Discount;
use poker_ai::solver::mccfr::Mccfr;
use poker_ai::solver::variant::Variant;
use poker_ai::validation::games::leduc::Leduc;
use poker_ai::validation::games::nlhe::{Deal, HeadsUpHoldem};
use poker_ai::validation::solver::best_response::exploitability;
use poker_core::make_card;

/// Real hold'em betting over nine equally likely deals: each player holds one
/// of three hands (strong / medium / weak on this board), so both face a
/// genuine range.  `stack_bb` sets the depth — and with it how large a single
/// sampled payoff is next to the margins the equilibrium is decided by.
fn curated(stack_bb: u32) -> HeadsUpHoldem {
    // Board: A♣ K♦ 7♥ 2♠ 9♣.
    let board = [make_card(12, 0), make_card(11, 1), make_card(5, 2), make_card(0, 3), make_card(7, 0)];
    let pair = |rank, s0, s1| [make_card(rank, s0), make_card(rank, s1)];
    // Seat 0: QQ, 88, or A♦K♣ (top two pair).  Seat 1: JJ, 33, or 77 (a set).
    let p0 = [pair(10, 0, 1), pair(6, 2, 3), [make_card(12, 1), make_card(11, 0)]];
    let p1 = [pair(9, 0, 1), pair(1, 2, 3), pair(5, 0, 1)];
    let mut deals = Vec::new();
    for &h0 in &p0 {
        for &h1 in &p1 {
            deals.push(Deal { holes: [h0, h1], board });
        }
    }
    HeadsUpHoldem::new(deals, stack_bb * 2, 2, 1, 0)
}

/// The schedules compared.  `(1.5, 0.5, 2)` is the paper's own alternative to
/// the recommended one; the last row keeps DCFR's positive-regret and averaging
/// weights and changes only how negative regret is treated.
fn schedules() -> Vec<(&'static str, Variant)> {
    vec![
        ("vanilla", Variant::Vanilla),
        ("DCFR 1.5/0/2", Variant::Dcfr(Discount::RECOMMENDED)),
        ("DCFR 1.5/0.5/2", Variant::Dcfr(Discount { alpha: 1.5, beta: 0.5, gamma: 2.0 })),
        ("DCFR 1.5/1/2", Variant::Dcfr(Discount { alpha: 1.5, beta: 1.0, gamma: 2.0 })),
        ("linear 1/1/1", Variant::Dcfr(Discount::LINEAR)),
    ]
}

/// Train one schedule from each seed, reading exploitability at every budget.
/// Returns `[budget][seed]`.
fn run<G: Game>(mk: &dyn Fn() -> G, variant: Variant, budgets: &[u64], seeds: u64) -> Vec<Vec<f64>> {
    let baseline = !std::env::args().any(|a| a == "no-baseline");
    let mut out = vec![Vec::new(); budgets.len()];
    for seed in 1..=seeds {
        let mut solver = Mccfr::with_seed(mk(), variant, seed);
        if baseline {
            solver = solver.with_baseline();
        }
        let game = mk();
        let mut done = 0;
        for (slot, &budget) in out.iter_mut().zip(budgets) {
            solver.train(budget - done);
            done = budget;
            slot.push(exploitability(&game, &solver.average_strategy()));
        }
    }
    out
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn compare<G: Game>(
    title: &str,
    unit: &str,
    mk: &dyn Fn() -> G,
    budgets: &[u64],
    seeds: u64,
    schedules: &[(&'static str, Variant)],
) {
    println!("== {title} — exploitability of the average strategy, {unit}, mean of {seeds} seeds (worst seed) ==");
    print!("{:>16}", "");
    for b in budgets {
        print!("  {:>20}", format!("{b} iters"));
    }
    println!();
    for &(name, variant) in schedules {
        print!("{name:>16}");
        for per_seed in run(mk, variant, budgets, seeds) {
            let worst = per_seed.iter().cloned().fold(f64::MIN, f64::max);
            print!("  {:>20}", format!("{:.5} ({:.5})", mean(&per_seed), worst));
        }
        println!();
    }
    println!();
}

fn main() {
    let arg = |name: &str| std::env::args().any(|a| a == name);
    if arg("baseline") {
        // What the VR-MCCFR baseline buys at equal iterations: run this mode
        // with and without `no-baseline` and compare.
        let production: Vec<_> = schedules().into_iter().filter(|(n, _)| *n == "DCFR 1.5/0/2").collect();
        let budgets = [25_000, 100_000, 400_000];
        compare("Leduc", "chips", &|| Leduc, &budgets, 6, &production);
        compare("curated NLHE, 4 bb", "bb", &|| curated(4), &budgets, 6, &production);
        compare("curated NLHE, 12 bb", "bb", &|| curated(12), &budgets, 6, &production);
        return;
    }
    if arg("deep") {
        // The follow-up: does the ranking at large budgets hold as the game
        // deepens and the budget grows?  Slow (exact best responses over the
        // 20 bb tree), so fewer seeds and only the three contenders.
        let three: Vec<_> = schedules().into_iter().filter(|(n, _)| !n.contains("/1/")).collect();
        compare("curated NLHE, 12 bb", "bb", &|| curated(12), &[400_000, 1_600_000], 3, &three);
        compare("curated NLHE, 20 bb", "bb", &|| curated(20), &[400_000, 1_600_000], 3, &three);
        return;
    }
    let (budgets, seeds): (&[u64], u64) =
        if arg("quick") { (&[20_000, 80_000], 3) } else { (&[25_000, 100_000, 400_000], 6) };
    let all = schedules();
    compare("Leduc", "chips", &|| Leduc, budgets, seeds, &all);
    compare("curated NLHE, 4 bb", "bb", &|| curated(4), budgets, seeds, &all);
    compare("curated NLHE, 12 bb", "bb", &|| curated(12), budgets, seeds, &all);
}
