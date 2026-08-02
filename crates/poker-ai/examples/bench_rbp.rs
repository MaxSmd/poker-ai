//! Regret-Based Pruning sensitivity: what `(θ, K)` costs and what it saves.
//!
//! RBP skips actions whose regret has stayed below `θ` for `K` consecutive
//! visits.  The saving is node visits; the risk is convergence, and the risk is
//! **not symmetric** — a θ that is too shallow prunes actions that belong to a
//! *mixed* equilibrium support, and the strategy converges to the wrong thing
//! while still looking like it converged.  That is why this sweep exists: θ has
//! to be deep relative to the game's regret scale, and "deep" is a property of
//! the game, not a universal constant.
//!
//! Leduc is the right test bed because its equilibrium is genuinely mixed, so a
//! bad θ shows up as exploitability rather than as a crash.  Read the two
//! columns together: node savings only count if exploitability held.
//!
//! A sweep is an experiment, not a gate — it answers "which θ should I use",
//! which has no pass/fail.  It ran as an `#[ignore]`d test until that stopped
//! being a useful thing to run on a schedule.
//!
//!   cargo run --release --example bench_rbp

use poker_ai::solver::dcfr::Discount;
use poker_ai::solver::mccfr::Mccfr;
use poker_ai::solver::pruning::PruningConfig;
use poker_ai::solver::variant::Variant;
use poker_ai::validation::games::leduc::Leduc;
use poker_ai::validation::solver::best_response::exploitability;

const TOTAL: u64 = 400_000;

/// θ/K pairs spanning shallow to deep.  The shallow end is included precisely
/// because it is the failure mode worth seeing.
const SWEEP: [(f64, u32); 3] = [(-50.0, 50), (-100.0, 100), (-300.0, 200)];

/// Sweep `(θ, K)` under one regret regime, reporting whether pruning fired at
/// all.  Returns true if any configuration actually skipped work.
fn sweep(label: &str, variant: Variant) -> bool {
    println!("== {label} ==");
    let mut plain = Mccfr::with_seed(Leduc, variant, 3);
    plain.train(TOTAL);
    let plain_expl = exploitability(&Leduc, &plain.average_strategy());
    let plain_nodes = plain.nodes_visited();
    println!("{:>16}  expl={plain_expl:.5}  nodes={plain_nodes}", "no pruning");

    let mut any_pruned = false;
    for &(theta, k) in &SWEEP {
        let cfg = PruningConfig { theta, k, start_fraction: 0.2, refresh_interval: 10_000 };
        let mut s = Mccfr::with_seed(Leduc, variant, 3).with_pruning(cfg, TOTAL);
        s.train(TOTAL);
        let expl = exploitability(&Leduc, &s.average_strategy());
        let nodes = s.nodes_visited();
        let pct = 100.0 * nodes as f64 / plain_nodes as f64;
        let verdict = if nodes == plain_nodes {
            "INERT — pruning never fired"
        } else if expl < 0.05 {
            "converged"
        } else {
            "DIVERGED — θ too shallow"
        };
        any_pruned |= nodes != plain_nodes;
        println!(
            "{:>16}  expl={expl:.5}  nodes={nodes} ({pct:.1}% of plain)  {verdict}",
            format!("θ={theta} K={k}")
        );
    }
    println!();
    any_pruned
}

fn main() {
    println!("RBP sensitivity on Leduc, external sampling, {TOTAL} iterations per config.\n");

    // Both regimes, because whether pruning can fire at all is a property of the
    // regret regime, not of (θ, K) — see the closing note.
    let dcfr = sweep("DCFR (α,β,γ)=(1.5,0,2) — the production default", Variant::Dcfr(Discount::RECOMMENDED));
    let vanilla = sweep("Vanilla CFR — undiscounted regret", Variant::Vanilla);

    println!("Read both columns: fewer nodes is only a win if exploitability held under 0.05.");
    println!("Leduc's equilibrium is mixed, so a shallow θ prunes real support and converges wrong.");
    if !dcfr {
        println!(
            "\nFINDING: RBP is inert under DCFR here. β=0 makes `negative_factor` a constant\n\
             0.5, so accumulated negative regret is halved every iteration and stays bounded\n\
             near zero — it never reaches any θ deep enough to be safe. RBP and this discount\n\
             schedule do not compose; pruning needs {}.",
            if vanilla { "an undiscounted (or β=1) regime to have any effect" } else { "investigation — it did not fire under vanilla either" }
        );
    }
}
