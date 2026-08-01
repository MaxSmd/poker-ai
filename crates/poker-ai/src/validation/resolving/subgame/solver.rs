//! Driving the subgame: which regret minimizer runs, for how long, and what
//! comes back.
//!
//! Predictive CFR⁺ is the default (its last iterate is the best strategy per
//! unit of time budget — the measured 0.0055 vs 0.0294 bb at 2k iterations);
//! DCFR's average is the designated fallback for regimes where CFR⁺'s
//! guarantees erode.  Both consume the identical tree built in [`super`], so
//! the switch is one line.

use std::collections::HashMap;
use std::time::Instant;

use poker_core::state::GameState;

use super::Subgame;
use crate::resolving::belief_state::BeliefState;
use crate::validation::resolving::leaf_eval::LeafEvaluator;
use crate::solver::variant::Variant;
use crate::validation::solver::full_cfr::Cfr;
use crate::validation::solver::predictive::PredictiveSolver;

/// Resolved subgame output.
pub struct Resolved {
    /// Strategy per information set (the resolver's deployable last iterate).
    pub strategy: HashMap<u64, Vec<f64>>,
    /// Number of enumerated deals (chance breadth).
    pub deals: usize,
    /// Number of information sets discovered.
    pub info_sets: usize,
}

/// Which regret minimizer resolves the subgame.
///
/// The default is **predictive** (CFR⁺): in the near-two-player,
/// full-traversal regime a subgame becomes once folds collapse the active set,
/// CFR⁺'s fast last iterate buys the best strategy per second.  There is also
/// a **DCFR fallback**: predictive RM⁺'s strong guarantees are a 2p0s
/// result, so a *multiway* subgame (several opponents still in) should fall back
/// to DCFR, which is empirically robust where the predictive guarantee erodes.
/// Both consume the identical subgame tree, so the fallback is a one-line switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    /// CFR⁺ last iterate — the default for heads-up / near-heads-up subgames.
    Predictive,
    /// Discounted CFR average — the robust fallback for multiway subgames.
    Dcfr,
}

/// Depth-limited subgame solver: builds the subgame and solves it with the chosen
/// regret minimizer (predictive CFR⁺ by default, DCFR for the multiway fallback).
pub struct SubgameSolver {
    /// Streets to solve before cutting to leaf estimates (1–2 is realistic).
    pub depth_limit: u32,
    /// Wall-clock budget per resolving call.
    pub time_budget_ms: u64,
    /// Regret minimizer used to resolve.
    pub kind: SolverKind,
    /// Optional blueprint warm-start (predictive only): seed regrets so the first
    /// iterate is the blueprint instead of uniform.  See [`crate::validation::resolving::warm_start`].
    warm_start: Option<HashMap<u64, Vec<f64>>>,
}

impl SubgameSolver {
    /// A predictive (CFR⁺) subgame solver — the default.
    pub fn new(depth_limit: u32, time_budget_ms: u64) -> Self {
        Self { depth_limit, time_budget_ms, kind: SolverKind::Predictive, warm_start: None }
    }

    /// Select the regret minimizer (e.g. [`SolverKind::Dcfr`] for a multiway
    /// fallback).
    pub fn with_solver(mut self, kind: SolverKind) -> Self {
        self.kind = kind;
        self
    }

    /// Warm-start the (predictive) solver's regrets from a blueprint, expressed
    /// over the subgame's own information sets.  Ignored on the DCFR path.
    pub fn with_warm_start(mut self, seed_regrets: HashMap<u64, Vec<f64>>) -> Self {
        self.warm_start = Some(seed_regrets);
        self
    }

    /// Resolve the subgame rooted at `root` over the given `beliefs`, training
    /// until the wall-clock budget is spent.  Returns the deployable strategy
    /// (CFR⁺ last iterate, or DCFR average on the fallback path).
    pub fn solve(
        &self,
        root: &GameState,
        beliefs: &[BeliefState],
        leaf_eval: &dyn LeafEvaluator,
    ) -> Resolved {
        let subgame = Subgame::new(root.clone(), beliefs, leaf_eval);
        let deals = subgame.num_deals();
        match self.kind {
            SolverKind::Predictive => {
                let mut solver = PredictiveSolver::new(subgame);
                if let Some(seed) = &self.warm_start {
                    solver.warm_start(seed.clone());
                }
                let start = Instant::now();
                loop {
                    solver.train(32);
                    if start.elapsed().as_millis() >= self.time_budget_ms as u128 {
                        break;
                    }
                }
                Resolved { strategy: solver.current_strategy(), deals, info_sets: solver.num_info_sets() }
            }
            SolverKind::Dcfr => {
                let mut solver = Cfr::new(subgame, Variant::Dcfr(crate::solver::dcfr::Discount::RECOMMENDED));
                let start = Instant::now();
                loop {
                    solver.train(32);
                    if start.elapsed().as_millis() >= self.time_budget_ms as u128 {
                        break;
                    }
                }
                Resolved { strategy: solver.average_strategy(), deals, info_sets: solver.num_info_sets() }
            }
        }
    }

    /// Deterministic resolve for a fixed iteration count (used by tests and when
    /// reproducibility matters more than a wall-clock budget).
    pub fn solve_for_iters(
        &self,
        root: &GameState,
        beliefs: &[BeliefState],
        leaf_eval: &dyn LeafEvaluator,
        iters: u64,
    ) -> Resolved {
        let subgame = Subgame::new(root.clone(), beliefs, leaf_eval);
        let deals = subgame.num_deals();
        match self.kind {
            SolverKind::Predictive => {
                let mut solver = PredictiveSolver::new(subgame);
                if let Some(seed) = &self.warm_start {
                    solver.warm_start(seed.clone());
                }
                solver.train(iters);
                Resolved { strategy: solver.current_strategy(), deals, info_sets: solver.num_info_sets() }
            }
            SolverKind::Dcfr => {
                let mut solver = Cfr::new(subgame, Variant::Dcfr(crate::solver::dcfr::Discount::RECOMMENDED));
                solver.train(iters);
                Resolved { strategy: solver.average_strategy(), deals, info_sets: solver.num_info_sets() }
            }
        }
    }
}
