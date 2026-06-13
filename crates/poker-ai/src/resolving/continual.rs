//! Continual re-solving driver (finding #4).
//!
//! Instead of re-deriving every decision from raw blueprint beliefs (Pluribus
//! style), continual re-solving **carries the opponent's counterfactual values
//! forward** (DeepStack style): the first decision bootstraps with a plain
//! range-vs-range resolve; every subsequent decision is a gadget-constrained
//! resolve ([`crate::resolving::gadget`]) anchored on the carried CFVs and
//! **warm-started** from the previous strategy.  Each resolve refreshes the
//! carried CFVs for the next.
//!
//! Two payoffs, both demonstrated in `examples/bench_continual.rs`:
//! * **Safety** — the gadget holds the opponent to the value it was previously
//!   guaranteed, so exploitability does not grow across decisions.
//! * **Speed** — warm-starting from the carried strategy reaches a target in
//!   fewer iterations than a cold resolve.
//!
//! Scope note: the carried object is the opponent's CFV indexed by its hand.
//! Re-solving the *same* boundary (the natural cadence each time it is our turn)
//! is exact; carrying across a *street* boundary reuses the previous street's
//! per-hand guarantee, which is exact in expectation over the dealt card (the
//! per-card refinement wants the [`Subgame`] to deal the next street as chance —
//! today it cuts there — flagged as the scale follow-up).

use std::collections::HashMap;

use poker_core::state::GameState;

use crate::resolving::belief_state::BeliefState;
use crate::resolving::cfv::opponent_cfvs;
use crate::resolving::gadget::ReSolver;
use crate::resolving::leaf_eval::LeafEvaluator;
use crate::resolving::subgame::{Subgame, SubgameSolver};
use crate::resolving::warm_start::{warm_start_regrets, DEFAULT_SCALE};

/// State carried between resolves in a continual re-solving session.
pub struct ContinualState {
    /// Our range (the re-solving player's).
    pub my_range: BeliefState,
    /// The opponent's range.
    pub opp_range: BeliefState,
    /// Opponent's carried per-hand counterfactual values (combo-indexed, bb);
    /// `None` before the first resolve (triggers the bootstrap).
    pub opp_cfv: Option<Vec<f64>>,
    /// Previous resolve's strategy, used to warm-start the next.
    pub last_strategy: Option<HashMap<u64, Vec<f64>>>,
}

impl ContinualState {
    /// Start a session with the given ranges and no carried values.
    pub fn new(my_range: BeliefState, opp_range: BeliefState) -> Self {
        Self { my_range, opp_range, opp_cfv: None, last_strategy: None }
    }

    /// Order the ranges into the `[player0, player1]` array `Subgame` expects.
    fn beliefs(&self, me: usize) -> [BeliefState; 2] {
        let mut b = [self.my_range.clone(), self.opp_range.clone()];
        if me == 1 {
            b.swap(0, 1);
        }
        b
    }

    /// Seed a session with externally supplied opponent CFVs (e.g. a blueprint's
    /// boundary values), so the very first resolve is already gadget-constrained.
    pub fn with_cfv(mut self, opp_cfv: Vec<f64>) -> Self {
        self.opp_cfv = Some(opp_cfv);
        self
    }
}

/// Drives continual re-solving over a sequence of decision points.
pub struct ContinualResolver {
    /// Iterations per resolve.
    pub iters: u64,
    /// Blueprint/warm-start confidence (see [`crate::resolving::warm_start`]).
    pub warm_start_scale: f64,
}

impl ContinualResolver {
    pub fn new(iters: u64) -> Self {
        Self { iters, warm_start_scale: DEFAULT_SCALE }
    }

    /// Resolve the decision at `root`, mutating the carried `state` (refreshing
    /// its opponent CFVs and last strategy).  Returns the deployable strategy.
    ///
    /// First call (no carried CFVs) bootstraps with a plain range-vs-range
    /// resolve; later calls use the safe, warm-started gadget resolve.
    pub fn resolve(
        &self,
        root: &GameState,
        state: &mut ContinualState,
        leaf_eval: &dyn LeafEvaluator,
    ) -> HashMap<u64, Vec<f64>> {
        let me = root.current_player();
        let opp = 1 - me;
        let beliefs = state.beliefs(me);

        let strategy = match &state.opp_cfv {
            None => SubgameSolver::new(1, 0)
                .solve_for_iters(root, &beliefs, leaf_eval, self.iters)
                .strategy,
            Some(cfv) => {
                let mut rs = ReSolver::new();
                if let Some(ls) = &state.last_strategy {
                    rs = rs.with_warm_start(warm_start_regrets(ls, self.warm_start_scale));
                }
                rs.solve_for_iters(root, &state.my_range, &state.opp_range, cfv, leaf_eval, self.iters)
                    .strategy
            }
        };

        // Refresh the carried opponent CFVs from this resolve (extracted on the
        // plain range-vs-range tree, whose play info keys the strategy shares).
        let sg = Subgame::new(root.clone(), &beliefs, leaf_eval);
        state.opp_cfv = Some(opponent_cfvs(&sg, &strategy, opp));
        state.last_strategy = Some(strategy.clone());
        strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolving::gadget::GadgetGame;
    use crate::resolving::leaf_eval::CheckdownLeafEval;
    use crate::solver::best_response::best_response_value;
    use poker_core::action::Action;
    use poker_core::legal_actions;
    use poker_core::make_card;
    use poker_core::state::{MAX_PLAYERS, NO_CARD};

    fn public_root(board: [u8; 5], stack: u32, target_street: u8) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        let mut used = 0u64;
        for &c in &board {
            if c != NO_CARD {
                used |= 1 << c;
            }
        }
        let mut spare = (0u8..52).filter(|&c| used & (1 << c) == 0);
        holes[0] = [spare.next().unwrap(), spare.next().unwrap()];
        holes[1] = [spare.next().unwrap(), spare.next().unwrap()];
        let mut gs = GameState::new(2, 2, 1, [stack; MAX_PLAYERS], holes, board, 0);
        while gs.street < target_street && !gs.is_terminal() {
            let acts = legal_actions(&gs);
            let act = if acts.iter().any(|&a| a == Action::Check) { Action::Check } else { Action::Call };
            gs.apply_action(act);
        }
        gs
    }

    fn river_board() -> [u8; 5] {
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)]
    }

    fn duel_ranges() -> (BeliefState, BeliefState) {
        let b0 = BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)],
            [make_card(6, 0), make_card(5, 0)],
        ]);
        let b1 = BeliefState::from_hands(&[
            [make_card(8, 0), make_card(8, 1)],
            [make_card(10, 0), make_card(9, 1)],
        ]);
        (b0, b1)
    }

    #[test]
    fn reentry_resolve_is_safe_and_stays_low_exploitability() {
        // Bootstrap the river, then re-solve the same boundary with the gadget
        // (the natural continual cadence).  The deployed object is *our*
        // strategy; its safety is how much the opponent can gain best-responding
        // to it.  The continual re-solve must be no more exploitable than the
        // bootstrap, and must hold the opponent to its carried guarantee.
        let (b0, b1) = duel_ranges();
        let root = public_root(river_board(), 20, 3);
        let me = root.current_player();
        let opp = 1 - me;
        let leaf = CheckdownLeafEval::new();

        let driver = ContinualResolver::new(4_000);
        let (my_range, opp_range) =
            if me == 0 { (b0.clone(), b1.clone()) } else { (b1.clone(), b0.clone()) };
        let mut state = ContinualState::new(my_range.clone(), opp_range.clone());

        // Bootstrap (plain resolve) → carries CFVs.  `ContinualState::beliefs`
        // always presents player0 = b0, player1 = b1, so score in that order.
        let boot = driver.resolve(&root, &mut state, &leaf);
        let guarantee = state.opp_cfv.clone().unwrap();
        let beliefs = [b0.clone(), b1.clone()];
        // Opponent's best-response value against *our* strategy (lower = safer);
        // the opponent's untrained part of the profile is irrelevant here since
        // it is best-responding.
        let br_boot = best_response_value(&Subgame::new(root.clone(), &beliefs, &leaf), opp, &boot);

        // Re-solve the same boundary (gadget, warm-started).
        let cont = driver.resolve(&root, &mut state, &leaf);
        let br_cont = best_response_value(&Subgame::new(root.clone(), &beliefs, &leaf), opp, &cont);
        println!("re-entry — opp BR vs bootstrap {br_boot:.5} bb, vs continual {br_cont:.5} bb");
        assert!(
            br_cont <= br_boot + 0.05,
            "continual re-solve must be no more exploitable than the bootstrap ({br_cont} vs {br_boot})"
        );

        // Safety: the gadget held the opponent to its carried guarantee.
        let game = GadgetGame::new(root.clone(), &my_range, &opp_range, &guarantee, &leaf);
        let mut worst = f64::MIN;
        for h in game.opp_hands() {
            worst = worst.max(game.follow_value(&cont, h) - game.cfv_of(h));
        }
        assert!(worst < 0.03, "opponent held to its guarantee (+ε); worst slack {worst}");
    }

    #[test]
    fn across_streets_carries_cfvs_into_a_safe_river_resolve() {
        // The cross-street wiring: resolve a turn, carry the opponent's per-hand
        // guarantee into a gadget-constrained river resolve.  We assert the
        // mechanism runs end-to-end and the river resolve is safe w.r.t. the
        // carried guarantee + produces valid distributions.  (Per-river the turn
        // guarantee is the river-averaged value — exact in expectation over the
        // dealt card; see the module scope note.)
        let (b0, b1) = duel_ranges();
        let turn = public_root(
            [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD],
            20,
            2,
        );
        let me = turn.current_player();
        let leaf = CheckdownLeafEval::new();
        let driver = ContinualResolver::new(3_000);
        let (my_range, opp_range) =
            if me == 0 { (b0.clone(), b1.clone()) } else { (b1.clone(), b0.clone()) };
        let mut state = ContinualState::new(my_range.clone(), opp_range.clone());

        // Resolve the turn → carries the opponent's turn guarantee.
        driver.resolve(&turn, &mut state, &leaf);
        let carried = state.opp_cfv.clone().unwrap();

        // Re-solve the river constrained by the carried turn CFVs.
        let river = public_root(river_board(), 20, 3);
        assert_eq!(river.current_player(), me, "OOP actor stable postflop");
        let river_strat = driver.resolve(&river, &mut state, &leaf);
        for probs in river_strat.values() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "valid distribution, got {sum}");
        }

        // Safety w.r.t. the carried guarantee on the river.
        let opp = 1 - me;
        let game = GadgetGame::new(river.clone(), &my_range, &opp_range, &carried, &leaf);
        let mut worst = f64::MIN;
        for h in game.opp_hands() {
            worst = worst.max(game.follow_value(&river_strat, h) - game.cfv_of(h));
        }
        println!("across-streets safety: max (follow − carried guarantee) = {worst:.5} bb");
        assert!(worst < 0.05, "river resolve held opp to the carried turn guarantee; slack {worst}");
        let _ = opp;
    }
}
