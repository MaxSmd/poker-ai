//! The re-solving gadget — safe continual re-solving (finding #4).
//!
//! A plain [`Subgame`] resolve is anchored only on the opponent's *blueprint
//! range*; it can hand the opponent more than an earlier resolve promised, so
//! exploitability can grow across decisions.  The **re-solving gadget** (Burch,
//! Johanson & Bowling 2014; the safety engine of DeepStack) fixes this by
//! constraining the resolve with the opponent's **counterfactual values** from
//! the previous resolve.
//!
//! At the gadget root the opponent — per hand `h` — chooses between:
//!
//! * **Terminate**: take the previously guaranteed value `cfv(h)` and leave.
//! * **Follow**: enter the real subgame holding `h` (our hand dealt from *our*
//!   range), and play it out.
//!
//! The opponent maximizes, so it follows only with hands where following beats
//! its guarantee — which forces *our* resolved strategy to hold every hand to
//! `≤ cfv(h)`.  Solving the gadget to equilibrium therefore yields a strategy
//! whose exploitability cannot exceed the carried guarantee: the safety
//! property.  The betting subtree under Follow is the ordinary [`Subgame`] tree
//! (delegated to a [`Subgame::play_context`]), so gadget play info sets share
//! the plain resolve's keyspace and warm-start from it.
//!
//! Heads-up: `me = root.current_player()` (the re-solving player), `opp = 1−me`
//! (the gadget chooser).

use std::collections::HashMap;

use poker_core::state::{GameState, NO_CARD};

use crate::abstraction::features::combo_index;
use crate::games::Game;
use crate::resolving::belief_state::BeliefState;
use crate::resolving::cfv::node_value;
use crate::resolving::leaf_eval::LeafEvaluator;
use crate::resolving::subgame::{Subgame, SubgameNode};
use crate::solver::predictive::PredictiveSolver;
use crate::util::hash::fnv1a;

/// Opponent gadget action: enter the subgame.
const FOLLOW: usize = 0;
/// Opponent gadget action: take the guaranteed counterfactual value and leave.
const TERMINATE: usize = 1;
/// Info-key marker for a gadget choice node — disjoint from betting keys.
const GADGET_MARKER: u8 = 0xA6;

/// A node of the re-solving gadget game.
#[derive(Clone, Debug)]
pub enum GadgetState {
    /// Chance: pick the opponent's hand `h` (the gadget reasons per hand).
    Root,
    /// The opponent's Follow/Terminate decision for hand `h`.
    Choice([u8; 2]),
    /// Terminal: the opponent took its guaranteed value for hand `h`.
    Terminate([u8; 2]),
    /// Chance: deal *our* hand from our range, opponent holding `h`.
    FollowDeal([u8; 2]),
    /// A real betting node (delegated to the inner [`Subgame`]).
    Play(SubgameNode),
}

/// The re-solving gadget as a [`Game`]: re-solves **our** strategy subject to the
/// opponent's carried counterfactual-value constraints.
pub struct GadgetGame<'a> {
    /// Play-method delegate (betting tree / leaf eval / info_key), no deals.
    inner: Subgame<'a>,
    me: usize,
    opp: usize,
    /// Root chance children: `(Choice(h), q(h))` over the opponent's hands.
    root_outcomes: Vec<(GadgetState, f64)>,
    /// Follow chance children per opponent hand: `(Play(deal-root), prob)`.
    follow: HashMap<[u8; 2], Vec<(GadgetState, f64)>>,
    /// The opponent's guaranteed counterfactual value per hand (big blinds).
    cfv_of: HashMap<[u8; 2], f64>,
}

impl<'a> GadgetGame<'a> {
    /// Build the gadget rooted at `root` (the public resolve state), with our
    /// range `my_range`, the opponent's prior range `opp_range` (the gadget's
    /// hand-selection weight `q`), and the opponent's carried per-hand
    /// counterfactual values `cfv` (combo-indexed, big blinds).
    ///
    /// Opponent hands that are blocked by the board, carry no finite `cfv`, or
    /// have no legal deal of our hand are dropped (they cannot meaningfully
    /// follow).  Any positive `q` preserves the safety guarantee; using the
    /// opponent's range reach makes the deployed averaged strategy weight hands
    /// by plausibility.
    pub fn new(
        root: GameState,
        my_range: &BeliefState,
        opp_range: &BeliefState,
        cfv: &[f64],
        leaf_eval: &'a dyn LeafEvaluator,
    ) -> Self {
        let me = root.current_player();
        let opp = 1 - me;

        let mut board_mask = 0u64;
        for &c in &root.board {
            if c != NO_CARD {
                board_mask |= 1 << c;
            }
        }

        let inner = Subgame::play_context(&root, leaf_eval);
        let mut root_outcomes: Vec<(GadgetState, f64)> = Vec::new();
        let mut follow: HashMap<[u8; 2], Vec<(GadgetState, f64)>> = HashMap::new();
        let mut cfv_of: HashMap<[u8; 2], f64> = HashMap::new();
        let mut q_total = 0.0;

        for (h, ph) in opp_range.iter_nonzero() {
            let mh = (1u64 << h[0]) | (1u64 << h[1]);
            if mh & board_mask != 0 {
                continue;
            }
            let c = cfv[combo_index(h[0], h[1])];
            if !c.is_finite() {
                continue; // no carried guarantee for this hand → cannot constrain it
            }
            // Deal our hand from our range under "follow" (card removal vs h, board).
            let mut deals: Vec<(GadgetState, f64)> = Vec::new();
            let mut tot = 0.0;
            for (h0, p0) in my_range.iter_nonzero() {
                let m0 = (1u64 << h0[0]) | (1u64 << h0[1]);
                if m0 & board_mask != 0 || m0 & mh != 0 {
                    continue;
                }
                let mut holes = [[0u8; 2]; 2];
                holes[me] = h0;
                holes[opp] = h;
                deals.push((GadgetState::Play(SubgameNode::deal(&root, holes)), p0));
                tot += p0;
            }
            if deals.is_empty() {
                continue; // opponent cannot follow with this hand
            }
            for d in &mut deals {
                d.1 /= tot;
            }
            follow.insert(h, deals);
            cfv_of.insert(h, c);
            root_outcomes.push((GadgetState::Choice(h), ph));
            q_total += ph;
        }
        if q_total > 0.0 {
            for o in &mut root_outcomes {
                o.1 /= q_total;
            }
        }

        Self { inner, me, opp, root_outcomes, follow, cfv_of }
    }

    /// The re-solving player (whose strategy we deploy).
    pub fn me(&self) -> usize {
        self.me
    }

    /// The gadget chooser / constrained opponent.
    pub fn opp(&self) -> usize {
        self.opp
    }

    /// The opponent hands the gadget reasons over (those it can follow with).
    pub fn opp_hands(&self) -> Vec<[u8; 2]> {
        self.root_outcomes
            .iter()
            .map(|(s, _)| match s {
                GadgetState::Choice(h) => *h,
                _ => unreachable!("root outcomes are Choice nodes"),
            })
            .collect()
    }

    /// The carried guarantee for opponent hand `h`.
    pub fn cfv_of(&self, h: [u8; 2]) -> f64 {
        self.cfv_of[&h]
    }

    /// The opponent's value with hand `h` **if it follows**, under `strategy` —
    /// the quantity the safety property bounds by [`cfv_of`](Self::cfv_of).  At
    /// the gadget equilibrium `follow_value(h) ≤ cfv(h)` for every hand.
    pub fn follow_value(&self, strategy: &HashMap<u64, Vec<f64>>, h: [u8; 2]) -> f64 {
        node_value(self, &GadgetState::FollowDeal(h), strategy, self.opp)
    }
}

impl Game for GadgetGame<'_> {
    type State = GadgetState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> GadgetState {
        GadgetState::Root
    }

    fn is_terminal(&self, state: &GadgetState) -> bool {
        match state {
            GadgetState::Terminate(_) => true,
            GadgetState::Play(n) => self.inner.is_terminal(n),
            _ => false,
        }
    }

    fn is_chance(&self, state: &GadgetState) -> bool {
        matches!(state, GadgetState::Root | GadgetState::FollowDeal(_))
    }

    fn utility(&self, state: &GadgetState, player: usize) -> f64 {
        match state {
            GadgetState::Terminate(h) => {
                let v = self.cfv_of[h];
                if player == self.opp {
                    v
                } else {
                    -v
                }
            }
            GadgetState::Play(n) => self.inner.utility(n, player),
            _ => panic!("utility at a non-terminal gadget node"),
        }
    }

    fn chance_outcomes(&self, state: &GadgetState) -> Vec<(GadgetState, f64)> {
        match state {
            GadgetState::Root => self.root_outcomes.clone(),
            GadgetState::FollowDeal(h) => self.follow[h].clone(),
            _ => panic!("chance_outcomes at a non-chance gadget node"),
        }
    }

    fn current_player(&self, state: &GadgetState) -> usize {
        match state {
            GadgetState::Choice(_) => self.opp,
            GadgetState::Play(n) => self.inner.current_player(n),
            _ => panic!("current_player at a non-decision gadget node"),
        }
    }

    fn num_actions(&self, state: &GadgetState) -> usize {
        match state {
            GadgetState::Choice(_) => 2, // FOLLOW / TERMINATE
            GadgetState::Play(n) => self.inner.num_actions(n),
            _ => panic!("num_actions at a non-decision gadget node"),
        }
    }

    fn apply(&self, state: &GadgetState, action: usize) -> GadgetState {
        match state {
            GadgetState::Choice(h) => {
                if action == FOLLOW {
                    GadgetState::FollowDeal(*h)
                } else {
                    debug_assert_eq!(action, TERMINATE);
                    GadgetState::Terminate(*h)
                }
            }
            GadgetState::Play(n) => GadgetState::Play(self.inner.apply(n, action)),
            _ => panic!("apply at a non-decision gadget node"),
        }
    }

    fn info_key(&self, state: &GadgetState) -> u64 {
        match state {
            GadgetState::Choice(h) => {
                let mut hh = *h;
                hh.sort_unstable();
                // Keyed on the chooser + its own hand + a marker, so the choice
                // info set is per-hand and cannot collide with a betting key.
                fnv1a(&[self.opp as u8, hh[0], hh[1], GADGET_MARKER])
            }
            GadgetState::Play(n) => self.inner.info_key(n),
            _ => panic!("info_key at a non-decision gadget node"),
        }
    }
}

/// Output of a gadget-constrained resolve.
pub struct ReSolved {
    /// Deployable strategy (CFR⁺ last iterate) over the gadget's info sets,
    /// including our betting strategy on the shared [`Subgame`] keyspace.
    pub strategy: HashMap<u64, Vec<f64>>,
    /// Linearly-averaged strategy — the smooth, monotone-converging object (the
    /// last iterate oscillates); used for convergence measurement.
    pub average: HashMap<u64, Vec<f64>>,
    /// Number of information sets discovered.
    pub info_sets: usize,
}

/// Gadget-constrained subgame solver — the safe re-solving entry point, mirroring
/// [`crate::resolving::subgame::SubgameSolver`] but constrained by carried CFVs.
#[derive(Default)]
pub struct ReSolver {
    warm_start: Option<HashMap<u64, Vec<f64>>>,
}

impl ReSolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Warm-start the (predictive) solver's regrets — typically the previous
    /// resolve's strategy converted via [`crate::resolving::warm_start`].
    pub fn with_warm_start(mut self, seed_regrets: HashMap<u64, Vec<f64>>) -> Self {
        self.warm_start = Some(seed_regrets);
        self
    }

    /// Resolve the gadget at `root` for a fixed iteration count.
    pub fn solve_for_iters(
        &self,
        root: &GameState,
        my_range: &BeliefState,
        opp_range: &BeliefState,
        cfv: &[f64],
        leaf_eval: &dyn LeafEvaluator,
        iters: u64,
    ) -> ReSolved {
        let game = GadgetGame::new(root.clone(), my_range, opp_range, cfv, leaf_eval);
        let mut solver = PredictiveSolver::new(game);
        if let Some(seed) = &self.warm_start {
            solver.warm_start(seed.clone());
        }
        solver.train(iters);
        ReSolved {
            strategy: solver.current_strategy(),
            average: solver.average_strategy(),
            info_sets: solver.num_info_sets(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolving::leaf_eval::CheckdownLeafEval;
    use crate::resolving::cfv::opponent_cfvs;
    use crate::resolving::subgame::{Subgame, SubgameSolver};
    use crate::solver::best_response::exploitability;
    use poker_core::action::Action;
    use poker_core::legal_actions;
    use poker_core::make_card;
    use poker_core::state::MAX_PLAYERS;

    fn river_board() -> [u8; 5] {
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)]
    }

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

    fn duel_ranges() -> (BeliefState, BeliefState) {
        let b0 = BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)], // trips (nuts-ish)
            [make_card(6, 0), make_card(5, 0)],   // air
        ]);
        let b1 = BeliefState::from_hands(&[
            [make_card(8, 0), make_card(8, 1)],   // bluff-catcher
            [make_card(10, 0), make_card(9, 1)],  // weak
        ]);
        (b0, b1)
    }

    /// The true equilibrium opponent CFVs for a river root: resolve unconstrained
    /// and extract them.  `me`/`opp` follow the root's actor.
    fn true_cfvs(root: &GameState, b0: &BeliefState, b1: &BeliefState) -> Vec<f64> {
        let leaf = CheckdownLeafEval::new();
        let resolved = SubgameSolver::new(1, 0).solve_for_iters(root, &[b0.clone(), b1.clone()], &leaf, 4_000);
        let sg = Subgame::new(root.clone(), &[b0.clone(), b1.clone()], &leaf);
        let opp = 1 - root.current_player();
        opponent_cfvs(&sg, &resolved.strategy, opp)
    }

    /// Order ranges so `my_range` belongs to the root actor (`me`) and
    /// `opp_range` to the gadget chooser — matching the carried CFV's player.
    fn by_seat<'a>(
        root: &GameState,
        b0: &'a BeliefState,
        b1: &'a BeliefState,
    ) -> (&'a BeliefState, &'a BeliefState) {
        if root.current_player() == 0 { (b0, b1) } else { (b1, b0) }
    }

    #[test]
    fn gadget_is_well_formed_and_zero_sum() {
        let (b0, b1) = duel_ranges();
        let root = public_root(river_board(), 20, 3);
        let cfv = true_cfvs(&root, &b0, &b1);
        let leaf = CheckdownLeafEval::new();
        let (mr, opr) = by_seat(&root, &b0, &b1);
        let game = GadgetGame::new(root.clone(), mr, opr, &cfv, &leaf);
        assert_eq!(game.opp(), 1 - root.current_player());
        assert!(!game.opp_hands().is_empty(), "gadget has opponent choice hands");

        // Zero-sum at every leaf; a Choice node owned by the opponent exists.
        fn walk(g: &GadgetGame, s: &GadgetState, opp: usize, saw_choice: &mut bool) {
            if g.is_terminal(s) {
                assert!((g.utility(s, 0) + g.utility(s, 1)).abs() < 1e-9, "zero-sum leaf");
                return;
            }
            if g.is_chance(s) {
                for (c, _) in g.chance_outcomes(s) {
                    walk(g, &c, opp, saw_choice);
                }
                return;
            }
            if matches!(s, GadgetState::Choice(_)) {
                *saw_choice = true;
                assert_eq!(g.current_player(s), opp, "the opponent chooses follow/terminate");
                assert_eq!(g.num_actions(s), 2);
            }
            for a in 0..g.num_actions(s) {
                walk(g, &g.apply(s, a), opp, saw_choice);
            }
        }
        let mut saw = false;
        walk(&game, &game.root(), game.opp(), &mut saw);
        assert!(saw, "the gadget must contain a follow/terminate choice node");
    }

    #[test]
    fn gadget_holds_opponent_to_its_guarantee() {
        // The safety property: after resolving, the opponent's value if it
        // follows is ≤ its carried guarantee for every hand.
        let (b0, b1) = duel_ranges();
        let root = public_root(river_board(), 20, 3);
        let cfv = true_cfvs(&root, &b0, &b1);
        let leaf = CheckdownLeafEval::new();
        let (mr, opr) = by_seat(&root, &b0, &b1);

        let resolved = ReSolver::new().solve_for_iters(&root, mr, opr, &cfv, &leaf, 5_000);
        let game = GadgetGame::new(root.clone(), mr, opr, &cfv, &leaf);

        let mut worst = f64::MIN;
        for h in game.opp_hands() {
            let slack = game.follow_value(&resolved.strategy, h) - game.cfv_of(h);
            worst = worst.max(slack);
        }
        println!("gadget safety: max (follow_value − guarantee) = {worst:.5} bb");
        assert!(worst < 0.02, "opponent must be held to its guarantee (+ε); worst slack {worst}");
    }

    #[test]
    fn true_cfvs_yield_no_distortion() {
        // Feeding the true equilibrium CFVs, the gadget reproduces a near-optimal
        // strategy (it does not distort a correct input): low exploitability in
        // the gadget game itself, measured by exact best response.
        let (b0, b1) = duel_ranges();
        let root = public_root(river_board(), 20, 3);
        let cfv = true_cfvs(&root, &b0, &b1);
        let leaf = CheckdownLeafEval::new();
        let (mr, opr) = by_seat(&root, &b0, &b1);

        let resolved = ReSolver::new().solve_for_iters(&root, mr, opr, &cfv, &leaf, 6_000);
        let game = GadgetGame::new(root.clone(), mr, opr, &cfv, &leaf);
        let expl = exploitability(&game, &resolved.strategy);
        println!("gadget exploitability with true CFVs: {expl:.5} bb");
        assert!(expl < 0.05, "gadget resolve with true CFVs should be near-optimal, got {expl}");
    }
}
