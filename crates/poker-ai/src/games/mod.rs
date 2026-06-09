//! Small, exactly-solvable extensive-form games used to validate the solver.
//!
//! The plan (Phase 3, validation protocol) is explicit: the blueprint solver
//! must be proven correct on games with known equilibria — Kuhn and Leduc —
//! *before* it is trusted on full No-Limit Hold'em.  A bug in a sampled 6-max
//! MCCFR loop manifests as "convergence is weird", not a crash, so the only way
//! to trust the solver is to first watch it reproduce a known exact solution.
//!
//! These games live behind the [`Game`] trait so that the *same* CFR and
//! best-response code that validates here is the code that later drives the
//! real game tree.  They are validation fixtures, not part of the production
//! NLHE path, which is why they sit in their own module rather than under
//! `abstraction` or `solver`.

pub mod blueprint;
pub mod kuhn;
pub mod leduc;
pub mod nlhe;
pub mod push_fold;

/// A two-player, zero-sum, perfect-recall extensive-form game with explicit
/// chance nodes.
///
/// States are cheap, cloneable values (the validation games are tiny, so the
/// solver clones states freely rather than threading a mutable cursor).  Action
/// are addressed by index `0..num_actions(state)`.
///
/// Information sets are identified by an opaque `u64` key.  The key must be
/// unique across *all* information sets in the game — including across players —
/// so that a single regret table can be addressed by key without collision.
pub trait Game {
    /// A node in the game tree.
    type State: Clone;

    /// Number of players (always 2 for the validation games).
    fn num_players(&self) -> usize;

    /// The root node.  May be a chance node (e.g. the initial deal).
    fn root(&self) -> Self::State;

    /// Whether `state` is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Whether `state` is a chance node (e.g. a card deal).
    fn is_chance(&self, state: &Self::State) -> bool;

    /// Terminal utility for `player` (zero-sum: the other player's utility is
    /// the negation).  Only meaningful at terminal states.
    fn utility(&self, state: &Self::State, player: usize) -> f64;

    /// Chance outcomes as `(child, probability)` pairs.  Probabilities sum to 1.
    /// Only called at chance nodes.
    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::State, f64)>;

    /// The player to act at a decision node.
    fn current_player(&self, state: &Self::State) -> usize;

    /// Number of legal actions at a decision node.
    fn num_actions(&self, state: &Self::State) -> usize;

    /// Apply action index `action` (`0..num_actions`) at a decision node.
    fn apply(&self, state: &Self::State, action: usize) -> Self::State;

    /// Globally-unique information-set key for the acting player at `state`.
    fn info_key(&self, state: &Self::State) -> u64;

    /// Key identifying a chance node for the VR-MCCFR baseline (a control
    /// variate is kept per chance node, sized to its number of outcomes).
    /// Distinct chance contexts should map to distinct keys; the default
    /// collapses all chance nodes to one, which is fine for single-deal games.
    fn chance_key(&self, _state: &Self::State) -> u64 {
        0
    }

    /// Whether the chance outcomes at `state` can be enumerated by
    /// [`chance_outcomes`](Game::chance_outcomes).
    ///
    /// The validation games (and the curated-deal NLHE bridge) have small,
    /// listable chance spaces and return `true`, which lets full-traversal CFR
    /// and the per-outcome VR-MCCFR baseline operate exactly.  A real NLHE deal
    /// — every hole-card and board combination — cannot be enumerated; such a
    /// game returns `false`, and the sampled solver must reach its children
    /// through [`sample_chance`](Game::sample_chance) instead.
    fn is_chance_enumerable(&self, _state: &Self::State) -> bool {
        true
    }

    /// Sample a single chance outcome at `state`, drawing uniform `[0, 1)` units
    /// from `next_unit` as needed.
    ///
    /// The default samples one outcome from the enumerated distribution and is
    /// correct for any enumerable chance node.  Games whose chance space is too
    /// large to enumerate (a full 52-card deal) override this to construct an
    /// outcome directly — e.g. by shuffling a deck with repeated `next_unit`
    /// draws — without ever materializing the outcome list.
    fn sample_chance(
        &self,
        state: &Self::State,
        mut next_unit: impl FnMut() -> f64,
    ) -> Self::State {
        let outcomes = self.chance_outcomes(state);
        let u = next_unit();
        let mut acc = 0.0;
        for (child, p) in &outcomes {
            acc += p;
            if u < acc {
                return child.clone();
            }
        }
        outcomes.last().expect("chance node must have at least one outcome").0.clone()
    }
}
