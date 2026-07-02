//! Small, exactly-solvable extensive-form games used to validate the solver.
//!
//! The validation protocol is explicit: the blueprint solver
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

/// A game traversed by a **mutable cursor** with `apply`/`undo`, for the sampled
/// solver's hot path.
///
/// [`Game`] returns a freshly-cloned child on every `apply`, which is the right,
/// clear design for the tiny validation games and exact best response.  But the
/// real-mechanics blueprint games ([`blueprint::BlueprintHoldem`],
/// [`push_fold::PushFoldHoldem`]) wrap a `poker_core::GameState`, whose clone
/// drags along a pre-allocated `UndoStack` — so the clone-based path
/// heap-allocates on *every tree node*, throwing away the zero-allocation
/// `apply_action`/`undo_action` design `poker_core` was built around.
///
/// `CursorGame` exposes that mutate-and-undo design to MCCFR: one `GameState` is
/// walked in place, children reached by `apply` and reversed by `undo`, with the
/// legal-action list (a `Copy` value) computed once per node and held on the
/// traverser's stack frame.  External-sampling MCCFR drives it through
/// [`crate::solver::mccfr::Mccfr::train_fast`] and
/// [`train_parallel_fast`](crate::solver::mccfr::Mccfr::train_parallel_fast),
/// which are proven **bit-identical** to the clone-based path for a fixed seed.
///
/// The trait is scoped to the **non-enumerable** sampled games: chance is a full
/// random deal reached only through [`sample_chance`](CursorGame::sample_chance),
/// so there is no enumerable-chance API here.
pub trait CursorGame {
    /// A mutable traversal cursor wrapping one in-place game state.
    type Cursor;

    /// A single legal action (applied to the cursor without re-deriving the
    /// legal set).
    type Action: Copy;

    /// The `Copy` list of legal actions at a node — computed once per node and
    /// indexed by action position `0..len`.
    type Actions: Copy + AsRef<[Self::Action]>;

    /// Number of players (always 2 for the heads-up blueprint games).
    fn num_players(&self) -> usize;

    /// A fresh cursor at the pre-deal chance root.
    fn root(&self) -> Self::Cursor;

    fn is_terminal(&self, cursor: &Self::Cursor) -> bool;
    fn is_chance(&self, cursor: &Self::Cursor) -> bool;

    /// Terminal utility for `player` (zero-sum).  Only valid at terminal nodes.
    fn utility(&self, cursor: &Self::Cursor, player: usize) -> f64;

    /// The player to act at a decision node.
    fn current_player(&self, cursor: &Self::Cursor) -> usize;

    /// The legal actions at the current decision node.
    fn legal(&self, cursor: &Self::Cursor) -> Self::Actions;

    /// Globally-unique information-set key for the acting player at `cursor`.
    fn info_key(&self, cursor: &Self::Cursor) -> u64;

    /// Apply the action at index `a` in [`legal`](CursorGame::legal) — `action`
    /// is `legal(cursor)[a]`, passed in so the cursor need not recompute the
    /// legal set.  `a` is recorded for perfect-recall history; the change is
    /// reversed by [`undo`](CursorGame::undo).
    fn apply(&self, cursor: &mut Self::Cursor, a: usize, action: Self::Action);

    /// Reverse the most recent [`apply`](CursorGame::apply).
    fn undo(&self, cursor: &mut Self::Cursor);

    /// Deal a chance outcome **in place** at the root, drawing uniform `[0, 1)`
    /// units from `next_unit`.  Reversed by [`undo_chance`](CursorGame::undo_chance).
    fn sample_chance(&self, cursor: &mut Self::Cursor, next_unit: impl FnMut() -> f64);

    /// Reverse [`sample_chance`](CursorGame::sample_chance), returning the cursor
    /// to the pre-deal root.
    fn undo_chance(&self, cursor: &mut Self::Cursor);
}

/// A [`CursorGame`] whose information-set space is **known up front** and maps to
/// a dense integer range, so the sampled solver can store regrets in a flat
/// Structure-of-Arrays table (computed index, no hashing, no dynamic discovery)
/// instead of a `HashMap` — the ~10×-smaller store the memory budget assumes.
///
/// The index is a pure function of the public situation (street, card bucket,
/// betting-sequence id), so the table layout is fixed at construction.  Only
/// games with a bounded, enumerable info-set space implement this; the validation
/// games and the unabstracted blueprint stay on the `HashMap` solver.
pub trait IndexedGame: CursorGame {
    /// Exclusive upper bound on [`info_set_index`](IndexedGame::info_set_index).
    fn info_set_capacity(&self) -> usize;

    /// Dense index in `0..info_set_capacity()` for the acting player's info set.
    fn info_set_index(&self, cursor: &Self::Cursor) -> usize;

    /// Number of legal actions at the info set with the given index (fixes the
    /// flat table's per-info-set width).
    fn actions_at(&self, index: usize) -> usize;
}
