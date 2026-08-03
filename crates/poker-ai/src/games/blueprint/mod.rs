//! Sampled, card-abstracted heads-up NLHE — the real blueprint target.
//!
//! The curated-deal bridge ([`crate::validation::games::nlhe`]) proved the wiring by enumerating a
//! handful of concrete deals.  A blueprint cannot enumerate: the chance space is
//! every hole-card and board combination, ~10^9 deals before the betting tree
//! even begins.  This module closes the two gaps the bridge left open:
//!
//!  1. **Sampled chance.**  [`Game::sample_chance`](crate::games::Game::sample_chance) deals a fresh random board
//!     and both hands by partial Fisher–Yates over a 52-card deck, so the
//!     solver never materializes the outcome list.  [`is_chance_enumerable`]
//!     returns `false`, which routes external-sampling MCCFR onto that path
//!     (and, correctly, makes the full-traversal solver and exact best response
//!     inapplicable — there is no finite tree to walk).
//!
//!  2. **Card abstraction in the key.**  Information sets are keyed on the
//!     *bucket* of the situation, not the raw cards: a per-street [`BucketMap`]
//!     ([`crate::abstraction`]) collapses strategically-similar `(hole, board)`
//!     situations together, which is what makes the regret table finite.
//!     Pre-flop uses the 169 suit-canonical hand classes directly; a street
//!     with no loaded abstraction falls back to its suit-canonical key (correct,
//!     just unabstracted).
//!
//! [`is_chance_enumerable`]: crate::games::Game::is_chance_enumerable

mod indexing;
mod keys;
mod traits;
#[cfg(test)]
mod tests;

use poker_core::action::{Action, ActionList};
use poker_core::legal_actions;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use crate::abstraction::bucket_map::BucketMap;
use crate::abstraction::hand_index::HandIndexer;

use indexing::Indexing;

/// Number of cards consumed by a heads-up deal: 2 hole cards each + 5 board.
const DEAL_CARDS: usize = 9;

/// Maximum game-tree depth a single hand can reach (`apply` calls without an
/// `undo`).  Sized to `poker_core`'s own undo-stack cap so the inline cursor
/// history can never overflow where the engine itself would not.
const MAX_DEPTH: usize = poker_core::undo::MAX_UNDO_DEPTH;

/// A heads-up NLHE game with sampled deals and per-street card abstraction.
pub struct BlueprintHoldem {
    stacks: [u32; MAX_PLAYERS],
    big_blind: u32,
    small_blind: u32,
    button: u8,
    /// Information abstraction for the post-flop streets, indexed
    /// `flop = 0, turn = 1, river = 2`.  `None` ⇒ that street is unabstracted.
    street_buckets: [Option<BucketMap>; 3],
    /// Dense hand indexers per post-flop street (`[2,3] / [2,4] / [2,5]`), used
    /// for the unabstracted fallback key when a street has no bucket map.
    indexers: [HandIndexer; 3],
    /// Maximum number of **raises per street** the betting abstraction allows.
    /// This is the dominant tree-size / memory lever (see `memory_estimate`):
    /// `poker_core` itself caps nothing (it re-offers reraises until stacks
    /// deplete), so bounding it is a blueprint-abstraction choice that lives
    /// here, not in the faithful engine.  `u32::MAX` (the `new` default) means
    /// uncapped — identical to the raw engine behaviour.
    raise_cap: u32,
    /// Dense info-set indexing for the flat SoA regret store, built by
    /// [`with_indexing`](BlueprintHoldem::with_indexing).  `None` until then;
    /// only the `HashMap`-keyed [`Game`]/[`crate::games::CursorGame`] paths work without
    /// it.  Present ⇒ the game also implements [`crate::games::IndexedGame`].
    indexing: Option<Indexing>,
}

/// A node: the pre-deal chance root (`gs == None`) or a play node wrapping a
/// concrete `GameState` plus the perfect-recall action history.
#[derive(Clone, Debug)]
pub struct BlueprintState {
    gs: Option<GameState>,
    history: Vec<u8>,
    /// Raises made so far on the **current** street (resets each street) — drives
    /// the [`BlueprintHoldem::raise_cap`] betting abstraction.
    street_raises: u8,
}

/// A zero-allocation traversal cursor for [`BlueprintHoldem`]: one `GameState`
/// walked in place via `apply_action`/`undo_action`, plus an inline
/// perfect-recall history (no per-node `Vec`).
pub struct BlueprintCursor {
    /// `None` at the pre-deal chance root; `Some` once a deal has been sampled.
    gs: Option<GameState>,
    /// Action indices taken from the root, the perfect-recall history.
    history: [u8; MAX_DEPTH],
    /// Current depth — number of valid entries in `history`.
    depth: usize,
    /// Raises made so far on the current street (the cursor counterpart of
    /// [`BlueprintState::street_raises`], maintained in place by `apply`/`undo`).
    street_raises: u8,
    /// `street_raises` *before* the action at each depth, so `undo` can restore
    /// it in O(1) (the inline counterpart of cloning the state).
    raises_at: [u8; MAX_DEPTH],
}

impl BlueprintHoldem {
    /// A game with equal starting stacks and no card abstraction loaded
    /// (every street keyed by its suit-canonical situation).
    pub fn new(stack: u32, big_blind: u32, small_blind: u32, button: u8) -> Self {
        let mut stacks = [0u32; MAX_PLAYERS];
        stacks[0] = stack;
        stacks[1] = stack;
        Self {
            stacks,
            big_blind,
            small_blind,
            button,
            street_buckets: [None, None, None],
            indexers: [
                HandIndexer::new(&[2, 3]),
                HandIndexer::new(&[2, 4]),
                HandIndexer::new(&[2, 5]),
            ],
            raise_cap: u32::MAX,
            indexing: None,
        }
    }

    /// Attach a street's information abstraction (`flop = 0, turn = 1,
    /// river = 2`).
    pub fn with_street_bucket(mut self, street: usize, buckets: BucketMap) -> Self {
        self.street_buckets[street] = Some(buckets);
        self
    }

    /// Cap the betting abstraction at `cap` raises per street — the tree-size
    /// lever (`memory_estimate` prints the exact footprint per stack × cap;
    /// heads-up 20 bb cap-2 is ~4.6 M info sets ≈ 0.21 GB on the SoA store).
    /// `0` is treated as `1` (at least the opening raise must stay legal or the
    /// tree degenerates to check/call only).
    pub fn with_raise_cap(mut self, cap: u32) -> Self {
        self.raise_cap = cap.max(1);
        self
    }

    /// Build the dense info-set `Indexing` so the game can drive the flat SoA
    /// regret store ([`crate::games::IndexedGame`] / [`crate::solver::mccfr::SoaMccfr`])
    /// — the ~10×-smaller blueprint store `docs/memory-budget.md` assumes.
    ///
    /// Requires a **finite raise cap** (the uncapped betting tree is unbounded)
    /// and a **full-coverage** abstraction on every post-flop street (the dense
    /// index has one slot per `(sequence, bucket)`, so an out-of-set situation
    /// has nowhere to go — unlike the `HashMap` path, which mints a fresh raw
    /// key).  Both are exactly the cloud-burst configuration; call this last,
    /// after [`with_raise_cap`](BlueprintHoldem::with_raise_cap) and
    /// [`with_street_bucket`](BlueprintHoldem::with_street_bucket).
    pub fn with_indexing(mut self) -> Self {
        assert!(
            self.raise_cap != u32::MAX,
            "SoA indexing needs a finite raise cap (call with_raise_cap first); \
             the uncapped betting tree is unbounded"
        );
        assert!(
            self.street_buckets.iter().all(Option::is_some),
            "SoA indexing needs full-coverage flop/turn/river bucket maps \
             (call with_street_bucket for every post-flop street)"
        );
        self.indexing = Some(self.build_indexing());
        self
    }

    /// Legal actions at `gs` after applying the raise-cap betting abstraction:
    /// once `street_raises` reaches the cap, sized `Raise`s are removed, leaving
    /// fold / check / call / all-in.  Both the clone and cursor paths route
    /// through this, so action indices (and thus info keys) stay identical.
    ///
    /// `AllIn` survives the cap deliberately.  It is the only *absorbing*
    /// aggressive action — once it is called, neither game can raise again — so
    /// keeping it means a raise war always has a terminating node and every
    /// real bet, however large and however deep in the war, has somewhere to
    /// map.  Dropping it (as this once did) left the abstraction not closed
    /// under opponent aggression: a shove past the cap could not be translated
    /// at all, and the playing agent was left with no node to act from.
    fn capped_legal(&self, gs: &GameState, street_raises: u8) -> ActionList {
        Self::capped_legal_at(gs, street_raises, self.raise_cap)
    }

    /// The raise-cap filter as a pure function of `(state, raise count, cap)`.
    /// Public so `bin/memory_estimate` enumerates the tree with **this exact
    /// policy** rather than a copy that could drift from the trained game.
    pub fn capped_legal_at(gs: &GameState, street_raises: u8, raise_cap: u32) -> ActionList {
        let full = legal_actions(gs);
        if (street_raises as u32) < raise_cap {
            return full;
        }
        let mut buf = [Action::Fold; 8];
        let mut n = 0;
        for &a in full.iter() {
            if !matches!(a, Action::Raise(_)) {
                buf[n] = a;
                n += 1;
            }
        }
        ActionList::from_actions(&buf[..n])
    }

    /// Raises on the current street after an action took `gs` from `old_street`/
    /// `old_bet` to its present state: reset on a street change, +1 when the bet
    /// level rose (a raise or all-in-raise), unchanged otherwise.  Public for the
    /// same reason as [`capped_legal_at`](Self::capped_legal_at).
    pub fn next_raises(prev: u8, old_street: u8, old_bet: u32, gs: &GameState) -> u8 {
        if gs.street != old_street {
            0
        } else if gs.current_bet > old_bet {
            prev.saturating_add(1)
        } else {
            prev
        }
    }

    /// Deal both hands and the full board from a freshly shuffled deck, drawing
    /// uniform units from `next_unit`.  Partial Fisher–Yates: only the first
    /// `DEAL_CARDS` positions are resolved.
    fn deal(&self, mut next_unit: impl FnMut() -> f64) -> GameState {
        // Cards are encoded `rank << 2 | suit`, so 0..52 enumerates the deck.
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        for i in 0..DEAL_CARDS {
            let span = 52 - i;
            let j = i + (next_unit() * span as f64) as usize;
            deck.swap(i, j.min(51));
        }
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [deck[0], deck[1]];
        holes[1] = [deck[2], deck[3]];
        let board = [deck[4], deck[5], deck[6], deck[7], deck[8]];
        GameState::new(2, self.big_blind, self.small_blind, self.stacks, holes, board, self.button)
    }

    // ------------------------------------------------------------------
    // Play-time API (`crate::play`): track a real hand through the abstract
    // game and read blueprint keys for arbitrary hypothetical holdings.
    // ------------------------------------------------------------------

    /// Construct a play node from concrete cards: both hole pairs plus the
    /// board known so far (`NO_CARD` for unrevealed cards), with no action
    /// history.  The entry point for play-time tracking of a real hand; advance
    /// it with [`Game::apply`](crate::games::Game::apply) using indices into [`actions`](Self::actions).
    pub fn play_state(&self, holes: [[u8; 2]; 2], board: [u8; 5]) -> BlueprintState {
        let mut all = [[NO_CARD; 2]; MAX_PLAYERS];
        all[0] = holes[0];
        all[1] = holes[1];
        let gs =
            GameState::new(2, self.big_blind, self.small_blind, self.stacks, all, board, self.button);
        BlueprintState { gs: Some(gs), history: Vec::new(), street_raises: 0 }
    }

    /// The capped legal actions at a play node — the very list whose indices
    /// [`Game::apply`](crate::games::Game::apply) takes and the info-key history records.
    pub fn actions(&self, state: &BlueprintState) -> ActionList {
        let gs = state.gs.as_ref().expect("actions at a play node");
        self.capped_legal(gs, state.street_raises)
    }

    /// The wrapped engine state of a play node (pot, bets, street — read-only).
    pub fn game_state<'s>(&self, state: &'s BlueprintState) -> &'s GameState {
        state.gs.as_ref().expect("game_state at a play node")
    }

    /// The information key the acting player at `state` would have if it held
    /// `hole` instead of its dealt cards — the belief-update primitive
    /// (likelihood of an observed action given each opponent hand).
    pub fn info_key_with_hole(&self, state: &BlueprintState, hole: [u8; 2]) -> u64 {
        let gs = state.gs.as_ref().expect("info_key_with_hole at a play node");
        let visible = gs.board_cards_count();
        self.key_for_cards(gs.current_player(), hole, &gs.board[..visible], &state.history)
    }

    /// The big blind in the game's chip units (play-time chip↔bb conversion).
    pub fn big_blind_chips(&self) -> u32 {
        self.big_blind
    }

    // ------------------------------------------------------------------
    // Raw-walk API (`crate::evaluation::vector_br`): drive the abstract
    // betting tree over a bare `GameState` (mutate-and-undo), outside the
    // `Game`/`CursorGame` plumbing, while staying on exactly the same
    // action menus, raise bookkeeping, and info keys as training.
    // ------------------------------------------------------------------

    /// The capped legal actions at a bare engine state (see `capped_legal`).
    pub fn capped_actions(&self, gs: &GameState, street_raises: u8) -> ActionList {
        self.capped_legal(gs, street_raises)
    }

    /// The info key `player` has holding `hole` at `(board, history)` — the
    /// card-based form of [`key_from_bucket`](Self::key_from_bucket), for
    /// callers that have concrete cards rather than a hoisted bucket.
    pub fn key_for(&self, player: usize, hole: [u8; 2], board: &[u8], history: &[u8]) -> u64 {
        self.key_for_cards(player, hole, board, history)
    }

    /// Street-raise counter after an action moved `gs` from
    /// `(old_street, old_bet)` to its current state (see `next_raises`).
    pub fn raises_after(&self, prev: u8, old_street: u8, old_bet: u32, gs: &GameState) -> u8 {
        Self::next_raises(prev, old_street, old_bet, gs)
    }

    /// The card bucket for every one of the 1326 hole combos on `board`
    /// (indexed by [`crate::abstraction::features::combo_index`]) — the bulk
    /// form of the per-hand bucket inside the info key, hoisted so tree
    /// walkers compute it once per board prefix instead of once per node.
    /// Combos that overlap the board get an arbitrary bucket (their reach is
    /// zero everywhere they could be queried).
    pub fn bucket_vector(&self, board: &[u8]) -> Vec<u64> {
        let mut out = vec![0u64; 1326];
        let mut blocked = 0u64;
        for &c in board {
            blocked |= 1 << c;
        }
        for hi in 1..52u8 {
            for lo in 0..hi {
                let idx = crate::abstraction::features::combo_index(hi, lo);
                if blocked & (1 << hi) != 0 || blocked & (1 << lo) != 0 {
                    continue;
                }
                out[idx] = self.situation_bucket(&[lo, hi], board);
            }
        }
        out
    }

    /// The per-player starting stack in chips (seat 0's; all seats equal).
    pub fn stack_chips(&self) -> u32 {
        self.stacks[0]
    }
}
