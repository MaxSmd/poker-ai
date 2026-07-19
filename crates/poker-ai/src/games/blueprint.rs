//! Sampled, card-abstracted heads-up NLHE — the real blueprint target.
//!
//! The curated-deal bridge ([`super::nlhe`]) proved the wiring by enumerating a
//! handful of concrete deals.  A blueprint cannot enumerate: the chance space is
//! every hole-card and board combination, ~10^9 deals before the betting tree
//! even begins.  This module closes the two gaps the bridge left open:
//!
//!  1. **Sampled chance.**  [`Game::sample_chance`] deals a fresh random board
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
//! [`is_chance_enumerable`]: Game::is_chance_enumerable

use poker_core::action::{Action, ActionList};
use poker_core::legal_actions;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use super::Game;
use crate::util::hash::Fnv1a;
use crate::abstraction::bucket_map::BucketMap;
use crate::abstraction::canonical::preflop_index;
use crate::abstraction::hand_index::HandIndexer;

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
    /// only the `HashMap`-keyed [`Game`]/[`super::CursorGame`] paths work without
    /// it.  Present ⇒ the game also implements [`super::IndexedGame`].
    indexing: Option<Indexing>,
}

/// A deal-independent enumeration of the abstract betting tree under the raise
/// cap, mapping every reachable decision **history** to a dense sequence id — the
/// backbone of the flat SoA info-set index.
///
/// The key fact that makes this exact: legal actions depend only on the *public*
/// chip state (pot / current bet / stacks / street), never on card identities.
/// So one skeleton deal enumerates every betting sequence the solver can ever
/// reach, and a dense info-set index is simply `sequence_offset + card_bucket`,
/// where the card bucket ranges over `0..buckets_for(street)`.  This partitions
/// information sets **identically** to the `HashMap` key
/// [`info_key_for`](BlueprintHoldem::info_key_for) (which keys on
/// `player + visible + bucket + history`, and player/visible are themselves pure
/// functions of the history) — proven by `indexed_partition_matches_info_key`.
struct Indexing {
    /// Per decision node: child node id for each action index (`-1` = the action
    /// leads to a terminal, i.e. no child decision node).
    children: Vec<[i32; 8]>,
    /// Per node: board cards visible (`0` pre-flop / `3` flop / `4` turn /
    /// `5` river) → which street's bucket count this node draws from.
    visible: Vec<u8>,
    /// Per node: the player to act (only used to reconstruct the info key on
    /// export — [`info_key_at`](BlueprintHoldem::info_key_at)).
    to_act: Vec<u8>,
    /// Per node: number of (capped) legal actions — the flat table's width.
    num_actions: Vec<u8>,
    /// Per node: parent node id (`-1` at the root) and the action index taken
    /// from the parent, to rebuild the perfect-recall history on export.
    parent: Vec<i32>,
    in_action: Vec<u8>,
    /// Per node: base dense index of its `buckets_for(visible)` bucket block.
    seq_offset: Vec<u32>,
    /// Number of legal actions per dense info-set index (drives the table layout).
    actions_by_index: Vec<u8>,
    /// Total info sets = `Σ buckets_for(visible[node])`.
    capacity: usize,
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

    /// Build the dense info-set [`Indexing`] so the game can drive the flat SoA
    /// regret store ([`super::IndexedGame`] / [`crate::solver::mccfr::SoaMccfr`])
    /// — the ~10×-smaller blueprint store the memory budget assumes (finding #4).
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

    fn indexing(&self) -> &Indexing {
        self.indexing.as_ref().expect("call with_indexing() before SoA training")
    }

    /// Number of card buckets a street contributes: 169 pre-flop classes, else
    /// the loaded bucket map's count.
    fn buckets_for_visible(&self, visible: u8) -> u32 {
        match visible {
            0 => 169,
            3..=5 => {
                self.street_buckets[visible as usize - 3].as_ref().expect("map present").num_buckets()
            }
            other => unreachable!("impossible board-card count {other}"),
        }
    }

    /// Enumerate the abstract betting tree from a skeleton deal (card identities
    /// are irrelevant to betting legality), then lay out the dense info-set
    /// index by giving each decision node a contiguous `buckets_for(street)` block.
    fn build_indexing(&self) -> Indexing {
        // Skeleton deal: any nine distinct real cards drive the public tree.
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [0, 1];
        holes[1] = [2, 3];
        let board = [4, 5, 6, 7, 8];
        let gs = GameState::new(2, self.big_blind, self.small_blind, self.stacks, holes, board, self.button);

        let mut idx = Indexing {
            children: Vec::new(),
            visible: Vec::new(),
            to_act: Vec::new(),
            num_actions: Vec::new(),
            parent: Vec::new(),
            in_action: Vec::new(),
            seq_offset: Vec::new(),
            actions_by_index: Vec::new(),
            capacity: 0,
        };
        self.walk_tree(&gs, 0, -1, 0, &mut idx);

        let mut cap = 0u32;
        for node in 0..idx.visible.len() {
            idx.seq_offset.push(cap);
            let nb = self.buckets_for_visible(idx.visible[node]);
            for _ in 0..nb {
                idx.actions_by_index.push(idx.num_actions[node]);
            }
            cap += nb;
        }
        idx.capacity = cap as usize;
        idx
    }

    /// Depth-first enumeration: allocate a node for the decision at `gs`, then
    /// recurse over its capped legal actions.  Returns the node id, or `-1` if
    /// `gs` is terminal (so the parent records "no child here").
    fn walk_tree(&self, gs: &GameState, street_raises: u8, parent: i32, in_action: u8, idx: &mut Indexing) -> i32 {
        if gs.is_terminal() {
            return -1;
        }
        let acts = self.capped_legal(gs, street_raises);
        let id = idx.children.len();
        idx.children.push([-1; 8]);
        idx.visible.push(gs.board_cards_count() as u8);
        idx.to_act.push(gs.current_player() as u8);
        idx.num_actions.push(acts.len() as u8);
        idx.parent.push(parent);
        idx.in_action.push(in_action);
        for a in 0..acts.len() {
            let (old_street, old_bet) = (gs.street, gs.current_bet);
            let mut child = gs.clone();
            child.apply_action(acts[a]);
            let sr = Self::next_raises(street_raises, old_street, old_bet, &child);
            let child_id = self.walk_tree(&child, sr, id as i32, a as u8, idx);
            idx.children[id][a] = child_id;
        }
        id as i32
    }

    /// The current player's card bucket, in `0..buckets_for(visible)`.  Mirrors
    /// [`situation_bucket`](BlueprintHoldem::situation_bucket) under full
    /// coverage (the precondition of [`with_indexing`]); an out-of-set situation
    /// — which cannot occur with a full-coverage map — falls back to bucket `0`.
    fn dense_bucket(&self, hole: &[u8; 2], board: &[u8]) -> usize {
        let visible = board.len();
        if visible == 0 {
            return preflop_index(hole) as usize;
        }
        self.street_buckets[visible - 3]
            .as_ref()
            .expect("map present")
            .bucket(hole, board)
            .map(|b| b as usize)
            .unwrap_or(0)
    }

    /// Reconstruct the `HashMap` info key for a dense index, so an SoA-trained
    /// strategy exports to the **same** `HashMap<u64, _>` artifact as the
    /// `HashMap`-solver path (identical bytes — see
    /// [`info_key_for`](BlueprintHoldem::info_key_for)).
    pub fn info_key_at(&self, index: usize) -> u64 {
        let idx = self.indexing();
        // The node owning this index is the last whose block starts at/below it.
        let node = idx.seq_offset.partition_point(|&o| o as usize <= index) - 1;
        let bucket = (index - idx.seq_offset[node] as usize) as u64;

        // Rebuild the perfect-recall history from the parent chain.
        let mut history = Vec::new();
        let mut n = node as i32;
        while idx.parent[n as usize] >= 0 {
            history.push(idx.in_action[n as usize]);
            n = idx.parent[n as usize];
        }
        history.reverse();

        let mut h = Fnv1a::new();
        h.write(idx.to_act[node]);
        h.write(idx.visible[node]);
        h.write_all(&bucket.to_le_bytes());
        h.write(0xFF);
        h.write_all(&history);
        h.finish()
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

    /// The abstracted information key for the situation `(hole, board)` at the
    /// given street: a bucket id when an abstraction covers it, otherwise the
    /// suit-canonical key folded to `u64`.
    fn situation_bucket(&self, hole: &[u8; 2], board: &[u8]) -> u64 {
        let visible = board.len();
        if visible == 0 {
            // Pre-flop: the 169 suit-canonical starting-hand classes.
            return preflop_index(hole) as u64;
        }
        let street = visible - 3; // flop = 0, turn = 1, river = 2
        match self.street_buckets.get(street).and_then(Option::as_ref) {
            Some(map) => match map.bucket(hole, board) {
                Some(b) => b as u64,
                // Outside the built set: stay correct by not abstracting.
                None => self.raw_index(street, hole, board),
            },
            None => self.raw_index(street, hole, board),
        }
    }

    /// Unabstracted key for a post-flop street: the raw dense hand index (which
    /// is itself suit-isomorphic and collision-free).  Used when no bucket map
    /// covers the situation — the same role the suit-canonical key played before.
    fn raw_index(&self, street: usize, hole: &[u8; 2], board: &[u8]) -> u64 {
        let mut cards = [0u8; 7];
        cards[0] = hole[0];
        cards[1] = hole[1];
        cards[2..2 + board.len()].copy_from_slice(board);
        self.indexers[street].index(&cards[..2 + board.len()])
    }

    /// Fold the information-set key for the acting player at `gs` with the given
    /// perfect-recall `history` (action indices), streamed straight into FNV-1a
    /// so neither the clone-based nor the cursor-based path allocates a `Vec`.
    fn info_key_for(&self, gs: &GameState, history: &[u8]) -> u64 {
        let player = gs.current_player();
        let hole = gs.hole_cards[player];
        let visible = gs.board_cards_count();
        self.key_for_cards(player, hole, &gs.board[..visible], history)
    }

    /// The information key `player` would have holding `hole` at the public
    /// situation `(board, history)` — the shared kernel of [`Game::info_key`]
    /// and the play-time belief updates (which ask "what key — and hence what
    /// blueprint strategy — would the opponent have with *this* hand?").
    fn key_for_cards(&self, player: usize, mut hole: [u8; 2], board: &[u8], history: &[u8]) -> u64 {
        hole.sort_unstable();
        let bucket = self.situation_bucket(&hole, board);
        self.key_from_bucket(player, board.len(), bucket, history)
    }

    /// Fold an already-computed card `bucket` into the info key — the hashing
    /// kernel of [`key_for_cards`](Self::key_for_cards), public so bulk walkers
    /// (`evaluation::vector_br`) can hoist the per-hand bucket computation out
    /// of the per-node loop and still land on identical blueprint keys.
    pub fn key_from_bucket(&self, player: usize, visible: usize, bucket: u64, history: &[u8]) -> u64 {
        let mut h = Fnv1a::new();
        h.write(player as u8);
        h.write(visible as u8);
        h.write_all(&bucket.to_le_bytes());
        h.write(0xFF); // separator so bucket bytes / history can't blur
        h.write_all(history);
        h.finish()
    }

    // ------------------------------------------------------------------
    // Play-time API (`crate::play`): track a real hand through the abstract
    // game and read blueprint keys for arbitrary hypothetical holdings.
    // ------------------------------------------------------------------

    /// Construct a play node from concrete cards: both hole pairs plus the
    /// board known so far (`NO_CARD` for unrevealed cards), with no action
    /// history.  The entry point for play-time tracking of a real hand; advance
    /// it with [`Game::apply`] using indices into [`actions`](Self::actions).
    pub fn play_state(&self, holes: [[u8; 2]; 2], board: [u8; 5]) -> BlueprintState {
        let mut all = [[NO_CARD; 2]; MAX_PLAYERS];
        all[0] = holes[0];
        all[1] = holes[1];
        let gs =
            GameState::new(2, self.big_blind, self.small_blind, self.stacks, all, board, self.button);
        BlueprintState { gs: Some(gs), history: Vec::new(), street_raises: 0 }
    }

    /// The capped legal actions at a play node — the very list whose indices
    /// [`Game::apply`] takes and the info-key history records.
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

impl Game for BlueprintHoldem {
    type State = BlueprintState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> BlueprintState {
        BlueprintState { gs: None, history: Vec::new(), street_raises: 0 }
    }

    fn is_terminal(&self, state: &BlueprintState) -> bool {
        state.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, state: &BlueprintState) -> bool {
        state.gs.is_none()
    }

    fn is_chance_enumerable(&self, _state: &BlueprintState) -> bool {
        false
    }

    fn utility(&self, state: &BlueprintState, player: usize) -> f64 {
        let gs = state.gs.as_ref().expect("utility at a play node");
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    /// Unsupported: the deal space is not enumerable.  The solver reaches
    /// children through [`sample_chance`](Game::sample_chance) instead.
    fn chance_outcomes(&self, _state: &BlueprintState) -> Vec<(BlueprintState, f64)> {
        unimplemented!("BlueprintHoldem chance is not enumerable; use sample_chance")
    }

    fn sample_chance(
        &self,
        _state: &BlueprintState,
        next_unit: impl FnMut() -> f64,
    ) -> BlueprintState {
        BlueprintState { gs: Some(self.deal(next_unit)), history: Vec::new(), street_raises: 0 }
    }

    fn current_player(&self, state: &BlueprintState) -> usize {
        state.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn num_actions(&self, state: &BlueprintState) -> usize {
        let gs = state.gs.as_ref().expect("num_actions at a play node");
        self.capped_legal(gs, state.street_raises).len()
    }

    fn apply(&self, state: &BlueprintState, action: usize) -> BlueprintState {
        let gs = state.gs.as_ref().expect("apply at a play node");
        let act = self.capped_legal(gs, state.street_raises)[action];
        let (old_street, old_bet) = (gs.street, gs.current_bet);
        let mut next_gs = gs.clone();
        next_gs.apply_action(act);
        let street_raises = Self::next_raises(state.street_raises, old_street, old_bet, &next_gs);
        let mut history = state.history.clone();
        history.push(action as u8);
        BlueprintState { gs: Some(next_gs), history, street_raises }
    }

    fn info_key(&self, state: &BlueprintState) -> u64 {
        let gs = state.gs.as_ref().expect("info_key at a play node");
        self.info_key_for(gs, &state.history)
    }
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

impl super::CursorGame for BlueprintHoldem {
    type Cursor = BlueprintCursor;
    type Action = Action;
    type Actions = ActionList;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> BlueprintCursor {
        BlueprintCursor {
            gs: None,
            history: [0; MAX_DEPTH],
            depth: 0,
            street_raises: 0,
            raises_at: [0; MAX_DEPTH],
        }
    }

    fn is_terminal(&self, c: &BlueprintCursor) -> bool {
        c.gs.as_ref().is_some_and(|g| g.is_terminal())
    }

    fn is_chance(&self, c: &BlueprintCursor) -> bool {
        c.gs.is_none()
    }

    fn utility(&self, c: &BlueprintCursor, player: usize) -> f64 {
        let gs = c.gs.as_ref().expect("utility at a play node");
        gs.terminal_payoffs()[player] as f64 / self.big_blind as f64
    }

    fn current_player(&self, c: &BlueprintCursor) -> usize {
        c.gs.as_ref().expect("current_player at a play node").current_player()
    }

    fn legal(&self, c: &BlueprintCursor) -> ActionList {
        self.capped_legal(c.gs.as_ref().expect("legal at a play node"), c.street_raises)
    }

    fn info_key(&self, c: &BlueprintCursor) -> u64 {
        let gs = c.gs.as_ref().expect("info_key at a play node");
        self.info_key_for(gs, &c.history[..c.depth])
    }

    fn apply(&self, c: &mut BlueprintCursor, a: usize, action: Action) {
        let gs = c.gs.as_mut().expect("apply at a play node");
        let (old_street, old_bet) = (gs.street, gs.current_bet);
        gs.apply_action(action);
        c.raises_at[c.depth] = c.street_raises;
        c.street_raises = Self::next_raises(c.street_raises, old_street, old_bet, gs);
        c.history[c.depth] = a as u8;
        c.depth += 1;
    }

    fn undo(&self, c: &mut BlueprintCursor) {
        c.depth -= 1;
        c.street_raises = c.raises_at[c.depth];
        c.gs.as_mut().expect("undo at a play node").undo_action();
    }

    fn sample_chance(&self, c: &mut BlueprintCursor, next_unit: impl FnMut() -> f64) {
        c.gs = Some(self.deal(next_unit));
        c.depth = 0;
        c.street_raises = 0;
    }

    fn undo_chance(&self, c: &mut BlueprintCursor) {
        c.gs = None;
        c.depth = 0;
        c.street_raises = 0;
    }
}

impl super::IndexedGame for BlueprintHoldem {
    fn info_set_capacity(&self) -> usize {
        self.indexing().capacity
    }

    /// `sequence_offset + card_bucket`.  The sequence is found by walking the
    /// enumerated betting tree along the cursor's inline history (O(depth), no
    /// allocation); the bucket is the acting player's current-street card bucket.
    fn info_set_index(&self, c: &BlueprintCursor) -> usize {
        let idx = self.indexing();
        let gs = c.gs.as_ref().expect("info_set_index at a play node");

        let mut node = 0usize;
        for &a in &c.history[..c.depth] {
            node = idx.children[node][a as usize] as usize;
        }
        debug_assert_eq!(idx.to_act[node] as usize, gs.current_player(), "tree node player matches");

        let player = gs.current_player();
        let mut hole = gs.hole_cards[player];
        hole.sort_unstable();
        let visible = gs.board_cards_count();
        debug_assert_eq!(idx.visible[node] as usize, visible, "tree node street matches");
        let bucket = self.dense_bucket(&hole, &gs.board[..visible]);

        idx.seq_offset[node] as usize + bucket
    }

    fn actions_at(&self, index: usize) -> usize {
        self.indexing().actions_by_index[index] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::{make_card, rank_of, suit_of};

    use crate::solver::cfr::Variant;
    use crate::solver::dcfr::Discount;
    use crate::solver::mccfr::Mccfr;

    /// Suit-rotate a card by `+1 (mod 4)` — for asserting suit isomorphism.
    fn rotate_suit(c: u8) -> u8 {
        make_card(rank_of(c), (suit_of(c) + 1) % 4)
    }

    /// A tiny deterministic unit source for driving `sample_chance` directly.
    fn unit_stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed | 1;
        move || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (v >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn sampled_deal_uses_nine_distinct_real_cards() {
        let game = BlueprintHoldem::new(100, 2, 1, 0);
        let root = game.root();
        assert!(game.is_chance(&root));
        assert!(!game.is_chance_enumerable(&root));

        for seed in 0..200u64 {
            let st = game.sample_chance(&root, unit_stream(seed));
            let gs = st.gs.as_ref().unwrap();
            let mut cards = Vec::new();
            cards.extend_from_slice(&gs.hole_cards[0]);
            cards.extend_from_slice(&gs.hole_cards[1]);
            cards.extend_from_slice(&gs.board);
            assert_eq!(cards.len(), DEAL_CARDS);
            assert!(cards.iter().all(|&c| c < 52), "every dealt card is a real card");
            cards.sort_unstable();
            cards.dedup();
            assert_eq!(cards.len(), DEAL_CARDS, "no card is dealt twice (seed {seed})");
        }
    }

    #[test]
    fn preflop_key_collapses_suit_isomorphic_hands() {
        // Two pre-flop situations that differ only by a global suit rotation must
        // share an information key (same 169-class), and they must differ from a
        // genuinely different starting hand.
        let game = BlueprintHoldem::new(100, 2, 1, 0);
        let mk = |holes: [[u8; 2]; 2]| {
            let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
            h[0] = holes[0];
            h[1] = holes[1];
            let board = [NO_CARD; 5];
            let gs = GameState::new(2, 2, 1, game.stacks, h, board, 0);
            BlueprintState { gs: Some(gs), history: Vec::new(), street_raises: 0 }
        };
        // A♠K♠ vs 7♦7♣  →  rotate every suit  →  A♥K♥ vs 7♣7♠.
        let base = mk([[make_card(12, 0), make_card(11, 0)], [make_card(5, 1), make_card(5, 2)]]);
        let rot = mk([
            [rotate_suit(make_card(12, 0)), rotate_suit(make_card(11, 0))],
            [rotate_suit(make_card(5, 1)), rotate_suit(make_card(5, 2))],
        ]);
        // The acting pre-flop player is the same in both; keys must match.
        assert_eq!(game.info_key(&base), game.info_key(&rot));

        // A different starting hand (Q♠J♠) keys differently.
        let other = mk([[make_card(10, 0), make_card(9, 0)], [make_card(5, 1), make_card(5, 2)]]);
        assert_ne!(game.info_key(&base), game.info_key(&other));
    }

    #[test]
    fn mccfr_runs_over_sampled_blueprint() {
        // The keystone smoke test: external sampling drives the real engine
        // through sampled deals + bucketed keys, completes, and produces valid
        // probability distributions at every discovered info set.
        let game = BlueprintHoldem::new(40, 2, 1, 0);
        let mut solver = Mccfr::new(game, Variant::Dcfr(Discount::RECOMMENDED));
        solver.train(2_000);
        assert!(solver.num_info_sets() > 0, "should discover info sets");
        for (_key, probs) in solver.average_strategy() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "strategy must be a distribution, got {sum}");
            assert!(probs.iter().all(|&p| p >= 0.0));
        }
    }

    #[test]
    fn baseline_mccfr_runs_over_sampled_blueprint() {
        // The VR-MCCFR chance baseline must gracefully no-op on a non-enumerable
        // chance node (no outcome list to index) yet still train cleanly.
        let game = BlueprintHoldem::new(40, 2, 1, 0);
        let mut solver = Mccfr::new(game, Variant::Vanilla).with_baseline();
        solver.train(1_000);
        assert!(solver.num_info_sets() > 0);
    }

    #[test]
    fn is_deterministic_for_fixed_seed() {
        let run = || {
            let game = BlueprintHoldem::new(40, 2, 1, 0);
            let mut s = Mccfr::with_seed(game, Variant::Vanilla, 99);
            s.train(1_000);
            s.num_info_sets()
        };
        assert_eq!(run(), run(), "same seed must visit the same info sets");
    }

    #[test]
    fn raise_cap_removes_sized_raises_but_never_the_all_in() {
        let game = BlueprintHoldem::new(200, 2, 1, 0).with_raise_cap(1);
        // Deep-stacked heads-up preflop: SB to act faces a raise/all-in menu.
        let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
        h[0] = [make_card(12, 0), make_card(11, 0)];
        h[1] = [make_card(5, 1), make_card(5, 2)];
        let gs = GameState::new(2, 2, 1, game.stacks, h, [NO_CARD; 5], 0);

        // Below the cap (0 raises so far) the opening raise is still offered.
        let under = game.capped_legal(&gs, 0);
        assert!(
            under.iter().any(|a| matches!(a, Action::Raise(_))),
            "opening raise must be legal below the cap, got {under:?}"
        );

        // At the cap, sized raises are gone but all-in remains: the abstraction
        // must stay closed under aggression, so a raise war always has a
        // terminating action for the tracker to map an opponent's shove onto.
        let at = game.capped_legal(&gs, 1);
        assert!(
            at.iter().all(|a| !matches!(a, Action::Raise(_))),
            "no sized reraise at the cap, got {at:?}"
        );
        assert!(
            at.iter().any(|a| matches!(a, Action::AllIn)),
            "all-in must survive the cap, got {at:?}"
        );
        assert!(
            at.iter().any(|a| matches!(a, Action::Fold | Action::Call | Action::Check)),
            "a passive action must remain, got {at:?}"
        );

        // The uncapped default never filters, however many raises have happened.
        let uncapped = BlueprintHoldem::new(200, 2, 1, 0);
        assert!(uncapped.capped_legal(&gs, 9).iter().any(|a| matches!(a, Action::Raise(_))));
    }

    /// The property the fix exists for: from any node, however many raises have
    /// already gone in, the acting player can still put the rest of the stack
    /// in.  Nothing an opponent does can leave the abstraction without an
    /// aggressive action to translate their bet onto.
    #[test]
    fn aggression_always_has_a_landing_spot_at_the_cap() {
        let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
        let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
        h[0] = [make_card(12, 0), make_card(11, 0)];
        h[1] = [make_card(5, 1), make_card(5, 2)];
        let gs = GameState::new(2, 2, 1, game.stacks, h, [NO_CARD; 5], 0);

        for raises in 0..8u8 {
            let acts = game.capped_legal(&gs, raises);
            assert!(
                acts.iter().any(|a| matches!(a, Action::Raise(_) | Action::AllIn)),
                "no aggressive action after {raises} raises: {acts:?}"
            );
        }
    }

    #[test]
    fn capped_clone_and_cursor_paths_agree() {
        // The cursor path maintains `street_raises` in place via apply/undo; it
        // must visit exactly the same (capped) info sets as the clone path.
        use crate::games::CursorGame;
        let mk = || BlueprintHoldem::new(40, 2, 1, 0).with_raise_cap(1);
        let _ = CursorGame::root(&mk()); // ensure the capped game is a CursorGame

        let mut clone_path = Mccfr::with_seed(mk(), Variant::Dcfr(Discount::RECOMMENDED), 5);
        clone_path.train(500);
        let mut cursor_path = Mccfr::with_seed(mk(), Variant::Dcfr(Discount::RECOMMENDED), 5);
        cursor_path.train_fast(500);
        assert_eq!(
            clone_path.num_info_sets(),
            cursor_path.num_info_sets(),
            "capped legal lists must match between the clone and cursor paths"
        );
    }

    #[test]
    fn indexed_preflop_only_partition_and_key_round_trip() {
        use crate::games::{CursorGame, IndexedGame};
        // stack == big blind: the BB is all-in from its blind, so the SB faces a
        // single fold/all-in decision and no post-flop node is ever created.  The
        // placeholder maps are therefore never queried — this keeps the test O(1)
        // while exercising the full IndexedGame plumbing (capacity, index,
        // actions_at, and the info_key_at export inverse).
        let game = BlueprintHoldem::new(2, 2, 1, 0)
            .with_raise_cap(1)
            .with_street_bucket(0, BucketMap::placeholder(&[2, 3], 50))
            .with_street_bucket(1, BucketMap::placeholder(&[2, 4], 50))
            .with_street_bucket(2, BucketMap::placeholder(&[2, 5], 50))
            .with_indexing();

        let cap = game.info_set_capacity();
        assert!(cap >= 169 && cap.is_multiple_of(169), "preflop-only capacity is a multiple of 169, got {cap}");

        // The dense index and the HashMap info key must induce the SAME partition,
        // and info_key_at must invert the index back to that key.
        let mut by_key: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let mut by_idx: std::collections::HashMap<usize, u64> = std::collections::HashMap::new();
        for seed in 0..500u64 {
            let mut c = CursorGame::root(&game);
            CursorGame::sample_chance(&game, &mut c, unit_stream(seed));
            let key = CursorGame::info_key(&game, &c);
            let idx = game.info_set_index(&c);
            assert!(idx < cap, "index in range");
            assert_eq!(game.actions_at(idx), 2, "fold/all-in menu");
            assert_eq!(game.info_key_at(idx), key, "info_key_at inverts the dense index");
            assert_eq!(*by_key.entry(key).or_insert(idx), idx, "key -> one index");
            assert_eq!(*by_idx.entry(idx).or_insert(key), key, "index -> one key");
        }
        assert!(by_key.len() > 100, "should see many distinct starting-hand classes");
    }

    /// Full post-flop coverage: builds the turn/river full-coverage maps (~280 MB)
    /// so the dense index has no out-of-set situation.  Confirms the dense index
    /// partitions information sets identically to the `HashMap` key on every
    /// street, that `info_key_at` inverts it, and that the SoA solver trains over
    /// the indexed full game to valid distributions.
    ///   cargo test -p poker-ai --release -- --ignored indexed_blueprint_postflop_and_soa
    /// Throughput comparison of the three SoA training paths on a realistic
    /// indexed blueprint tree (20 bb, cap-2) — the parallel-scaling
    /// deliverable.  Prints nodes/s per configuration; the assertions are a
    /// loose sanity ordering so a busy machine cannot flake the test.
    ///   cargo test -p poker-ai --release -- --ignored --nocapture atomic_scaling
    #[test]
    #[ignore]
    fn atomic_scaling_benchmark() {
        use crate::solver::cfr::Variant;
        use crate::solver::dcfr::Discount;
        use crate::solver::mccfr::SoaMccfr;
        use std::time::Instant;

        let mk = || {
            BlueprintHoldem::new(40, 2, 1, 0)
                .with_raise_cap(2)
                .with_street_bucket(0, BucketMap::full_coverage_mod(&[2, 3], 40))
                .with_street_bucket(1, BucketMap::full_coverage_mod(&[2, 4], 40))
                .with_street_bucket(2, BucketMap::full_coverage_mod(&[2, 5], 40))
                .with_indexing()
        };
        let iters = 200_000u64;
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
            let nps = bench(&name, &mut |s| s.train_atomic(iters, threads));
            atomic_best = atomic_best.max(nps);
        }
        assert!(atomic_best > serial, "atomic best {atomic_best} should beat serial {serial}");
        assert!(
            atomic_best > parallel,
            "atomic best {atomic_best} should beat batched parallel {parallel}"
        );
    }

    #[test]
    #[ignore]
    fn indexed_blueprint_postflop_and_soa() {
        use crate::games::{CursorGame, IndexedGame};
        use crate::solver::mccfr::SoaMccfr;

        let mk = || {
            BlueprintHoldem::new(12, 2, 1, 0) // 6bb: check lines reach every street under cap 1
                .with_raise_cap(1)
                .with_street_bucket(0, BucketMap::full_coverage_mod(&[2, 3], 40))
                .with_street_bucket(1, BucketMap::full_coverage_mod(&[2, 4], 40))
                .with_street_bucket(2, BucketMap::full_coverage_mod(&[2, 5], 40))
                .with_indexing()
        };
        let game = mk();
        let cap = game.info_set_capacity();
        assert!(cap > 0, "non-empty index");

        // Roll out full hands, checking the partition + round-trip at every
        // decision node on every street.
        let mut by_key: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let mut rng = 0x00C0_FFEEu64;
        let mut next = || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..3000 {
            let mut c = CursorGame::root(&game);
            CursorGame::sample_chance(&game, &mut c, &mut next);
            while !CursorGame::is_terminal(&game, &c) {
                let key = CursorGame::info_key(&game, &c);
                let idx = game.info_set_index(&c);
                assert!(idx < cap, "index in range");
                assert_eq!(game.info_key_at(idx), key, "info_key_at inverts the dense index");
                assert_eq!(*by_key.entry(key).or_insert(idx), idx, "key -> one index (same partition)");
                let acts = CursorGame::legal(&game, &c);
                let n = acts.as_ref().len();
                let a = ((next() * n as f64) as usize).min(n - 1);
                CursorGame::apply(&game, &mut c, a, acts.as_ref()[a]);
            }
        }
        assert!(by_key.len() > 200, "should exercise many post-flop info sets, got {}", by_key.len());

        // The SoA solver trains over the indexed full game and yields valid
        // probability distributions at every visited info set.
        let mut soa: SoaMccfr<BlueprintHoldem> =
            SoaMccfr::with_seed(mk(), Variant::Dcfr(Discount::RECOMMENDED), 1).with_baseline();
        soa.train(20_000);
        let mut visited = 0;
        for i in 0..soa.capacity() {
            if soa.is_visited(i) {
                visited += 1;
                let p = soa.average_strategy_at(i);
                let sum: f64 = p.iter().sum();
                assert!((sum - 1.0).abs() < 1e-9, "distribution at {i} sums to {sum}");
                assert!(p.iter().all(|&x| x >= 0.0));
            }
        }
        assert!(visited > 100, "SoA training should visit many info sets, got {visited}");
    }
}
