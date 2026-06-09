# Building a High-Performance Poker Core Engine in Rust

## Design Decisions, Tradeoffs, and Performance

> **~20 minute read** · Intermediate to Advanced

---

## Table of Contents

1. [Why Poker AI Is Hard](#1-why-poker-ai-is-hard)
2. [The Role of the Core Engine](#2-the-role-of-the-core-engine)
3. [Card Encoding: Small Numbers, Big Wins](#3-card-encoding-small-numbers-big-wins)
4. [Hand Strength Evaluation](#4-hand-strength-evaluation)
   - [The `make_hand` Encoding](#the-make_hand-encoding)
   - [evaluate_5: One Pass, No Sort, No Heap](#evaluate_5-one-pass-no-sort-no-heap)
   - [Straight Detection and the Wheel, via a Pattern Table](#straight-detection-and-the-wheel-via-a-pattern-table)
   - [Extending to 6 and 7 Cards](#extending-to-6-and-7-cards)
   - [The LUT Evaluator: The Hot-Path Fast Path](#the-lut-evaluator-the-hot-path-fast-path)
5. [GameState: Packing Six Players into Tight Memory](#5-gamestate-packing-six-players-into-tight-memory)
   - [Bitmasks for Player Status](#bitmasks-for-player-status)
   - [Pre-Dealt Board Cards (Public Chance Sampling)](#pre-dealt-board-cards-public-chance-sampling)
   - [Posting Blinds at Construction](#posting-blinds-at-construction)
6. [apply_action and undo_action: Zero Allocation in the Hot Path](#6-apply_action-and-undo_action-zero-allocation-in-the-hot-path)
   - [Delta Undo: Recording Only What Changed](#delta-undo-recording-only-what-changed)
   - [Debug-Only Invariants](#debug-only-invariants)
   - [Raise Semantics: Total Level, Not Increment](#raise-semantics-total-level-not-increment)
   - [AllIn vs Call: A Canonical Distinction](#allin-vs-call-a-canonical-distinction)
   - [Side-Pot Handling at Showdown](#side-pot-handling-at-showdown)
7. [The Undo Stack: Pre-Allocated Tree Traversal](#7-the-undo-stack-pre-allocated-tree-traversal)
8. [Action Abstraction: Taming the Infinite Action Space](#8-action-abstraction-taming-the-infinite-action-space)
   - [Why Abstraction Is Necessary](#why-abstraction-is-necessary)
   - [Pot-Fraction Sizing by Street — in Integer Arithmetic](#pot-fraction-sizing-by-street--in-integer-arithmetic)
   - [abstract_raise_amounts: Stack-Allocated Return](#abstract_raise_amounts-stack-allocated-return)
   - [Snapping Raw Bets to the Nearest Abstract Size](#snapping-raw-bets-to-the-nearest-abstract-size)
9. [legal_actions: Combining Rules and Abstraction](#9-legal_actions-combining-rules-and-abstraction)
10. [Performance: Benchmarking Kuhn Poker Tree Traversal](#10-performance-benchmarking-kuhn-poker-tree-traversal)
    - [Why Kuhn Poker?](#why-kuhn-poker)
    - [Recursive vs Explicit-Stack Traversal](#recursive-vs-explicit-stack-traversal)
    - [Benchmark Results and Interpretation](#benchmark-results-and-interpretation)
11. [Key Design Tradeoffs](#11-key-design-tradeoffs)
    - [Two-Tier Evaluation: Algorithmic Reference + Compact LUT](#two-tier-evaluation-algorithmic-reference--compact-lut)
    - [Pre-Dealt vs Lazily-Dealt Board](#pre-dealt-vs-lazily-dealt-board)
    - [Delta Undo vs Full-Snapshot Undo](#delta-undo-vs-full-snapshot-undo)
    - [Integer vs Floating-Point Bet Sizing](#integer-vs-floating-point-bet-sizing)
    - [Bitmask State vs Enum Arrays](#bitmask-state-vs-enum-arrays)
    - [Abstract Action Space vs Full NLHE Tree](#abstract-action-space-vs-full-nlhe-tree)
12. [Lessons for Other Performance-Critical Projects](#12-lessons-for-other-performance-critical-projects)
13. [What Comes Next](#13-what-comes-next)

---

## 1. Why Poker AI Is Hard

Chess and Go have well-defined search spaces: both players see the full board at all times. Poker is different. Each player holds **private information** — hole cards — that their opponents cannot observe. This makes poker a game of **imperfect information**, and solving it requires fundamentally different algorithms.

The breakthrough came when Carnegie Mellon's Libratus defeated world-class human professionals in 2017, followed by Pluribus achieving superhuman 6-player performance in 2019. Unlike prior systems that relied on hand-crafted heuristics, these bots used **Counterfactual Regret Minimization (CFR)** — an iterative self-play algorithm that converges to a Nash equilibrium strategy with no domain knowledge beyond the game rules.

The catch: running CFR at scale requires traversing an enormous game tree billions of times. The full 6-player NLHE tree has on the order of 10¹⁶⁰ nodes when you account for all possible card combinations and action sequences. Even Pluribus — which ran on a 64-core server for eight days — needed aggressive abstraction to make this tractable.

This means the **game engine** sits at the very bottom of the performance-critical path. Every nanosecond saved in `apply_action` translates directly into more CFR iterations, and more iterations translate into a stronger strategy. A poorly engineered engine is a permanent bottleneck; a well-engineered one is invisible infrastructure that the solver just takes for granted.

This article is a deep dive into `poker-core`: a purpose-built Rust crate that aims to be exactly that kind of invisible, fast infrastructure.

---

## 2. The Role of the Core Engine

`poker-core` has one job: **represent and advance a 6-max No-Limit Texas Hold'em game state as fast as possible**, with perfect correctness. Its public API is tiny ([`lib.rs`](../crates/poker-core/src/lib.rs)):

| Function / Type | Purpose |
|----------------|---------|
| `GameState::new(...)` | Construct a new hand, post blinds |
| `apply_action(action)` | Advance state, push undo record |
| `undo_action()` | Restore previous state |
| `is_terminal()` | Check if the hand is over |
| `terminal_payoffs()` | Return chip deltas at showdown |
| `legal_actions(state)` | Enumerate valid abstract actions (returns a stack-allocated `ActionList`) |
| `evaluate_5/6/7(cards)` | Algorithmic hand evaluation (the reference) |
| `evaluate_5/6/7_lut(cards)` | Lookup-table hand evaluation (the hot-path fast path) |

Everything else — the CFR solver, information abstraction, belief tracking — lives in `poker-ai` and calls into these primitives. The contract is simple: **the engine never allocates on the heap in the hot path**, and every operation is O(1) or O(small constant).

There are two evaluators on purpose: an **algorithmic** one that needs no precomputed tables and serves as the correctness oracle, and a **lookup-table (LUT)** one that is faster and is what `GameState` actually calls at showdown. Section 4 covers both.

---

## 3. Card Encoding: Small Numbers, Big Wins

The most fundamental design decision is how to represent a playing card. There are 52 cards; they need to fit in as little memory as possible and support fast rank/suit extraction.

The chosen encoding packs both rank and suit into a single byte:

```
card = rank * 4 + suit

rank: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=T (Ten), 9=J, 10=Q, 11=K, 12=A
suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
```

This gives values in the range `[0, 51]`, fitting in a `u8`. The three helper functions are each one instruction ([`evaluator.rs`](../crates/poker-core/src/evaluator.rs)):

```rust
#[inline]
pub fn rank_of(card: u8) -> u8 { card >> 2 }      // bits 7–2

#[inline]
pub fn suit_of(card: u8) -> u8 { card & 3 }        // bits 1–0

#[inline]
pub fn make_card(rank: u8, suit: u8) -> u8 { (rank << 2) | suit }
```

The sentinel value `NO_CARD = 0xFF` marks board slots not yet revealed (pre-flop, turn, river). Since valid cards are in `[0, 51]`, `0xFF` cannot collide.

**Why not `(suit * 13) + rank`?** The rank-major layout means `rank_of` is a single right-shift and consecutive ranks produce consecutive byte values. More importantly, both evaluators key off the **13-bit rank bitmask** `rank_bits |= 1 << rank_of(c)`, and a rank-major encoding makes that mask trivial to build and to test against straight patterns.

**Why not a `struct Card { rank: u8, suit: u8 }`?** A struct would consume 2 bytes (or more due to alignment) per card, and array operations would touch more cache lines. For the evaluator, which reads 5–7 cards in a tight loop, the `u8` packing keeps an entire hand in a few cache lines.

---

## 4. Hand Strength Evaluation

### The `make_hand` Encoding

Hand strength is returned as a `u32` where **higher is always better**. The encoding packs category and tiebreaker ranks into a single comparable integer:

```
bits 23–20: hand category (0–8)
bits 19–16: first tiebreaker rank (4 bits)
bits 15–12: second tiebreaker rank
bits 11–8:  third tiebreaker rank
bits  7–4:  fourth tiebreaker rank
bits  3–0:  fifth tiebreaker rank
```

Category codes:

| Code | Hand |
|------|------|
| 8 | Straight Flush |
| 7 | Four of a Kind |
| 6 | Full House |
| 5 | Flush |
| 4 | Straight |
| 3 | Three of a Kind |
| 2 | Two Pair |
| 1 | One Pair |
| 0 | High Card |

The packing function is:

```rust
#[inline]
pub(crate) fn make_hand(cat: u8, r1: u8, r2: u8, r3: u8, r4: u8, r5: u8) -> u32 {
    ((cat as u32) << 20)
        | ((r1 as u32) << 16)
        | ((r2 as u32) << 12)
        | ((r3 as u32) << 8)
        | ((r4 as u32) << 4)
        | (r5 as u32)
}
```

The beauty of this design: **comparing two hands is just `a > b`** — no special comparison logic needed. A full house (category 6) always beats a flush (category 5) because `6 << 20 > 5 << 20`. Within the same category, tiebreakers are embedded in decreasing significance, so the highest-ranking tiebreaker wins automatically. This is the standard "score word" technique used by fast evaluators.

### evaluate_5: One Pass, No Sort, No Heap

The reference evaluator classifies a 5-card hand in a **single pass** — no sort. It builds a 13-element rank-frequency table and a 13-bit rank bitmask while simultaneously checking the flush condition:

```rust
pub fn evaluate_5(cards: &[u8; 5]) -> u32 {
    let mut freq = [0u8; 13];
    let mut rank_bits = 0u16;
    let first_suit = suit_of(cards[0]);
    let mut is_flush = true;

    for &c in cards {
        let r = rank_of(c);
        freq[r as usize] += 1;
        rank_bits |= 1u16 << r;
        if suit_of(c) != first_suit { is_flush = false; }
    }
    // ... scan freq high→low for quads/trips/pairs, then classify ...
}
```

All intermediate data structures — `freq` (13 bytes), `rank_bits` (2 bytes), and a handful of scalars — are fixed-size and live in registers or one cache line. The function has **zero heap allocations**. Quads / trips / pairs are found by scanning `freq` from rank 12 down to 0, which naturally yields tiebreakers in descending significance.

The previous implementation sorted the five ranks. Sorting was removed: the frequency table gives the same information in a single linear pass and avoids the branch-heavy comparison network entirely.

### Straight Detection and the Wheel, via a Pattern Table

The **wheel** (A-2-3-4-5) is the classic edge case in poker evaluation: the Ace plays low, giving a five-high straight that a naive "top minus bottom equals four" test misses. Rather than special-case it, the evaluator tests the rank bitmask against a precomputed table of all ten 5-card straight patterns, **with the wheel as an explicit entry**:

```rust
const STRAIGHT_TABLE: [(u16, u8); 10] = [
    (0b1_1111_0000_0000, 12), // A-K-Q-J-T
    (0b0_1111_1000_0000, 11), // K-Q-J-T-9
    // ...
    (0b0_0000_0001_1111,  4), // 6-5-4-3-2
    (0b1_0000_0000_1111,  3), // A-5-4-3-2 (wheel, five-high → top rank 3)
];

#[inline]
pub(crate) fn find_best_straight(rank_bits: u16) -> (bool, u8) {
    for (mask, top) in STRAIGHT_TABLE {
        if rank_bits & mask == mask { return (true, top); }
    }
    (false, 0)
}
```

The wheel's mask sets the Ace bit (position 12) and the 2–5 bits (positions 0–3), with a "top rank" of 3 (Five). So `make_hand(4, 3, ...)` is produced, which correctly ranks below a Six-high straight (`make_hand(4, 4, ...)`). Because patterns are listed highest-first, the first match is always the best straight — and the same routine works for 6- and 7-card masks (where more than five bits are set), since it simply finds the highest pattern fully contained in the mask.

### Extending to 6 and 7 Cards

Texas Hold'em players choose the best 5 cards from 7 (2 hole + 5 board). The earlier version enumerated all C(7,5) = 21 five-card subsets and evaluated each. That has been replaced: `evaluate_6` and `evaluate_7` build the rank-frequency table, a **per-suit rank bitmask**, and a per-suit count in **one pass**, then classify directly:

- **Flush / straight-flush**: any suit with ≥ 5 cards has its rank bitmask checked against `find_best_straight` (catching straight flushes, including the wheel) and against the flush ranking otherwise.
- **Non-flush hands**: delegated to a single shared routine, `best_non_flush_rank(freq, rank_bits)`, used by *both* the algorithmic and LUT evaluators so the quads/full-house/trips/two-pair/pair/high-card logic lives in exactly one place.

This single-pass approach is dramatically faster than 21 subset evaluations and, crucially, shares its non-flush classifier with the LUT path — eliminating a whole class of "the two evaluators disagree" bugs.

### The LUT Evaluator: The Hot-Path Fast Path

`poker-core` ships a second evaluator in [`lut_eval.rs`](../crates/poker-core/src/lut_eval.rs) — and it is the one `GameState` actually calls at showdown. It is a compact, Cactus-Kev-style lookup evaluator built from two tables generated at **compile time** by [`build.rs`](../crates/poker-core/build.rs) and embedded in the binary:

- **`FLUSH_LUT: [u32; 8192]`** — indexed by a 13-bit rank bitmask. Entries whose popcount is 5 hold the rank of the corresponding flush / straight flush; flush classification becomes a single array lookup.
- **`NOFLUSH_LUT: [(u32, u32); 16384]`** — an open-addressed hash table keyed by the **product of the five rank primes** (`2, 3, 5, 7, …, 41`). Non-flush classification is a prime-product multiply plus 1–2 probes.

These tables are tiny — on the order of tens of kilobytes, not the hundred-plus megabytes of a full 7-card lookup evaluator — so they comfortably coexist in cache with the CFR regret tables rather than evicting them. Compile-time assertions verify the table sizes match the constants in `lut_eval.rs`, and an `#[ignore]`d test cross-checks `evaluate_5_lut` against `evaluate_5` on **all C(52,5) = 2,598,960** hands. `evaluate_7_lut` reduces 6+ suited cards to their top five ranks and checks straight-flushes on the full suited mask (so wheel straight-flushes with six or seven suited cards are handled correctly).

The two evaluators give bit-identical results; the algorithmic one is the oracle, the LUT one is the speed.

---

## 5. GameState: Packing Six Players into Tight Memory

`GameState` ([`state.rs`](../crates/poker-core/src/state.rs)) holds all mutable state for one hand of 6-max NLHE:

```rust
pub struct GameState {
    pub stacks:          [u32; MAX_PLAYERS],   // chip stacks
    pub street_bets:     [u32; MAX_PLAYERS],   // chips committed this street
    pub total_committed: [u32; MAX_PLAYERS],   // chips committed entire hand
    pub board:           [u8;  5],             // all 5 community cards (pre-dealt)
    pub hole_cards:      [[u8; 2]; MAX_PLAYERS],
    pub street:          u8,                   // 0=preflop,1=flop,2=turn,3=river,4=terminal
    pub to_act:          u8,
    pub num_players:     u8,
    pub button:          u8,
    pub big_blind:       u32,
    pub pot:             u32,                   // cached Σ total_committed (O(1) pot())
    pub current_bet:     u32,                   // highest street_bet (amount to call)
    pub min_raise:       u32,                   // minimum raise increment
    pub folded:          u8,                    // bitmask: bit i set if player i folded
    pub allin:           u8,                    // bitmask: bit i set if player i all-in
    pub last_aggressor:  u8,                    // last bettor/raiser (0xFF = none)
    pub players_to_act:  u8,                    // active players still to act this round
    pub undo:            UndoStack,             // pre-allocated undo history
}
```

`MAX_PLAYERS = 6`. The inline struct is on the order of ~150 bytes; the only heap it owns is the undo stack's pre-allocated buffer (Section 7). The `pot` field is maintained incrementally on every action so `pot()` is O(1) — there is never an O(n) sum over `total_committed` in the hot path.

### Bitmasks for Player Status

`folded` and `allin` are both `u8` bitmasks: bit `i` corresponds to player `i`. Status checks and updates are single instructions, and counts use a hardware popcount over a player-width mask:

```rust
self.folded |= 1 << p;                                  // mark p folded

#[inline]
pub fn count_active(&self) -> u8 {                       // non-folded AND non-all-in
    let player_mask = (1u8 << self.num_players) - 1;
    (player_mask & !(self.folded | self.allin)).count_ones() as u8
}
```

The alternative — a `[PlayerStatus; 6]` enum array — would use 6+ bytes instead of 2 and turn single bitwise ops into branchy comparisons. For state copied into many undo records over a traversal, those bytes and branches add up.

### Pre-Dealt Board Cards (Public Chance Sampling)

All five board cards are passed at construction time, even though only 3 are visible on the flop, 4 on the turn, and 5 on the river. Unrevealed cards are hidden by `board_cards_count()`:

```rust
pub fn board_cards_count(&self) -> usize {
    match self.street {
        0 => 0,  // preflop
        1 => 3,  // flop
        2 => 4,  // turn
        _ => 5,  // river or terminal
    }
}
```

This is designed for **Public Chance Sampling (PCS)**: fix the public board at the root of a subtree and traverse all player action paths beneath it. The alternative — lazily sampling each community card as streets advance — would push board-sampling logic into `apply_action` and complicate the traversal loop. With pre-dealt boards, the engine is stateless with respect to card dealing; the CFR layer owns sampling entirely.

### Posting Blinds at Construction

`GameState::new(num_players, big_blind, small_blind, stacks, hole_cards, board, button)` automatically posts the blinds, so the state is valid immediately after construction. The position logic handles the heads-up special case, where the button **is** the small blind and acts first preflop:

```rust
// Heads-up (n == 2): button is the SB; multi-way (n >= 3): button is behind the blinds.
let (sb, bb) = if n == 2 {
    (button as usize, (button as usize + 1) % n)
} else {
    ((button as usize + 1) % n, (button as usize + 2) % n)
};

// First to act preflop: heads-up → the button (SB); multi-way → UTG (player after BB).
let first_to_act = if n == 2 { button as usize } else { (button as usize + 3) % n };
```

Blind amounts are clamped to the player's stack (`small_blind.min(stacks[sb])`), so a player too short to post a full blind goes all-in for what they have. `players_to_act` is seeded from `count_active()`, which naturally includes the big blind's preflop option to raise after a limped pot.

---

## 6. apply_action and undo_action: Zero Allocation in the Hot Path

`apply_action` is the engine's most performance-critical function — called once per tree edge in a CFR traversal. It records a **delta** of what is about to change, mutates in place, then advances the turn.

### Delta Undo: Recording Only What Changed

A single action only ever changes **one player's** chips. So rather than snapshot all three `[u32; MAX_PLAYERS]` arrays, the undo record stores just the acting player's old per-player values plus the scalar fields ([`undo.rs`](../crates/poker-core/src/undo.rs)):

```rust
let p = self.to_act as usize;
let record = UndoRecord {
    action, player: p as u8,
    old_stack:           self.stacks[p],
    old_street_bet:      self.street_bets[p],
    old_total_committed: self.total_committed[p],
    old_street, old_to_act: self.to_act,
    old_current_bet:     self.current_bet,
    old_min_raise:       self.min_raise,
    old_folded: self.folded, old_allin: self.allin,
    old_last_aggressor:  self.last_aggressor,
    old_players_to_act:  self.players_to_act,
    old_pot:             self.pot,
    street_changed:      false,                // set later if the street closes
    old_street_bets:     self.street_bets,     // only used when street_changed
};
self.undo.push(record);
```

The one wrinkle: when a betting round closes, `advance_street` resets **every** player's `street_bets` to zero. That is the single case where one-player-delta is insufficient, so the full pre-reset `street_bets` array is captured in `old_street_bets` and the record is flagged via `mark_street_changed()` after the fact. `undo_action` then branches on that flag:

```rust
if rec.street_changed {
    self.street_bets = rec.old_street_bets;      // restore the whole array
} else {
    self.street_bets[p] = rec.old_street_bet;    // patch one slot
}
// ...restore stacks[p], total_committed[p], pot, and all scalars...
```

The measured record size is **64 bytes** (versus ~88 for the old full snapshot). At a 256-deep pre-allocated stack that is ~16 KB of undo memory per `GameState`, and the common-case restore touches one player slot rather than three arrays.

### Debug-Only Invariants

In debug builds, `apply_action` enforces two invariants that compile out entirely in release:

- **Chip conservation**: `Σ stacks + Σ total_committed` must be identical before and after every action — a `debug_assert_eq!` that has caught more than one off-by-one in the betting logic.
- **Abstraction discipline**: a `Raise(total)` action must appear in `legal_actions(self)` — preventing raw, off-abstraction bet sizes from silently bypassing the blueprint.

These give the correctness benefits of aggressive assertions during development and testing without paying for them in the release traversal loop.

### Raise Semantics: Total Level, Not Increment

`Action::Raise(u32)` carries the **total street-bet level** the raising player is moving to, not a raise-by amount. If `current_bet` is 20 and a player raises to 60, the action is `Raise(60)`, not `Raise(40)`. This makes the action space easier to reason about: abstraction computes target levels directly, comparison between two raises is meaningful, and applying a raise is just `extra = total_bet - street_bets[p]`.

### AllIn vs Call: A Canonical Distinction

When a player's remaining stack is at most the call amount, `legal_actions` emits `AllIn` instead of `Call`:

```rust
if state.stacks[p] > to_call {
    actions.push(Action::Call);
} else {
    actions.push(Action::AllIn);   // calling would commit the whole stack
    return actions;
}
```

Both commit the same chips, but the distinction matters to CFR and information abstraction: a **voluntary** all-in is a different signal than a call that happens to be for stack. Encoding both as `Call` would erase that, so committing actions are always `AllIn`, distinguishable with a single pattern match.

### Side-Pot Handling at Showdown

`terminal_payoffs` returns each player's chip delta relative to their starting stack, handling side pots when players are all-in for different amounts:

1. Sort the unique `total_committed` levels — these are the pot-tier boundaries.
2. For each tier, every player who committed at least `level` contributes `(level - prev_level)` chips, **including folded players** (their chips stay in the pot but they cannot win).
3. Award the tier to the highest-ranked **eligible** hand (not folded, committed ≥ level); split ties evenly.

```rust
let contributor_count = (0..n).filter(|&i| self.total_committed[i] >= level).count() as u32;
let side_pot = contributor_count * (level - prev_level);
// eligible = not folded AND total_committed >= level; award side_pot to best eligible hand
```

Note `side_pot` uses **`contributor_count`**, not the eligible count — folded contributors' chips must remain in the pot for conservation to hold. Odd chips that don't divide evenly among tied winners go to the first winner seated left of the button, matching standard casino rules (Robert's Rules of Poker §15). Everything is stack-allocated; the sort is O(6 log 6); no `Vec` or `HashMap` is created.

---

## 7. The Undo Stack: Pre-Allocated Tree Traversal

The functional approach to game-tree traversal clones state at each node. The performance-sensitive approach is mutate-and-undo: keep one state object, mutate going down the tree, restore going back up. The undo stack's pre-allocation discipline is what makes that allocation-free ([`undo.rs`](../crates/poker-core/src/undo.rs)):

```rust
pub const MAX_UNDO_DEPTH: usize = 256;

pub struct UndoStack { records: Vec<UndoRecord> }

impl UndoStack {
    pub fn new() -> Self {
        Self { records: Vec::with_capacity(MAX_UNDO_DEPTH) }   // one-time allocation
    }
    pub fn push(&mut self, record: UndoRecord) {
        assert!(self.records.len() < MAX_UNDO_DEPTH, "undo stack overflow — game tree too deep");
        self.records.push(record);
    }
    pub fn pop(&mut self) -> Option<UndoRecord> { self.records.pop() }
    pub fn mark_street_changed(&mut self) {
        if let Some(rec) = self.records.last_mut() { rec.street_changed = true; }
    }
}
```

`Vec::with_capacity(256)` allocates the buffer once in `GameState::new`. After that, `push`/`pop` only move the Vec's length field and read/write the pre-allocated slots — no heap traffic during traversal. The overflow guard is a hard `assert!` (not `debug_assert!`): a 256-ply overflow signals a corrupt game loop, and failing loudly even in release is the safer choice.

**Why 256?** A poker hand has at most `4 streets × (a few raises + calls)` plies; even pathological action sequences stay well under 50. At 64 bytes per record, 256 slots cost ~16 KB per `GameState` — a generous margin for a negligible footprint.

**Alternative — explicit cloning.** Cloning a ~150-byte `GameState` per node is fast in isolation, but across millions of nodes per second the allocator churn (allocate going down, free going back up) dominates. The undo stack makes memory traffic completely predictable.

---

## 8. Action Abstraction: Taming the Infinite Action Space

### Why Abstraction Is Necessary

No-Limit Hold'em allows bets of any integer chip amount from the minimum raise up to the player's stack. A continuous bet-size space makes the tree infinite, and CFR cannot converge on it. The standard fix is **action abstraction**: replace the continuous space with a small set of representative sizes that are (1) strategically diverse and (2) few enough to keep the tree tractable.

### Pot-Fraction Sizing by Street — in Integer Arithmetic

Abstract bet sizes are expressed as fractions of the pot, by street, and stored as **`(numerator, denominator)` integer pairs** — never floats ([`betting.rs`](../crates/poker-core/src/betting.rs)):

```rust
pub const FLOP_BET_FRACS:    &[(u32, u32)] = &[(33, 100), (67, 100), (1, 1)];
pub const TURN_BET_FRACS:    &[(u32, u32)] = &[(1, 2), (3, 4), (1, 1)];
pub const RIVER_BET_FRACS:   &[(u32, u32)] = &[(1, 2), (3, 4), (1, 1), (3, 2)]; // incl. 1.5× overbet
pub const PREFLOP_BET_FRACS: &[(u32, u32)] = &[(1, 2), (1, 1), (2, 1)];
```

The river gets an extra 1.5× overbet because **overbets are disproportionately important on the river**: with no cards to come, polarized ranges benefit from oversized bets that put opponents to hard decisions. Omit it and the agent never learns it, leaving value on the table.

**Why integer pairs and not `f64`?** Determinism. Floating-point rounding can differ across platforms, compiler versions, and optimization levels, which would make the abstract game tree itself non-reproducible — a nightmare for a solver whose entire validation story rests on deterministic runs. All sizing is integer-only, with explicit round-half-up.

### abstract_raise_amounts: Stack-Allocated Return

The function returns a fixed-size array plus a count — never a `Vec`:

```rust
pub fn abstract_raise_amounts(
    pot: u32, current_bet: u32, min_raise: u32, street: u8,
) -> ([u32; 6], usize) {
    let mut amounts = [0u32; 6];
    let mut count = 0usize;
    let fracs: &[(u32, u32)] = match street { 0 => PREFLOP_BET_FRACS, 1 => FLOP_BET_FRACS, /* ... */ };

    for &(num, den) in fracs {
        let base = pot as u64 + current_bet as u64;
        let raise_size = ((base * num as u64 + den as u64 / 2) / den as u64) as u32; // round-half-up
        let new_bet = current_bet + raise_size.max(min_raise);                       // floor at min_raise
        if count == 0 || amounts[count - 1] != new_bet {                             // dedup
            amounts[count] = new_bet;
            count += 1;
        }
    }
    (amounts, count)
}
```

Two things changed from a naive design. First, the raise floor is **`min_raise`**, not the big blind — `min_raise` tracks the size of the last raise, which is the correct NLHE minimum after prior aggression. Second, the sizing base is `pot + current_bet`: a "1× pot bet" conventionally means betting the pot *after* the caller matches the current bet, so with `pot = 100` and `current_bet = 20`, a pot-sized raise targets `current_bet + (120) = 140`. The dedup guard collapses fractions that round to the same chip amount, and the maximum (river, 4 sizes) fits comfortably in the 6-slot array.

### Snapping Raw Bets to the Nearest Abstract Size

When a real game provides an off-abstraction bet (from a human or another bot), `abstract_bet_size(pot, current_bet, raw_bet, min_raise, street)` maps it to the nearest abstract level by absolute chip distance. This nearest-neighbor mapping is the standard way an abstracted solver copes with opponents who don't bet on the grid.

---

## 9. legal_actions: Combining Rules and Abstraction

`legal_actions` is where game rules and action abstraction meet. It returns a stack-allocated `ActionList` — a `[Action; 8]` buffer with a length, that `Deref`s to `&[Action]` so all slice methods work — so **even the legal-action enumeration is heap-free** ([`action.rs`](../crates/poker-core/src/action.rs)):

```rust
pub fn legal_actions(state: &GameState) -> ActionList {
    let mut actions = ActionList::new();
    if state.is_terminal() { return actions; }

    let p = state.to_act as usize;
    let to_call = state.current_bet.saturating_sub(state.street_bets[p]);
    let max_bet = state.stacks[p] + state.street_bets[p];

    // passive options
    if to_call == 0 {
        actions.push(Action::Check);
    } else {
        actions.push(Action::Fold);
        if state.stacks[p] > to_call {
            actions.push(Action::Call);
        } else {
            actions.push(Action::AllIn);   // calling commits the stack
            return actions;
        }
    }

    // aggressive options: abstract raises that clear the min-raise, then AllIn
    let min_raise_total = state.current_bet + state.min_raise;
    let (abstract_bets, n) =
        abstract_raise_amounts(state.pot, state.current_bet, state.min_raise, state.street);
    let mut allin_added = false;
    for &bet_level in &abstract_bets[..n] {
        if bet_level < min_raise_total { continue; }
        if bet_level >= max_bet {
            if !allin_added { actions.push(Action::AllIn); allin_added = true; }
            break;
        }
        actions.push(Action::Raise(bet_level));
    }
    if !allin_added && max_bet >= min_raise_total { actions.push(Action::AllIn); }

    actions
}
```

The 8-slot capacity covers the worst case (fold/check/call + up to 4 abstract raises + all-in). Key correctness points: `players_to_act` (maintained in `apply_action`) is what tells the engine when a betting round closes; the `min_raise_total` filter blocks sub-minimum raises; and `saturating_sub` defends against underflow in `to_call`.

---

## 10. Performance: Benchmarking Kuhn Poker Tree Traversal

### Why Kuhn Poker?

Before benchmarking on 6-max NLHE (an enormous tree), the benchmark ([`poker-ai/src/bin/benchmark.rs`](../crates/poker-ai/src/bin/benchmark.rs)) validates the engine on **Kuhn Poker**: 3 cards (J, Q, K), 2 players, one betting round, 6 deals. Kuhn serves two roles: a **correctness check** (the small tree's terminal count can be reasoned about by hand) and a **throughput measurement** (500,000 full traversals of all 6 deals).

The mapping onto the NLHE engine: 2 players, button = player 0 (the SB, who acts first heads-up), 3-chip stacks, small blind = 1, big blind = 2, no board cards (showdown evaluation isn't exercised during traversal). After the blinds are posted the button/SB has 2 chips behind and the BB has 1, which tightly bounds the raise sizes and keeps the per-deal tree small. Each Kuhn card is duplicated into both hole-card slots with different suits to satisfy the 2-card interface.

### Recursive vs Explicit-Stack Traversal

The benchmark compares two traversal algorithms, both allocation-free in the hot path. A `fill_legal_actions` helper writes into a caller-supplied `[Action; 8]` array, and the explicit-stack variant keeps its frames in a fixed `[StackFrame; 64]` on the system stack:

```rust
fn traverse_recursive(state: &mut GameState) -> u64 {
    if state.is_terminal() { return 1; }
    let mut actions = [Action::Fold; MAX_ACTIONS];
    let n = fill_legal_actions(state, &mut actions);
    let mut count = 0;
    for &action in &actions[..n] {
        state.apply_action(action);
        count += traverse_recursive(state);
        state.undo_action();
    }
    count
}
```

The explicit-stack version is semantically identical but manages its own frame array, trading register-friendliness for bounded stack depth.

### Benchmark Results and Interpretation

Measured on the development machine (Apple M1, single thread, release build) via `cargo run --release --bin benchmark`:

```
UndoRecord size (delta):  64 bytes  (old snapshot ~88 bytes; depth 256 → ~16 KB)
Terminal nodes per full traversal (6 Kuhn deals): 108

Recursive:        ~299,000 traversals/sec  (1.671 s / 500,000 iters)
Explicit stack:   ~280,000 traversals/sec  (1.786 s / 500,000 iters)
Winner: Recursive (1.07× faster than explicit stack)

evaluate_5:  ~1.6e9 evals/sec   (constant-input microbench; heavily optimized)
evaluate_6:  ~87M  evals/sec    (~11.5 ns/eval)
evaluate_7:  ~86M  evals/sec    (~11.7 ns/eval)
```

The recursive version wins by a small margin. The compiler keeps loop counters and action indices in registers across recursive calls (the explicit stack must spill them to its frame array), and the tight `for` loop is more branch-predictor-friendly than the explicit `loop { … }`. For Kuhn's shallow tree, OS call-stack overhead is negligible; on a much deeper NLHE tree the explicit stack could close the gap by avoiding frame setup. The headline point holds either way: **neither method allocates**, so the choice is a microarchitecture detail, not an algorithmic one.

Two notes on the numbers. First, the `evaluate_5` figure is a microbenchmark on a *fixed* hand and is largely constant-folded by LLVM — treat `evaluate_6/7` (~12 ns) as the representative cost. That ~12 ns is roughly an order of magnitude faster than the old 21-subset evaluator, and showdowns occur only at terminal leaves, never in the inner `apply_action` loop — so hand evaluation is comfortably off the critical path. Second, absolute traversal throughput is machine- and tree-specific; reproduce it on your own hardware rather than treating these as universal constants.

At ~300K full six-deal traversals/sec with 108 terminals each, the engine sustains tens of millions of terminal-path visits per second on a single laptop core — fast enough that, as intended, the binding constraint on solver convergence is abstraction quality and CFR efficiency, not the game engine.

---

## 11. Key Design Tradeoffs

### Two-Tier Evaluation: Algorithmic Reference + Compact LUT

**Decision**: Ship *both* an algorithmic single-pass evaluator and a compact lookup-table evaluator, and use the LUT one in `GameState`.

| Approach | Speed (7-card) | Memory | Role |
|----------|----------------|--------|------|
| Algorithmic | ~12 ns/hand | ~0 KB | Correctness oracle; no tables needed |
| LUT (`lut_eval`) | faster still | ~tens of KB (compile-time) | Hot-path evaluator `GameState` calls |

This is a *both/and*, not an *either/or*. The algorithmic evaluator is the reference an `#[ignore]`d exhaustive test validates the LUT against on all 2.6M five-card hands; the LUT is the speed. Crucially the LUT here is **tens of kilobytes**, not the hundred-plus-megabyte tables of full 7-card lookup evaluators, so it does not evict the CFR regret tables from cache. Both share one non-flush classifier, so they cannot silently diverge.

### Pre-Dealt vs Lazily-Dealt Board

**Decision**: Pre-deal all 5 board cards at `GameState::new` rather than sampling them as streets advance.

- **Pro**: Enables Public Chance Sampling naturally; clean separation between the CFR sampling layer and the engine.
- **Con**: The engine can't "deal the turn now" on its own; the caller provides all cards upfront and constructs a fresh `GameState` to resample a runout.

For CFR with public chance sampling this is unambiguously correct.

### Delta Undo vs Full-Snapshot Undo

**Decision**: Record a **delta** (the acting player's old chip fields + scalars, plus the full `street_bets` array only when a street closes), not a full snapshot of every array.

- **Delta**: 64-byte record; common-case restore patches one player slot; needs the `street_changed` flag for the round-closing case.
- **Full snapshot**: simpler restore (one memcpy), but ~88 bytes and three arrays copied on every action regardless of what changed.

The delta is both smaller and, in the common case, cheaper to apply — at the cost of one branch in `undo_action`. (This reverses an earlier full-snapshot design.)

### Integer vs Floating-Point Bet Sizing

**Decision**: Express pot fractions as integer `(numerator, denominator)` pairs and compute bet sizes with integer round-half-up, never `f64`.

- **Integer**: bit-for-bit reproducible across platforms and compilers — essential for a deterministic solver and its validation tests.
- **Floating-point**: marginally more convenient to read, but introduces platform-dependent rounding that would make the abstract game tree itself non-reproducible.

For a system whose correctness story depends on deterministic runs, integer math is the only defensible choice.

### Bitmask State vs Enum Arrays

**Decision**: Use `u8` bitmasks for `folded` and `allin` rather than `[PlayerStatus; 6]`.

Bitmasks fit in a register, make `count_active()` a single popcount, and keep `UndoRecord` small. The cost is slightly less readable code and an 8-player ceiling — fine for 6-max.

### Abstract Action Space vs Full NLHE Tree

**Decision**: Limit bet sizes to 3–4 abstract pot fractions per street rather than arbitrary chip amounts.

This is a fundamental poker-AI decision, not a micro-optimization. Abstraction makes the tree finite (so CFR converges) at the cost of **abstraction error** — strategies optimal in the abstract game may be suboptimal in the full game. The bet-size calibration (notably the river overbet) exists to minimize that error in the spots that matter most.

---

## 12. Lessons for Other Performance-Critical Projects

The patterns in `poker-core` generalize beyond poker:

**1. Pre-allocate everything reused in a hot loop.** `UndoStack::new()` is the only allocation-bearing constructor; all subsequent traversal is allocation-free. "Allocate once, reuse forever" applies to any system with a hot path.

**2. Measure early; don't guess.** The benchmark is a first-class citizen, run before building the solver to confirm the traversal rate is adequate. If it had come back 100× slower, the engine architecture — not the solver — would have been the thing to rethink.

**3. Keep two implementations when one is the oracle.** The algorithmic evaluator exists *so that* the fast LUT evaluator can be exhaustively validated against it. A slow, obviously-correct reference is worth its weight when the fast path is subtle.

**4. Make invariants free in release.** Chip-conservation and abstraction-discipline checks run as `debug_assert!` in tests and development, then vanish in the release traversal loop — correctness during dev, speed in production.

**5. Encoding matters more than algorithm for hot-path data.** The `rank*4+suit` card byte, the bitmask player state, the rank-prime product, and the packed `u32` hand score all turn common operations into single instructions. For data touched billions of times, representation dominates.

**6. Separate concerns at module boundaries.** The engine knows nothing about CFR; the evaluator knows nothing about game state; the betting module knows nothing about legal actions. Each is testable and replaceable in isolation.

---

## 13. What Comes Next

`poker-core` is Phase 1 of a multi-phase roadmap (`poker-ai-plan-v3.md`) to build a competitive heads-up / 6-max NLHE AI. The engine is the foundation; the strength comes from everything built on top in the `poker-ai` crate:

**Information Abstraction**: compress the astronomical hand/board space into a tractable set of buckets using equity-distribution features clustered with K-Means++. Coarse abstraction is a permanent ceiling no amount of solver iterations can overcome.

**Blueprint CFR Solver**: Discounted CFR (DCFR) with VR-MCCFR baselines, optimistic updates, and regret-based pruning over the abstracted tree, with external sampling making it tractable. Validated first on Kuhn and Leduc against known equilibria, then on real mechanics.

**Evaluation**: exact best-response exploitability where the chance space is enumerable, and a sampled Local Best Response (LBR) estimator for the non-enumerable full game.

**Subgame Resolving**: a depth-limited real-time solver that, instead of playing the blueprint directly, re-solves the current public state within a time budget using Bayesian belief tracking — the component that turns a blueprint into a strong player.

Every one of these depends on `poker-core`'s performance. At tens of millions of node visits per second on a single core, the bottleneck sits where it should: in abstraction quality and CFR efficiency, not the engine.

---

## Summary

`poker-core` is a performance-critical system with a clear contract and explicit, sometimes-reversed tradeoffs:

- **Single-byte card encoding** enables one-instruction rank/suit extraction and trivial rank bitmasks
- **Two evaluators** — an algorithmic single-pass reference and a compact compile-time LUT (the hot path) — give both a correctness oracle and speed, without a giant cache-evicting table
- **Bitmask player state** packs six players' status into two bytes
- **Pre-dealt board cards** enable Public Chance Sampling without complicating the engine
- **Delta undo** records only what changed (64 bytes/record), keeping tree traversal allocation-free
- **Integer pot-fraction sizing** keeps the abstract game tree deterministic across platforms
- **Abstract action space** makes the tree finite while preserving the strategically important sizes
- **Pre-allocated undo stack** removes heap allocation from the hot path entirely

The result is an engine that traverses tens of millions of nodes per second on a single core, with hand evaluation an order of magnitude off the critical path — fast enough that solver convergence is bottlenecked by abstraction and algorithm quality, not by the engine. That's the goal: infrastructure that disappears.

---

*This article describes the `poker-core` crate as implemented in this repository. The broader solver pipeline — DCFR/MCCFR, information abstraction, exploitability evaluation, and subgame resolving — lives in `poker-ai` and is described in `poker-ai-plan-v3.md` and the companion progress notes.*
