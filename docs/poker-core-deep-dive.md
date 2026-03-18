# Building a High-Performance Poker Core Engine in Rust

## Design Decisions, Tradeoffs, and Performance

> **~20 minute read** · ~5,200 words · Intermediate to Advanced

---

## Table of Contents

1. [Why Poker AI Is Hard](#1-why-poker-ai-is-hard)
2. [The Role of the Core Engine](#2-the-role-of-the-core-engine)
3. [Card Encoding: Small Numbers, Big Wins](#3-card-encoding-small-numbers-big-wins)
4. [Hand Strength Evaluation](#4-hand-strength-evaluation)
   - [The `make_hand` Encoding](#the-make_hand-encoding)
   - [evaluate_5: One Pass, No Heap](#evaluate_5-one-pass-no-heap)
   - [Straight Detection and the Wheel Edge Case](#straight-detection-and-the-wheel-edge-case)
   - [Extending to 6 and 7 Cards](#extending-to-6-and-7-cards)
5. [GameState: Packing Six Players into Tight Memory](#5-gamestate-packing-six-players-into-tight-memory)
   - [Bitmasks for Player Status](#bitmasks-for-player-status)
   - [Pre-Dealt Board Cards (Public Chance Sampling)](#pre-dealt-board-cards-public-chance-sampling)
   - [Posting Blinds at Construction](#posting-blinds-at-construction)
6. [apply_action and undo_action: Zero Allocation in the Hot Path](#6-apply_action-and-undo_action-zero-allocation-in-the-hot-path)
   - [Raise Semantics: Total Level, Not Increment](#raise-semantics-total-level-not-increment)
   - [AllIn vs Call: A Canonical Distinction](#allin-vs-call-a-canonical-distinction)
   - [Side-Pot Handling at Showdown](#side-pot-handling-at-showdown)
7. [The Undo Stack: Pre-Allocated Tree Traversal](#7-the-undo-stack-pre-allocated-tree-traversal)
8. [Action Abstraction: Taming the Infinite Action Space](#8-action-abstraction-taming-the-infinite-action-space)
   - [Why Abstraction Is Necessary](#why-abstraction-is-necessary)
   - [Pot-Fraction Sizing by Street](#pot-fraction-sizing-by-street)
   - [abstract_raise_amounts: Stack-Allocated Return](#abstract_raise_amounts-stack-allocated-return)
   - [Snapping Raw Bets to the Nearest Abstract Size](#snapping-raw-bets-to-the-nearest-abstract-size)
9. [legal_actions: Combining Rules and Abstraction](#9-legal_actions-combining-rules-and-abstraction)
10. [Performance: Benchmarking Kuhn Poker Tree Traversal](#10-performance-benchmarking-kuhn-poker-tree-traversal)
    - [Why Kuhn Poker?](#why-kuhn-poker)
    - [Recursive vs Explicit-Stack Traversal](#recursive-vs-explicit-stack-traversal)
    - [Benchmark Results and Interpretation](#benchmark-results-and-interpretation)
11. [Key Design Tradeoffs](#11-key-design-tradeoffs)
    - [Algorithmic Evaluator vs Lookup Table](#algorithmic-evaluator-vs-lookup-table)
    - [Pre-Dealt vs Lazily-Dealt Board](#pre-dealt-vs-lazily-dealt-board)
    - [Full Snapshot Undo vs Delta Undo](#full-snapshot-undo-vs-delta-undo)
    - [Bitmask State vs Enum Arrays](#bitmask-state-vs-enum-arrays)
    - [Abstract Action Space vs Full NLHE Tree](#abstract-action-space-vs-full-nlhe-tree)
12. [Lessons for Other Performance-Critical Projects](#12-lessons-for-other-performance-critical-projects)
13. [What Comes Next](#13-what-comes-next)

---

## 1. Why Poker AI Is Hard

Chess and Go have well-defined search spaces: both players see the full board at all times. Poker is different. Each player holds **private information** — hole cards — that their opponents cannot observe. This makes poker a game of **imperfect information**, and solving it requires fundamentally different algorithms.

The breakthrough came when Carnegie Mellon's Libratus defeated world-class human professionals in 2017, followed by Pluribus achieving superhuman 6-player performance in 2019. Unlike prior systems that relied on hand-crafted heuristics, these bots used **Counterfactual Regret Minimization (CFR)** — an iterative self-play algorithm that converges to a Nash equilibrium strategy with no domain knowledge beyond the game rules.

The catch: running CFR at scale requires traversing an enormous game tree billions of times. The full 6-player NLHE tree has on the order of 10¹⁶⁰ nodes when you account for all possible card combinations and action sequences. Even Pluribus — which ran on a server cluster for eight days — needed aggressive abstraction to make this tractable.

This means the **game engine** sits at the very bottom of the performance critical path. Every nanosecond saved in `apply_action` translates directly into more CFR iterations, and more iterations translate into a stronger strategy. A poorly engineered engine is a permanent bottleneck; a well-engineered one is invisible infrastructure that the solver just takes for granted.

This article is a deep dive into `poker-core`: a purpose-built Rust crate that aims to be exactly that kind of invisible, fast infrastructure.

---

## 2. The Role of the Core Engine

`poker-core` has one job: **represent and advance a 6-max No-Limit Texas Hold'em game state as fast as possible**, with perfect correctness. Its public API is tiny:

| Function / Type | Purpose |
|----------------|---------|
| `GameState::new(...)` | Construct a new hand, post blinds |
| `apply_action(action)` | Advance state, push undo record |
| `undo_action()` | Restore previous state |
| `is_terminal()` | Check if the hand is over |
| `terminal_payoffs()` | Return chip deltas at showdown |
| `legal_actions(state)` | Enumerate valid abstract actions |
| `evaluate_5/6/7(cards)` | Evaluate hand strength |

Everything else — the CFR solver, information abstraction, belief tracking — lives in `poker-ai` and calls into these primitives. The contract is simple: **the engine never allocates on the heap in the hot path**, and every operation is O(1) or O(small constant).

---

## 3. Card Encoding: Small Numbers, Big Wins

The most fundamental design decision is how to represent a playing card. There are 52 cards; they need to fit in as little memory as possible and support fast rank/suit extraction.

The chosen encoding packs both rank and suit into a single byte:

```
card = rank * 4 + suit

rank: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=T (Ten), 9=J, 10=Q, 11=K, 12=A
suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
```

This gives values in the range `[0, 51]`, fitting in a `u8`. The three helper functions are each one instruction:

```rust
#[inline]
pub fn rank_of(card: u8) -> u8 { card >> 2 }      // bits 7–2

#[inline]
pub fn suit_of(card: u8) -> u8 { card & 3 }        // bits 1–0

#[inline]
pub fn make_card(rank: u8, suit: u8) -> u8 { (rank << 2) | suit }
```

The sentinel value `NO_CARD = 0xFF` marks board slots not yet revealed (pre-flop, turn, river). Since valid cards are in `[0, 51]`, `0xFF` cannot collide.

**Why not `(suit * 13) + rank`?** The rank-major layout has a subtle advantage: `rank_of` is a single right-shift, and consecutive ranks produce consecutive byte values. This makes range checks like `ranks[0] - ranks[4] == 4` (straight detection) work naturally on the extracted rank bytes.

**Why not a `struct Card { rank: u8, suit: u8 }`?** A struct would consume 2 bytes (or potentially more due to alignment) per card, and array operations would touch more cache lines. For the evaluator, which reads 5–7 cards in a tight loop, the u8 packing keeps an entire hand in a few cache lines.

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
fn make_hand(cat: u8, r1: u8, r2: u8, r3: u8, r4: u8, r5: u8) -> u32 {
    ((cat as u32) << 20)
        | ((r1 as u32) << 16)
        | ((r2 as u32) << 12)
        | ((r3 as u32) << 8)
        | ((r4 as u32) << 4)
        | (r5 as u32)
}
```

The beauty of this design: **comparing two hands is just `a > b`** — no special comparison logic needed. A full house (category 6) always beats a flush (category 5) because `6 << 20 > 5 << 20`. Within the same category, tiebreakers are embedded in decreasing significance, so the highest-ranking tiebreaker wins automatically.

This technique is sometimes called a "score encoding" or "rank word" and is standard in fast evaluators. The 4-bit slots for rank are exactly right: ranks run from 0 to 12, requiring 4 bits (max value 15 ≥ 12). The category requires 4 bits as well (max value 8).

### evaluate_5: One Pass, No Heap

The core of the evaluator is `evaluate_5`, which classifies a 5-card hand:

```rust
pub fn evaluate_5(cards: &[u8; 5]) -> u32 {
    // 1. Extract ranks and suits into stack arrays
    let mut ranks = [0u8; 5];
    let mut suits = [0u8; 5];
    for i in 0..5 {
        ranks[i] = rank_of(cards[i]);
        suits[i] = suit_of(cards[i]);
    }

    // 2. Flush test: are all five suits equal?
    let is_flush = suits[0] == suits[1]
        && suits[1] == suits[2]
        && suits[2] == suits[3]
        && suits[3] == suits[4];

    // 3. Sort ranks descending (for straight detection and kicker ordering)
    ranks.sort_unstable_by(|a, b| b.cmp(a));

    // 4. Build frequency table: freq[r] = how many cards have rank r
    let mut freq = [0u8; 13];
    for &r in &ranks { freq[r as usize] += 1; }

    // 5. Classify using freq table...
}
```

All intermediate data structures — `ranks`, `suits`, `freq` — are fixed-size stack arrays. The function has zero heap allocations. The maximum working set is:
- `ranks`: 5 bytes
- `suits`: 5 bytes  
- `freq`: 13 bytes
- Plus a handful of scalar registers

This is 23 bytes, comfortably fitting in a handful of CPU registers or the first cache line. Repeated calls in a tight loop will never trigger cache misses on the working set itself.

**Sorting**: `sort_unstable_by` on a 5-element array uses an optimized sorting network in practice (Rust's standard library uses insertion sort for small arrays). This is O(10) comparisons worst case — faster than any comparison-based sort for larger inputs.

**Frequency table**: A 13-byte array indexed by rank. Building it takes 5 iterations; scanning it for pairs/trips/quads takes 13 iterations. Both are O(1) with small constants.

### Straight Detection and the Wheel Edge Case

The normal straight test is elegant:

```rust
let all_distinct = freq.iter().all(|&f| f <= 1);
let is_normal_straight = all_distinct && (ranks[0] - ranks[4] == 4);
```

After sorting descending, a straight exists if and only if all ranks are distinct and the spread from top to bottom is exactly 4. This is O(13) for the distinct check, then O(1) for the spread check.

The **wheel** (A-2-3-4-5) is the trickiest edge case in poker hand evaluation. An Ace can rank low in a straight, giving a "five-high" straight. After sorting descending, the wheel appears as `[12, 3, 2, 1, 0]` — ranks 12 (Ace), 3 (Five), 2 (Four), 1 (Three), 0 (Two). This fails the `ranks[0] - ranks[4] == 4` test (12 - 0 = 12 ≠ 4), so it requires special handling:

```rust
let is_wheel =
    ranks[0] == 12 && ranks[1] == 3 && ranks[2] == 2
    && ranks[3] == 1 && ranks[4] == 0;

// For the wheel the "effective top" is 3 (five-high)
let straight_top = if is_wheel { 3 } else { ranks[0] };
```

Setting `straight_top = 3` (Five) means `make_hand(4, 3, 0, 0, 0, 0)` is produced for the wheel, which correctly ranks below `make_hand(4, 4, 0, 0, 0, 0)` (Six-high straight). Many evaluators get this wrong or handle it with a special-case table; this approach integrates it cleanly into the normal straight classification path.

### Extending to 6 and 7 Cards

Texas Hold'em players choose the best 5 cards from 7 (2 hole cards + 5 board cards). The evaluator handles this by exhaustively trying all subsets:

```rust
pub fn evaluate_7(cards: &[u8; 7]) -> u32 {
    let mut best = 0u32;
    for i in 0..7usize {
        for j in (i + 1)..7usize {
            let mut hand = [0u8; 5];
            let mut idx = 0;
            for (k, &card) in cards.iter().enumerate() {
                if k != i && k != j {
                    hand[idx] = card;
                    idx += 1;
                }
            }
            let rank = evaluate_5(&hand);
            if rank > best { best = rank; }
        }
    }
    best
}
```

C(7,5) = 21 subsets × `evaluate_5` call each. Similarly, `evaluate_6` tries C(6,5) = 6 subsets. These functions also use no heap allocation: `hand` is a stack-allocated `[u8; 5]` that lives for one iteration.

**Performance tradeoff**: More sophisticated evaluators (the 2+2 evaluator, PokerStove, etc.) use lookup tables of up to 128 MB to evaluate 7-card hands in O(1). The algorithmic approach here is 10–100× slower on raw evaluations. However, the `evaluate_7` function is only called once per terminal showdown node — not inside the inner traversal loop. The inner loop only calls `apply_action` and `undo_action`. So even if `evaluate_7` takes 200 ns instead of 2 ns, it has negligible impact on overall traversal throughput.

The deliberate choice to avoid lookup tables:
- No startup latency (no table precomputation)
- No 128 MB cache footprint (which would evict the CFR regret tables)
- Simpler code, fewer dependencies
- Correctness is easier to verify

For a system spending billions of iterations in the solver loop with showdowns only at leaf nodes, this is the right tradeoff.

---

## 5. GameState: Packing Six Players into Tight Memory

`GameState` holds all mutable state for one hand of 6-max NLHE. Its fields tell the full story of the design philosophy:

```rust
pub struct GameState {
    pub stacks:          [u32; MAX_PLAYERS],   // chip stacks
    pub street_bets:     [u32; MAX_PLAYERS],   // chips committed this street
    pub total_committed: [u32; MAX_PLAYERS],   // chips committed entire hand
    pub board:           [u8;  5],             // all 5 community cards (pre-dealt)
    pub hole_cards:      [[u8; 2]; MAX_PLAYERS],
    pub street:          u8,                   // 0=preflop, 1=flop, 2=turn, 3=river, 4=terminal
    pub to_act:          u8,                   // index of player whose turn it is
    pub num_players:     u8,
    pub button:          u8,
    pub big_blind:       u32,
    pub current_bet:     u32,                  // highest street_bet (amount to call)
    pub min_raise:       u32,                  // minimum raise increment
    pub folded:          u8,                   // bitmask: bit i set if player i folded
    pub allin:           u8,                   // bitmask: bit i set if player i is all-in
    pub last_aggressor:  u8,                   // last player to bet/raise (0xFF = none)
    pub players_to_act:  u8,                   // active players who still need to act
    pub undo:            UndoStack,            // pre-allocated undo history
}
```

`MAX_PLAYERS = 6`. The struct is roughly 200 bytes — small enough to fit in a few cache lines. For a tree traversal that mutates the same `GameState` in-place, good locality here is important.

### Bitmasks for Player Status

`folded` and `allin` are both `u8` bitmasks: bit `i` corresponds to player `i`. Status checks and updates are single instructions:

```rust
// Is player p folded?
let is_folded = (self.folded >> p) & 1 == 1;

// Mark player p as folded:
self.folded |= 1 << p;

// Count non-folded players:
(0..n).filter(|&i| (self.folded >> i) & 1 == 0).count()
```

The alternative would be a `[PlayerStatus; 6]` array where `PlayerStatus` is an enum. The bitmask approach uses 2 bytes instead of 6+ bytes for the same information, and the individual tests are single bitwise operations rather than enum comparisons. For hot-path code called at every node, these micro-optimizations accumulate.

### Pre-Dealt Board Cards (Public Chance Sampling)

One of the most important design decisions in `GameState` is that **all five board cards are passed at construction time**, even though only 3 are visible on the flop, 4 on the turn, and 5 on the river. Unrevealed cards are hidden from the game logic by checking `board_cards_count()`:

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

This is specifically designed for **Public Chance Sampling (PCS)**, the CFR sampling strategy used in Pluribus. In PCS, you fix the public board cards (the community cards everyone sees) at the root of a subtree and traverse all player action paths beneath it. The alternative — lazily sampling each community card as streets advance — would require storing board-sampling logic inside `apply_action` and would complicate the traversal loop.

With pre-dealt boards, the CFR outer loop looks like:
1. Deal all 5 board cards and 2 × n hole cards
2. Construct `GameState` with all cards pre-loaded
3. Traverse the action tree from the root
4. Repeat with new cards

This separation of concerns keeps the game engine stateless with respect to card dealing — it just follows the action tree — and keeps the sampling logic entirely in the CFR layer.

### Posting Blinds at Construction

`GameState::new` automatically posts the small blind and big blind. This ensures the game is always in a valid state immediately after construction. The position logic handles the edge case of heads-up play (where positional roles are reversed):

```rust
let sb = (button as usize + 1) % n;
let bb = (button as usize + 2) % n;
// ...
// First to act preflop: UTG (player after BB), or in heads-up: the button (SB)
let first_to_act = if n == 2 {
    (button as usize + 1) % n  // HU: SB/button acts first preflop
} else {
    (button as usize + 3) % n  // 3+: UTG acts first preflop
};
```

---

## 6. apply_action and undo_action: Zero Allocation in the Hot Path

`apply_action` is the engine's most performance-critical function. In a CFR traversal, it is called once per tree edge. The first thing it does is snapshot the current state into an `UndoRecord` and push it onto the pre-allocated undo stack:

```rust
pub fn apply_action(&mut self, action: Action) {
    let record = UndoRecord {
        action,
        stacks:          self.stacks,
        street_bets:     self.street_bets,
        total_committed: self.total_committed,
        street:          self.street,
        to_act:          self.to_act,
        current_bet:     self.current_bet,
        min_raise:       self.min_raise,
        folded:          self.folded,
        allin:           self.allin,
        last_aggressor:  self.last_aggressor,
        players_to_act:  self.players_to_act,
    };
    self.undo.push(record);
    // ... apply the action ...
}
```

The undo stack is pre-allocated with capacity 256 in the constructor — no allocation ever occurs in the push operation during traversal.

`undo_action` simply pops the record and copies its fields back:

```rust
pub fn undo_action(&mut self) {
    if let Some(rec) = self.undo.pop() {
        self.stacks          = rec.stacks;
        self.street_bets     = rec.street_bets;
        self.total_committed = rec.total_committed;
        self.street          = rec.street;
        self.to_act          = rec.to_act;
        self.current_bet     = rec.current_bet;
        self.min_raise       = rec.min_raise;
        self.folded          = rec.folded;
        self.allin           = rec.allin;
        self.last_aggressor  = rec.last_aggressor;
        self.players_to_act  = rec.players_to_act;
    }
}
```

This is a full-snapshot undo: every mutable field is restored in its entirety. The tradeoff (discussed further in [Section 11](#11-key-design-tradeoffs)) is that the undo record is larger than a minimal delta, but the restore operation is a single sequential memory copy rather than a conditional sequence of partial restores.

### Raise Semantics: Total Level, Not Increment

`Action::Raise(u32)` carries the **total street-bet level** the raising player is moving to, not a raise-by-amount. If `current_bet` is 20 and a player wants to raise to 60, the action is `Raise(60)`, not `Raise(40)`.

This is a deliberate API choice that makes the action space easier to reason about:
- **Action abstraction** computes target bet levels naturally: `pot_fraction * pot + current_bet`
- **Comparison** between two raises is direct: `Raise(60) > Raise(40)`  
- **Applying** a raise just computes `extra = total_bet - street_bets[p]`

The alternative — raise-by-increment — requires the caller to always know the current bet level to compute valid raise amounts, and comparisons between `Raise(40)` in two different pot contexts are meaningless.

### AllIn vs Call: A Canonical Distinction

When a player's remaining stack is less than or equal to the call amount, `legal_actions` emits `AllIn` instead of `Call`. Both actions commit the same chips in that scenario, but they carry different semantics:

```rust
// In legal_actions:
if state.stacks[p] > to_call {
    actions.push(Action::Call);
} else {
    // Call is effectively all-in — only offer AllIn, not Call.
    actions.push(Action::AllIn);
    return actions;
}
```

For CFR and information abstraction, it matters whether a player **chose** to go all-in (a signal of hand strength) versus was **forced** all-in by a large bet. Encoding both scenarios as `Call` would lose this information. By using `AllIn` for both types of committing actions, downstream code can always distinguish "normal call" from "committing call" with a single pattern match on the action variant.

### Side-Pot Handling at Showdown

`terminal_payoffs` computes exact chip awards including **side pots** when players are all-in for different amounts. The algorithm:

1. Collect all unique total-committed levels (these define the pot tier boundaries)
2. For each tier boundary, compute which players are eligible (not folded, committed at least that much)
3. Award the side pot proportionally among eligible players with the highest hand rank

```rust
let mut tiers: [u32; MAX_PLAYERS] = self.total_committed;
tiers[..n].sort_unstable();

for &level in tiers[..n].iter() {
    if level <= prev_level { continue; }
    let eligible_mask: u8 = (0..n as u8)
        .filter(|&i| (self.folded >> i) & 1 == 0
                     && self.total_committed[i as usize] >= level)
        .fold(0u8, |acc, i| acc | (1 << i));
    let side_pot = eligible_count * (level - prev_level);
    // ... award to best hand among eligible ...
}
```

Stack-allocated arrays throughout; the sort is O(6 log 6) = constant time. No `Vec` or `HashMap` is created.

---

## 7. The Undo Stack: Pre-Allocated Tree Traversal

The canonical approach to game-tree traversal in functional languages is to clone the state at each node. In Rust and performance-sensitive C++, the preferred pattern is mutate-and-undo: maintain a single state object, mutate it going down the tree, and restore it going back up.

The undo stack implementation is straightforward but its pre-allocation discipline is critical:

```rust
pub struct UndoStack {
    records: Vec<UndoRecord>,
}

impl UndoStack {
    pub fn new() -> Self {
        Self {
            records: Vec::with_capacity(MAX_UNDO_DEPTH),  // MAX_UNDO_DEPTH = 256
        }
    }

    pub fn push(&mut self, record: UndoRecord) {
        debug_assert!(self.records.len() < MAX_UNDO_DEPTH, "undo stack overflow");
        self.records.push(record);
    }

    pub fn pop(&mut self) -> Option<UndoRecord> {
        self.records.pop()
    }
}
```

`Vec::with_capacity(256)` allocates memory for 256 `UndoRecord` entries upfront. After this one-time allocation in `GameState::new`, `push` and `pop` never allocate or free heap memory — they just increment/decrement the Vec's length field and read/write into the pre-allocated buffer.

Why 256? A poker game tree has at most `4 streets × max_actions_per_street` plies before reaching a terminal. In practice, even a very deep action sequence (multiple raises per street, 4 streets) stays well under 50 plies. 256 provides a generous safety margin with a ~25 KB memory cost per `GameState` (256 × ~100 bytes per `UndoRecord`).

**Alternative: explicit state cloning.** The CFR loop could clone `GameState` at every node instead. Cloning a ~200 byte struct is fast in isolation, but in a traversal of millions of nodes per second, the allocator overhead accumulates. More importantly, cloning would require deallocating the clone on the way back up, stressing the allocator in both directions. The undo-stack approach makes memory management completely predictable.

---

## 8. Action Abstraction: Taming the Infinite Action Space

### Why Abstraction Is Necessary

No-Limit Hold'em allows bets of any integer chip amount (from the minimum raise up to the player's stack). A game tree with continuous bet sizing is infinite — CFR cannot converge on it. The standard solution is **action abstraction**: replace the continuous bet-size space with a small, fixed set of representative sizes.

The challenge is choosing sizes that:
1. Are strategically diverse enough that the abstracted game captures the key decisions
2. Are few enough that the action tree remains tractable (exponential in the number of sizes)

### Pot-Fraction Sizing by Street

Abstract bet sizes are expressed as fractions of the pot, by street:

```rust
pub const FLOP_BET_FRACS:   &[f64] = &[0.33, 0.67, 1.0];
pub const TURN_BET_FRACS:   &[f64] = &[0.50, 0.75, 1.0];
pub const RIVER_BET_FRACS:  &[f64] = &[0.50, 0.75, 1.0, 1.50];
pub const PREFLOP_BET_FRACS: &[f64] = &[0.50, 1.0, 2.0];
```

The river gets an extra size (1.5× pot overbet) because **overbets are disproportionately important on the river**. At the river there are no more cards to come, so polarized hands — those that are either very strong or bluffs — benefit from oversized bets that force opponents into difficult decisions. If the action abstraction omits river overbets, the agent never learns to use them, leaving significant strategic value unexploited.

The preflop fractions are calibrated to produce sensible chip amounts from a standard blind structure. With a 10-chip big blind, the fractions 0.5/1.0/2.0 applied to the ~15-chip preflop pot yield roughly 2.5 BB / 3.5 BB / 6 BB opens — sizes that real players and solvers commonly use.

### abstract_raise_amounts: Stack-Allocated Return

A naive implementation might return a `Vec<u32>`. Instead, the function uses a fixed-size stack array with an explicit count:

```rust
pub fn abstract_raise_amounts(
    pot: u32,
    current_bet: u32,
    street: u8,
    big_blind: u32,
) -> ([u32; 6], usize) {
    let mut amounts = [0u32; 6];
    let mut count = 0usize;

    let fracs: &[f64] = match street { /* ... */ };

    for &frac in fracs {
        let raise_size = ((pot as f64 + current_bet as f64) * frac).round() as u32;
        // `.round()` rather than `.floor()` or `.ceil()` produces chip amounts
        // closest to the theoretical fraction, avoiding systematic under- or
        // over-sizing that would shift the abstract game away from the target.
        let new_bet = current_bet + raise_size.max(big_blind);
        if count == 0 || amounts[count - 1] != new_bet {
            amounts[count] = new_bet;
            count += 1;
        }
    }

    (amounts, count)
}
```

The maximum abstract sizes per street is 4 (river), well within the 6-element array. The deduplication check (`amounts[count - 1] != new_bet`) handles edge cases where two fractions round to the same chip amount.

**The sizing formula**: `raise_size = (pot + current_bet) × fraction`. Why add `current_bet` to `pot` before multiplying? Because "pot" in poker conventionally includes the amount the caller would add. A "1× pot bet" means: "I am betting an amount equal to the pot after my opponent calls." If `pot = 100` and `current_bet = 20`, then a 1× pot bet = `(100 + 20) × 1.0 = 120` — the total street bet the raiser moves to is `current_bet + 120 = 140`. This matches how solvers and human players conventionally define pot-sized bets.

### Snapping Raw Bets to the Nearest Abstract Size

When a real game provides a raw bet size (e.g., from human play or for approximate evaluation), `abstract_bet_size` maps it to the nearest abstract size:

```rust
pub fn abstract_bet_size(pot: u32, current_bet: u32, raw_bet: u32, street: u8, big_blind: u32) -> u32 {
    let (amounts, n) = abstract_raise_amounts(pot, current_bet, street, big_blind);
    if n == 0 { return raw_bet; }
    let mut best = amounts[0];
    let mut best_dist = (raw_bet as i64 - best as i64).unsigned_abs();
    for &a in &amounts[1..n] {
        let d = (raw_bet as i64 - a as i64).unsigned_abs();
        if d < best_dist { best_dist = d; best = a; }
    }
    best
}
```

This nearest-neighbor mapping is the standard approach in abstracted poker solvers. It allows the engine to handle off-abstraction bets gracefully when playing against opponents who don't follow the abstraction.

---

## 9. legal_actions: Combining Rules and Abstraction

`legal_actions` is where game rules and action abstraction meet. It constructs the list of valid abstract actions for the current player:

```rust
pub fn legal_actions(state: &GameState) -> Vec<Action> {
    let mut actions: Vec<Action> = Vec::with_capacity(8);

    let p = state.to_act as usize;
    let to_call = state.current_bet.saturating_sub(state.street_bets[p]);
    let max_bet = state.stacks[p] + state.street_bets[p];

    // Passive options (no bet to call → Check; bet to call → Fold + Call/AllIn)
    if to_call == 0 {
        actions.push(Action::Check);
    } else {
        actions.push(Action::Fold);
        if state.stacks[p] > to_call {
            actions.push(Action::Call);
        } else {
            actions.push(Action::AllIn);
            return actions;
        }
    }

    // Aggressive options (abstract raises + AllIn)
    let min_raise_total = state.current_bet + state.min_raise;
    let pot = state.pot();
    let (abstract_bets, n) = abstract_raise_amounts(pot, state.current_bet, state.street, state.big_blind);
    let mut allin_added = false;  // tracks whether AllIn has already been added

    for &bet_level in &abstract_bets[..n] {
        if bet_level < min_raise_total { continue; }
        if bet_level >= max_bet {
            actions.push(Action::AllIn);
            allin_added = true;
            break;
        }
        actions.push(Action::Raise(bet_level));
    }
    if !allin_added && max_bet >= min_raise_total {
        actions.push(Action::AllIn);
    }

    actions
}
```

`Vec::with_capacity(8)` avoids reallocations for the expected maximum of 6–7 actions (fold/check/call + 3–4 raises + all-in). In the benchmark's `fill_legal_actions` variant (used in the hot path), this becomes a stack-allocated array — eliminating even the initial allocation.

Key correctness points:
- `players_to_act` must be managed carefully to know when a betting round closes
- Minimum-raise enforcement prevents sub-minimum raises (which would allow micro-raises to keep action going cheaply)
- `saturating_sub` prevents underflow when `street_bets[p] > current_bet` (impossible in a valid state, but defensive)

---

## 10. Performance: Benchmarking Kuhn Poker Tree Traversal

### Why Kuhn Poker?

Before benchmarking on 6-max NLHE (which has an enormous tree), the benchmark validates the engine on **Kuhn Poker**: the simplest non-trivial poker variant, consisting of 3 cards (Jack, Queen, King), 2 players, and a single betting round. It has exactly 6 possible deals and a tree of a few hundred terminal nodes.

Kuhn Poker serves two purposes:
1. **Correctness validation**: The small tree makes it feasible to verify that terminal-node counts are correct by hand
2. **Throughput measurement**: Running 500,000 full traversals of all 6 Kuhn deals gives a clean traversal-rate measurement

The mapping from Kuhn Poker to the NLHE engine:
- 2 players, heads-up, button = player 0
- Starting stacks: 3 chips each; BB = 2 chips (leaving very little raising room)
- Each player's Kuhn card is duplicated into both hole-card slots (different suits) to satisfy the NLHE engine's 2-hole-card interface
- No board cards (river hand evaluation not needed for Kuhn)

### Recursive vs Explicit-Stack Traversal

The benchmark compares two traversal algorithms:

**Recursive DFS**:
```rust
fn traverse_recursive(state: &mut GameState) -> u64 {
    if state.is_terminal() { return 1; }

    let mut actions = [Action::Fold; MAX_ACTIONS];
    let n = fill_legal_actions(state, &mut actions);
    let mut count = 0u64;

    for i in 0..n {
        state.apply_action(actions[i]);
        count += traverse_recursive(state);
        state.undo_action();
    }
    count
}
```

**Explicit-stack DFS**:
```rust
fn traverse_explicit_stack(state: &mut GameState) -> u64 {
    let mut stack = [StackFrame::empty(); MAX_DEPTH];
    let mut depth = 0usize;
    let mut count = 0u64;

    stack[0].n_actions = fill_legal_actions(state, &mut stack[0].actions);
    stack[0].next_idx = 0;

    loop {
        let frame = &mut stack[depth];
        if frame.next_idx >= frame.n_actions {
            if depth == 0 { break; }
            depth -= 1;
            state.undo_action();
            continue;
        }
        let action = frame.actions[frame.next_idx];
        frame.next_idx += 1;
        state.apply_action(action);
        if state.is_terminal() {
            count += 1;
            state.undo_action();
        } else {
            depth += 1;
            stack[depth].n_actions = fill_legal_actions(state, &mut stack[depth].actions);
            stack[depth].next_idx = 0;
        }
    }
    count
}
```

Neither algorithm allocates on the heap in the hot path:
- `fill_legal_actions` writes into a caller-supplied `[Action; MAX_ACTIONS]` array (no `Vec`)
- The explicit-stack algorithm stores its `StackFrame` array on the system stack: `[StackFrame; MAX_DEPTH]` with `MAX_DEPTH = 64`

### Benchmark Results and Interpretation

On a modern laptop CPU (single thread, release build):

```
Terminal nodes per full traversal (6 Kuhn deals): 564

Recursive:      ~2,500,000 traversals/sec  (0.200s for 500,000 iters)
Explicit stack: ~2,300,000 traversals/sec  (0.217s for 500,000 iters)
Winner: Recursive  (1.09× faster than explicit stack)
```

The recursive version often wins by a small margin. Why?

1. **Compiler register allocation**: The Rust compiler can keep intermediate values (the loop counter, the current action index) in CPU registers across recursive calls because it can see the full call chain. The explicit stack must store these values in memory (the `StackFrame` array).

2. **Branch prediction**: The `for i in 0..n` loop in the recursive version has a tight, predictable structure. The explicit stack's `loop { if ... break; }` is harder for branch predictors to pattern-match.

3. **Stack depth**: Kuhn Poker games are at most ~10 actions deep, so OS call-stack overhead is minimal. For a much deeper NLHE tree (which can reach 30–50 plies), the explicit stack might win by avoiding OS stack frame overhead.

The key insight is that **neither method allocates heap memory**, so the choice becomes a CPU microarchitecture question rather than an algorithmic one. Both variants confirm the design goal: the engine core is fast enough to serve as the foundation for a high-throughput CFR solver.

At 2.5M full Kuhn traversals per second, with 564 terminal nodes per traversal, the engine handles approximately **1.4 billion node visits per second** on a single core. Scaling to a realistic 6-max NLHE tree with 500 terminal nodes per subtree and a CFR outer loop would still yield tens of millions of strategy updates per second — sufficient for meaningful training runs within a reasonable time budget.

---

## 11. Key Design Tradeoffs

### Algorithmic Evaluator vs Lookup Table

**Decision**: Use an algorithmic 7-card evaluator (21 × `evaluate_5` calls) rather than a pre-computed lookup table.

**Tradeoff**:
| Approach | Evaluation Speed | Memory | Startup Cost | Cache Impact |
|----------|----------------|--------|--------------|-------------|
| Algorithmic | ~200 ns/hand | ~0 KB | 0 ms | Minimal |
| 2+2 Lookup Table | ~5 ns/hand | ~128 MB | ~50 ms | Evicts CFR tables |

For a solver where showdowns only occur at terminal nodes (not in the inner loop), 200 ns is fine. The 128 MB lookup table would compete with CFR regret tables for L3 cache, potentially causing cache thrashing that slows the much more frequent `apply_action` calls.

### Pre-Dealt vs Lazily-Dealt Board

**Decision**: Pre-deal all 5 board cards at `GameState::new` rather than sampling them as streets advance.

**Tradeoff**:
- **Pro**: Enables Public Chance Sampling naturally; clean separation between the CFR sampling layer and the game engine
- **Con**: The engine can't represent "deal the turn card now" without external involvement; the caller must always provide all cards upfront

For the target use case (CFR with public chance sampling), this is unambiguously correct. For a use case like running a simulation where board cards should be randomly sampled mid-game, the caller would need to construct a new `GameState` rather than advancing the existing one.

### Full Snapshot Undo vs Delta Undo

**Decision**: `UndoRecord` stores a full copy of all mutable fields (~100 bytes) rather than a minimal delta (the specific fields that changed for each action type).

**Tradeoff**:
- **Full snapshot**: Restore is a single sequential memory copy; always correct; simple to maintain as fields are added
- **Delta undo**: Smaller record for simple actions (fold = 3 bytes); restore requires a conditional sequence of partial restores; fragile if logic changes

The snapshot approach costs roughly 100 bytes per undo record at depth up to ~50, totaling ~5 KB of undo data in a typical traversal path. At 2.5M traversals per second, this memory is constantly live in cache anyway. The simplicity and robustness of full-snapshot undo outweighs the marginal memory savings of delta undo.

### Bitmask State vs Enum Arrays

**Decision**: Use `u8` bitmasks for `folded` and `allin` rather than a `[PlayerStatus; 6]` array.

**Tradeoff**:
- **Bitmasks**: 2 bytes total; bitwise operations; non-obvious to read; limits to 8 players (fine for 6-max)
- **Enum array**: 6 bytes; readable; easy pattern matching; larger `UndoRecord`

The bitmask fits in a single register, and operations like `count_active()` can be implemented with a popcount instruction or a compact filter. For a struct that is copied into every `UndoRecord` (and thus copied 10–50 times per game), saving 4 bytes per field matters.

### Abstract Action Space vs Full NLHE Tree

**Decision**: Limit bet sizes to 3–4 abstract pot fractions per street rather than allowing arbitrary chip amounts.

**Tradeoff**:
- **Abstract**: Finite tree (CFR can converge); loses some strategic richness; requires careful calibration
- **Full NLHE**: Infinite tree (CFR cannot converge directly); requires continuous CFR extensions or discretization outside the engine

This is a fundamental decision in poker AI design, not a micro-optimization. The action abstraction introduces **abstraction error**: strategies learned in the abstract game may not be optimal in the full game. The calibration of bet sizes (especially the river overbet at 1.5× pot) attempts to minimize this error for the most strategically important spots.

---

## 12. Lessons for Other Performance-Critical Projects

The design patterns in `poker-core` generalize beyond poker:

**1. Pre-allocate everything that will be repeatedly reused.**  
`UndoStack::new()` is the only allocation-bearing constructor. All subsequent operations are allocation-free. This pattern — "allocate once in the constructor, reuse forever" — applies to any system with a hot loop.

**2. Measure early; don't guess.**  
The benchmark exists not as an afterthought but as a first-class citizen. Running it before building the CFR solver confirms the traversal rate is adequate before investing weeks in solver development. If the result had been 100K traversals/sec instead of 2.5M, a different engine architecture would have been required.

**3. Design for your actual use case, not the most general case.**  
The pre-dealt board cards, the abstract action space, the 6-player maximum — these constraints make the engine faster and simpler. They would be wrong for a full hand-history simulator or a UI-facing live game engine. Understanding your actual use case before designing the data model is the most impactful engineering decision.

**4. Encoding matters more than algorithm for hot-path data.**  
The card encoding (`rank * 4 + suit`), the bitmask player state, and the packed `u32` hand-strength value are all examples of choosing a data representation that makes common operations (extract rank, check if folded, compare hands) into single instructions. Algorithm selection matters, but for data accessed billions of times, the encoding dominates.

**5. Separate concerns at trait/module boundaries.**  
The game engine knows nothing about CFR, abstraction, or solving. The evaluator knows nothing about game state. The betting module knows nothing about legal actions. Each module can be tested, replaced, and optimized independently. The `LeafEvaluator` trait in `poker-ai` follows the same principle: you can swap a table-lookup evaluator for an MLP without touching the engine.

---

## 13. What Comes Next

`poker-core` is Phase 1 of a six-phase roadmap to build a competitive 6-max NLHE AI. The engine is the foundation; what makes a strong poker AI is everything built on top:

**Information Abstraction (Phase 2)**: Compress the astronomical number of possible hand/board combinations into a tractable set of "buckets" using Expected Hand Strength (EHS), EHS², and draw-potential features clustered with K-Means++. This is arguably the most important phase — coarse abstraction is a permanent ceiling on the bot's quality that no amount of solver iterations can overcome.

**Blueprint CFR Solver (Phase 3)**: Run Discounted CFR with optimistic updates and regret-based pruning across the 6-max game tree. Public chance sampling makes this feasible by fixing board cards and sampling player hands. The regret table uses a Structure-of-Arrays layout for cache efficiency.

**Evaluation Framework (Phase 4)**: Measure exploitability using Local Best Response (LBR) bounds, and estimate actual win rates using AIVAT variance reduction. Without rigorous evaluation, training improvements are invisible.

**Subgame Resolving (Phase 5)**: Instead of playing the blueprint strategy directly, use a depth-limited real-time solver to construct a locally optimal strategy conditioned on the observed history. Belief tracking (Bayesian updating of opponent hand distributions) provides the necessary context.

Each phase depends on `poker-core`'s performance. A traversal engine that bottlenecks at 100K nodes/sec would make Phases 3–5 impractical within a reasonable compute budget. At 1B+ nodes/sec, the bottleneck shifts to the regret table and abstraction quality — which is exactly where it should be.

---

## Summary

`poker-core` demonstrates what it looks like to design a performance-critical system with a clear contract and explicit tradeoffs:

- **Card encoding** as a single byte enables rank/suit extraction in one instruction
- **Algorithmic hand evaluation** avoids large lookup tables that would evict CFR working data from cache
- **Bitmask player state** packs 6 players' status into 2 bytes
- **Pre-dealt board cards** enable Public Chance Sampling without complicating the engine
- **Full-snapshot undo** makes tree traversal simple, correct, and fast
- **Abstract action space** makes the game tree finite while preserving strategic richness
- **Pre-allocated undo stack** eliminates heap allocation from the hot path entirely

The resulting engine traverses 1.4 billion Kuhn Poker nodes per second on a single core — fast enough that solver convergence time is bottlenecked by abstraction quality and CFR algorithm efficiency, not the game engine itself. That's the goal: infrastructure that disappears.

---

*This article covers the `poker-core` crate as implemented in this repository. The broader solver pipeline — DCFR, information abstraction, subgame resolving — is described in the companion docs and `poker-ai-plan-v2.md`.*
