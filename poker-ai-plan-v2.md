# Poker AI Implementation Plan v2 (2026)
### Post-Pluribus, Budget-Constrained, Theoretically Grounded

**Solo Developer · 14 Weeks · €400 Compute Budget**

---

## Table of Contents

1. [Guiding Philosophy](#guiding-philosophy)
2. [Memory Budget — Do This First](#memory-budget--do-this-first)
3. [Architecture Overview](#architecture-overview)
4. [Phase-by-Phase Plan](#phase-by-phase-plan)
5. [Repository Structure](#repository-structure)
6. [Budget Allocation](#budget-allocation)
7. [Honest Expectations](#honest-expectations)
8. [Reading List](#reading-list)

---

## Guiding Philosophy

You cannot out-scale Pluribus on €400. What you can do is replicate its *engineering discipline* at smaller scale:

- **You have two enemies, not one.** During training, variance kills convergence. At play time, abstraction error is a permanent ceiling on strategy quality. These enemies compete for budget: finer abstractions reduce play-time error but inflate memory and training time, increasing variance pressure. Every design decision is a tradeoff between them — make it explicitly.
- **Abstraction quality and solver correctness are co-dependent.** A mediocre solver on good abstractions outperforms a great solver on bad abstractions. But a buggy solver makes a good abstraction *look* bad, sending you on a wild goose chase re-clustering. Invest in abstraction, but only once your solver is verified correct on toy games.
- **Validate on small games first.** Kuhn Poker and Leduc Poker have known exact solutions. If your solver doesn't converge correctly there, it won't converge correctly anywhere.
- **Do not over-engineer.** Unnecessary complexity is fatal for a solo 14-week project. The architecture below is the minimum sufficient system, not a research lab codebase.

---

## Memory Budget — Do This First

Before committing to any abstraction or action abstraction parameters, compute your memory footprint. This determines whether your plan is feasible.

**Formula:**

```
total_memory = num_info_sets × avg_actions_per_set × 2 × 4 bytes
                                                     ↑   ↑
                                             (regret + strategy_sum) × f32
```

**Info set count drivers:**

| Factor | Multiplier |
|--------|-----------|
| Preflop buckets (per position) | 169 × 6 positions = 1,014 |
| Flop buckets | 500–800 |
| Turn buckets | 500–800 |
| River buckets | 1,000–1,500 |
| Bet sequences per street | Depends on action abstraction — see below |
| Active player configurations | Subsets of 6 players who haven't folded |

The combinatorial explosion across streets, bet sequences, positions, and active player configurations will produce tens to hundreds of millions of info sets. **Run this estimate for your specific action abstraction before buying RAM or finalizing bucket counts.**

**Decision gates:**

- If total fits in 64GB with headroom → proceed with tabular, buy RAM upgrade.
- If total is 64–128GB → reduce action abstraction first, then reduce bucket counts. Consider 128GB if budget allows.
- If total exceeds 128GB → you must either drastically coarsen the abstraction or introduce function approximation at resolving leaves (see Architecture section).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Runner / CLI                        │
│              (train | benchmark | play)                  │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │ poker-core  │  │ abstraction │  │  blueprint  │
   │             │  │             │  │   solver    │
   │ Game state  │  │ L2 K-Means │  │  DCFR +     │
   │ Hand eval   │  │ Bucket maps │  │  Optimistic │
   │ Action gen  │  │ EHS² feats  │  │  updates    │
   └─────────────┘  └─────────────┘  └──────┬──────┘
                                            │
                                     ┌──────▼──────┐
                                     │  resolving  │
                                     │             │
                                     │ Depth-ltd   │
                                     │ Belief-     │
                                     │ tracking    │
                                     │ Pluggable   │
                                     │ leaf eval   │
                                     └──────┬──────┘
                                            │
                                     ┌──────▼──────┐
                                     │ evaluation  │
                                     │             │
                                     │ Local BR    │
                                     │ AIVAT       │
                                     │ Self-play   │
                                     │ Exploitab.  │
                                     └─────────────┘
```

**Key design decisions:**

- **Pure tabular regret storage (f32, SoA layout) for the blueprint.** No neural components in the training loop. This is the debuggable choice.
- **Pluggable leaf evaluator at the resolving layer.** The default implementation returns blueprint table lookups. The interface accepts a second implementation backed by a simple MLP, to be trained *after* blueprint is verified, *only if* abstraction coarseness becomes the bottleneck. This is a trait boundary, not a neural network commitment — zero additional complexity until you choose to implement the second variant.
- Public chance sampling throughout blueprint training.
- Two crates only: `poker-core` and `poker-ai`. Minimal interface surface.

**On neural components:** The blueprint solver stays fully tabular. Neural nets are excluded from the training loop to keep debugging tractable: when a tabular solver produces a bad strategy, you can inspect exact regret values for the exact info set that's misbehaving. If resolving quality is later limited by coarse blueprint leaf values, a small MLP can be trained as a supervised regression on the blueprint's own values — not RL, not end-to-end, just fitting a function to known data. If the fit is bad, you inspect residuals by bucket, by street, by position. If resolving is worse with the net than without, you fall back to raw blueprint lookups. This is a Phase 5 decision, not a Phase 1 decision.

---

## Action Abstraction

This is equally important as card abstraction and determines your memory footprint.

**Blueprint bet sizes (fractions of pot):**

| Street | Bet sizes | Raise sizes |
|--------|----------|-------------|
| Preflop | Standard open raises (2.5bb), 3-bet (3×), 4-bet (2.5×) | — |
| Flop | 0.33, 0.67, 1.0 | 0.5, 1.0 of new pot |
| Turn | 0.5, 0.75, 1.0 | 0.67, 1.0 |
| River | 0.5, 0.75, 1.0, 1.5 (overbet) | 0.5, 1.0 |
| All streets | All-in always available | — |

**Why these specific sizes:**

- 3 bet sizes per street is the practical minimum for strategic diversity. Fewer than 3 and you can't distinguish between thin value bets and pot-building bets.
- The river gets 4 sizes because river play is where most exploitability lives and overbets are strategically important.
- More sizes = exponentially more bet sequences = exponentially more info sets. Adding a 4th flop bet size roughly doubles the tree size from flop onward.

**Each additional bet size approximately doubles the game tree from that street onward.** Tune this based on the memory budget calculation above.

The resolver (Phase 5) handles off-tree bet sizes. The blueprint action abstraction determines the skeleton the resolver builds on — with 3 bet sizes per street, the resolver has reasonable anchor points. With only 2, resolving degrades noticeably.

---

## Phase-by-Phase Plan

### Phase 1 — High-Performance Engine (Weeks 1–2)

**Goal:** Fast, correct game state management. Measure throughput on full game tree traversals, not just raw `apply_action` calls.

**Tasks:**

- Represent `GameState` as a packed struct. Use bitmasks for active players, folded players, and board cards.
- Implement `apply_action` and `undo_action` with no heap allocation. Use an undo stack, not cloning.
- Integrate an existing lookup-table hand evaluator. The Two Plus Two evaluator handles 7-card evaluation. Ensure you also have a plan for 5- and 6-card evaluation (flop/turn) during equity calculation in Phase 2 — you'll need Monte Carlo rollouts or enumeration there, not just final hand ranking.
- Write a benchmark binary that measures throughput as **full Kuhn Poker tree traversals per second**, not raw state transitions. Raw `apply_action` calls will always be fast — the real bottleneck in MCCFR is hand evaluation at terminals and regret table cache misses.

**What to avoid:**
- `Vec` allocations in the hot path
- `HashMap` for anything touched per-node

**On recursion vs explicit stack:** Don't assume an explicit stack is faster. Recursive traversal with the Rust compiler's optimizations often outperforms explicit stacks because the compiler can keep variables in registers across calls. An explicit stack forces manual pack/unpack of state, which can be slower and is more error-prone. **Profile both on your actual hardware before committing.** The "explicit stack is always faster" claim is a C-era heuristic that doesn't reliably hold with modern compilers and deep inlining.

**Deliverable:** CLI benchmark tool. Output: full tree traversals/second on Kuhn Poker.

---

### Phase 1.5 — Heads-Up Validation Pipeline (Week 2–3, overlapping)

**Goal:** A working heads-up (2-player) training and evaluation pipeline before tackling 6-player.

**Why this step matters:**

- In 2-player zero-sum games, exact best response is computationally feasible. You can measure true exploitability, not just the LBR lower bound.
- This separates abstraction errors from multiplayer convergence issues. If your 2-player solver produces high exploitability, the problem is in your solver or abstraction, not in the inherent non-convergence of multiplayer CFR.
- It's roughly one day of additional work: the solver, abstraction, and evaluation code are the same; you just set player count to 2 and use a simpler game tree.

**Tasks:**

- Train a heads-up NLHE blueprint using your Phase 2 abstraction and Phase 3 solver.
- Compute exact best response exploitability (feasible in 2-player).
- Compare your exploitability to known benchmarks (Slumbot's published numbers, if available).
- Fix any issues found before moving to 6-player.

**Deliverable:** Exploitability number for heads-up NLHE. Confidence that your solver and abstraction are correct before multiplayer complexity enters.

---

### Phase 2 — Information Abstraction (Weeks 3–4)

**Goal:** A serialized abstraction map that the solver loads at startup.

**The core concept:** Group similar information sets into buckets. The quality of this bucketing determines your ceiling more than anything else in the project.

**Feature engineering:**

Use potential-aware features, not just raw equity:
- `EHS` — Expected Hand Strength against random opponent hands
- `EHS²` — Second moment, captures variance in outcomes
- Draw potential — probability of improving to a strong hand

Note on blocker effects: Blockers (your holding removing cards from opponent ranges) are inherently combinatorial and hard to encode in a fixed-dimensional feature vector without it becoming very high-dimensional. Rather than a dedicated blocker feature, rely on the EHS histogram to implicitly capture some blocker information — hands with unusual equity distributions due to blocking effects will naturally cluster differently. If you want explicit blocker modeling, do it at resolving time (Phase 5), not in the static abstraction.

**Bucketing strategy:**

| Street   | Buckets | Notes |
|----------|---------|-------|
| Preflop  | 169 × 6 | Canonical hands **per position**. Collapsing positions loses one of the most important strategic dimensions in 6-max. |
| Flop     | 500–800 | K-Means++ with L2 distance on EHS histogram |
| Turn     | 500–800 | Re-cluster conditioning on flop bucket |
| River    | 1000–1500 | Highest fidelity. This is where exploitability leaks. |

Note: 500 flop buckets is quite coarse. Hands like middle pair with a flush draw and middle pair without one get merged despite playing very differently. If memory allows (check the budget calculation), push toward 800+ on the flop. Bucket counts are your primary tuning knob against the memory ceiling.

**Clustering distance metric:**

Use **L2 distance on discretized EHS histograms**, not raw EMD. K-Means with EMD doesn't have a closed-form centroid update — computing the EMD barycenter requires solving a linear program at each K-Means iteration, which is extremely slow. L2 on a fixed-bin histogram is a well-known proxy for EMD: it's fast, has a trivial centroid update (just average the histograms), and is empirically nearly as good for poker abstraction. If you find L2 clustering insufficient after evaluation, you can try Wasserstein distance with the Sinkhorn approximation, but start with L2.

**Implementation:**
- K-Means++ initialization (better than random K-Means)
- Discretize equity distributions into 50-bin histograms
- L2 distance between histograms as clustering metric
- Use `ndarray` for vector math
- Serialize the final map with `serde` + `bincode` for fast loading

**Deliverable:** `abstraction.bin` file. Validate: inspect bucket contents manually. Each bucket should contain hands that feel intuitively similar.

---

### Phase 3 — DCFR with Optimistic Updates (Weeks 5–9)

**Goal:** A converging blueprint strategy. Validate on small games before running on full NLHE.

**Timeline note:** This phase gets 5 weeks, not 4. Debugging multiplayer MCCFR is notoriously slow — bugs manifest as "convergence is weird" rather than crashes, and root-causing them requires patience. A solid blueprint with no resolving is more useful than a buggy blueprint with a resolving layer on top. The extra week is borrowed from Phase 5.

**Algorithm stack:**

1. **DCFR (Brown & Sandholm, 2019)** — your base. Apply discount `d(t) = t^α / (t^α + 1)` to cumulative regrets. Alpha ≈ 1.5 works well empirically. This suppresses early noise cheaply.

2. **Optimistic updates (Farina et al., 2021)** — add a momentum term to the regret update. Instead of `R_t = R_{t-1} + r_t`, use `R_t = R_{t-1} + 2*r_t - r_{t-1}`. This is a one-line change but accelerates last-iterate convergence. **Important caveat:** optimistic MCCFR has no convergence guarantees in multiplayer games. DCFR itself only provably converges in two-player zero-sum settings. In 6-player, you're targeting an approximate Nash, and convergence is empirical, not guaranteed. This means oscillating regrets in multiplayer might not always indicate a bug — they may be inherent. Your diagnostics need to account for this.

3. **Public chance sampling** — sample the public board cards once per iteration, then traverse all player paths given that board. This dramatically reduces multiplayer variance compared to outcome sampling.

   **Critical implementation detail:** "Traverse all player paths" does not mean enumerating all opponent hand combinations. In 6-player, the number of private hand combinations across 5 opponents is astronomical. You must subsample opponent hands. The standard approach is **external sampling**: for each player being updated, sample a single hand for each opponent and traverse only that combination. This introduces variance but is the only tractable option. Be aware that this makes your effective sampling scheme closer to "public chance + external" than pure public chance sampling.

4. **Regret-Based Pruning (RBP)** — after the first 20% of iterations, stop traversing branches whose cumulative regret is below threshold `θ` for `K` consecutive iterations. Start with θ and K as **configurable parameters**, not hardcoded constants. Run sensitivity analysis on Leduc before committing to values.

   **Interaction warning:** RBP and optimistic updates can interfere. Pruning a branch with negative regret that is about to receive a large optimistic correction permanently distorts the strategy. Monitor for branches that are pruned, then would have recovered if traversed. One practical safeguard: periodically (every N iterations) do a full unpruned traversal to check if pruned branches have become relevant.

**Memory layout:**

Use Structure of Arrays, not Array of Structures:

```rust
// GOOD - cache-friendly for regret updates
struct RegretTable {
    regrets: Vec<f32>,       // flat: [infoset_0_action_0, infoset_0_action_1, ...]
    strategy_sum: Vec<f32>,  // same layout
    num_actions: Vec<u8>,    // per info set
    offsets: Vec<u32>,       // start index per info set
}
```

Store as `f32`. If you hit memory limits, consider `bf16` (bfloat16), **not** IEEE `f16`. bf16 has the same exponent range as f32, so it handles the large magnitude range of cumulative regrets. IEEE f16 will overflow or underflow on cumulative regrets surprisingly fast due to its narrow exponent range.

**Validation protocol:**

Before running on full NLHE:
1. Implement Kuhn Poker (3 cards, 2 players, trivial rules)
2. Run solver until convergence
3. Compare output strategy to known exact solution
4. Exploitability should be <0.001 bb/hand
5. Repeat for Leduc Poker
6. Run heads-up NLHE (Phase 1.5) and verify exploitability

If your solver fails on any of these, debug before scaling up.

**Deliverable:** Convergence graph showing exploitability over iterations on Kuhn and Leduc. Heads-up NLHE exploitability number. Blueprint checkpoint file for 6-player NLHE after full training run.

---

### Phase 4 — Blueprint Stabilization (Weeks 10–11)

**Goal:** Verify your blueprint is actually converging before enabling resolving.

**Evaluation:**

Self-play win rate is insufficient. A strategy can win at self-play and be heavily exploitable. Use Local Best Response (LBR) to compute a lower bound on exploitability. LBR works by fixing your strategy and computing the best response on a sampled subset of the game tree — computationally tractable unlike exact best response.

**Understand LBR's limits:** LBR measures how much a *local, single-action* deviation can exploit your strategy. It doesn't detect coordinated multi-street exploits (e.g., an opponent who check-raises flop specifically to set up a river bluff). For your budget, LBR is the right pragmatic choice, but don't treat low LBR as proof of low exploitability.

**What to look for:**
- Regret per bucket. Any bucket with oscillating rather than decreasing regret indicates a leaky abstraction boundary — two dissimilar hands are being mapped to the same bucket and the solver can't find a stable strategy for both.
- Strategy stability between checkpoints. If your strategy at 100k iterations is drastically different from 80k iterations, you need more iterations before resolving.
- Sanity checks: preflop ranges should look recognizable. If your solver is 3-betting 72o at 40%, something is wrong.
- **Compare heads-up exploitability (exact BR) to heads-up LBR.** This calibrates how much LBR underestimates true exploitability in your specific abstraction.

**Deliverable:** LBR exploitability estimate. Strategy checkpoint used for resolving.

---

### Phase 5 — Depth-Limited Resolving with Belief Tracking (Weeks 12–13)

**Goal:** A bot that handles any bet size, including those not in the blueprint's action abstraction.

**What you're actually building:**

Depth-limited resolving with explicit belief tracking over opponent hands. The belief tracking is inspired by ReBeL's Public Belief State framework. However, full ReBeL uses a *learned value function* (trained via self-play RL) to evaluate subgame leaf nodes. You are replacing that with blueprint table lookups. This makes your system much closer to Pluribus-style resolving with the addition of explicit belief tracking — which is still a meaningful improvement, but be precise about what it is so you debug the right things.

The practical consequence: if resolving produces bad strategies, the first suspect should be **leaf value quality** (blueprint accuracy at subgame boundaries), not the belief tracking logic. This is also the point where the optional MLP leaf evaluator (see Architecture section) becomes relevant — if blueprint leaf values are too coarse, a learned approximation trained on the blueprint's own data can improve resolving without changing the blueprint at all.

**Implementation:**

1. Maintain a belief distribution over each opponent's hole cards given observed actions. Use **independent marginals per opponent** — the joint distribution over 5 opponents' holdings is combinatorially intractable. Independent marginals introduce correlation errors (e.g., if you hold A♠K♠, the marginals for opponent 1 and opponent 2 can both assign probability to holding A♠) but are the standard tractable approximation.
2. When entering Turn or River, spawn a subgame solver.
3. Solve the subgame using DCFR with the blueprint's values as leaf node estimates.
4. Depth limit: 1–2 streets is realistic on your compute budget.
5. Time budget: 2–5 seconds per resolving call.

**Make the leaf evaluator a pluggable trait:**

```rust
trait LeafEvaluator {
    fn evaluate(&self, state: &GameState, beliefs: &BeliefState) -> Vec<f64>;
}

struct BlueprintLeafEval { /* looks up blueprint table */ }
struct NeuralLeafEval { /* optional: small MLP inference */ }
```

Default to `BlueprintLeafEval`. Only implement `NeuralLeafEval` if evaluation in Phase 4 shows that coarse blueprint leaf values are the bottleneck.

**What to avoid:**
- Full subtree recomputation (too slow)
- Resolving on every street from the start (preflop resolving is expensive and rarely worth it at this scale)
- Calling this "ReBeL" in your documentation or mental model — it sets wrong expectations for debugging

**Stress tests:**
- Feed the bot unusual bet sizes (2.3x pot overbet, min-raise, all-in on flop)
- Verify strategy remains balanced and doesn't degenerate
- Check-raise lines and multiway pots are the most common failure modes

**Deliverable:** Bot that accepts arbitrary bet sizes and produces a coherent strategy within time budget.

---

### Phase 6 — Evaluation (Week 14)

**Goal:** Understand exactly how strong your bot is and where it leaks.

**Evaluation protocol:**

- **AIVAT variance-reduced evaluation** (Burch et al., 2018) for all win-rate measurements. Raw win rate in poker has enormous variance — even 10,000 hands gives very wide confidence intervals. AIVAT uses the known blueprint strategy as a baseline to dramatically tighten confidence intervals. Without it, tournament results between bot versions are mostly noise.
- Round-robin tournament between blueprint checkpoints (10k, 50k, 200k iterations), measured with AIVAT.
- LBR exploitability on river subgames specifically — this is where most exploitability lives.
- Forced stress scenarios: large overbets on wet boards, unusual 3-bet sizes, multiway all-ins.
- Head-to-head against a random policy (should win overwhelmingly) and against a tight-aggressive rule-based bot (should win comfortably).
- If available, evaluate against open-source bots with known strength (e.g., Slumbot API for heads-up) for external calibration.

**What not to do:**

Do not evaluate by win rate against yourself in self-play. It tells you almost nothing about exploitability.

**Deliverable:** AIVAT-adjusted win rates. LBR exploitability estimate. List of known weaknesses for future iteration.

---

## Repository Structure

```
poker-ai/
│
├── Cargo.toml                    # Workspace root
├── README.md
├── .gitignore
│
├── docs/
│   ├── architecture.md           # System design decisions
│   ├── memory-budget.md          # Info set count estimates, memory arithmetic
│   ├── abstraction.md            # Bucketing methodology and validation
│   ├── action-abstraction.md     # Bet sizing choices and rationale
│   ├── cfr-notes.md              # Algorithm notes and convergence analysis
│   └── experiments.md            # Training run logs and results
│
├── data/
│   ├── abstraction.bin           # Serialized abstraction map (generated)
│   └── blueprint/                # Blueprint checkpoints (generated)
│       ├── checkpoint_10k.bin
│       └── checkpoint_final.bin
│
├── crates/
│
│   ├── poker-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── state.rs           # GameState struct, packed representation
│   │       ├── action.rs          # Action enum, legal action generation
│   │       ├── betting.rs         # Pot geometry, bet sizing, action abstraction
│   │       ├── evaluator.rs       # Hand strength (wraps lookup table crate)
│   │       └── undo.rs            # Undo stack for zero-allocation traversal
│   │
│   └── poker-ai/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           │
│           ├── abstraction/
│           │   ├── mod.rs
│           │   ├── features.rs    # EHS, EHS², draw potential computation
│           │   ├── clustering.rs  # K-Means++ with L2 on EHS histograms
│           │   └── bucket_map.rs  # Info set → bucket lookup
│           │
│           ├── solver/
│           │   ├── mod.rs
│           │   ├── dcfr.rs        # DCFR update rule with discounting
│           │   ├── optimistic.rs  # Optimistic update (momentum term)
│           │   ├── sampling.rs    # Public chance + external sampling
│           │   ├── pruning.rs     # Regret-based pruning (configurable params)
│           │   └── regret_table.rs # SoA regret storage, f32
│           │
│           ├── resolving/
│           │   ├── mod.rs
│           │   ├── belief_state.rs # Per-opponent marginal hand distributions
│           │   ├── subgame.rs      # Subgame solver (depth-limited DCFR)
│           │   ├── leaf_eval.rs    # Pluggable trait: blueprint lookup or MLP
│           │   └── warm_start.rs   # Initialize from blueprint values
│           │
│           ├── evaluation/
│           │   ├── mod.rs
│           │   ├── local_br.rs     # Local Best Response exploitability bound
│           │   ├── aivat.rs        # AIVAT variance-reduced win rate estimation
│           │   ├── self_play.rs    # Head-to-head match runner
│           │   └── metrics.rs      # NPS counter, convergence tracking
│           │
│           └── bin/
│               ├── train.rs        # Blueprint training entrypoint
│               ├── benchmark.rs    # Tree traversal benchmark
│               ├── cluster.rs      # Run abstraction clustering
│               ├── memory_estimate.rs # Compute info set count for given params
│               └── play.rs         # Interactive CLI play
│
└── scripts/
    ├── plot_convergence.py         # Plot exploitability over iterations
    ├── analyze_buckets.py          # Inspect bucket contents
    └── run_tournament.sh           # Round-robin evaluation script
```

**Why two crates, not eight:**

`poker-core` is pure game logic with no solver dependency. Everything else lives in `poker-ai`. This boundary is the only one that matters for a solo developer — it lets you test game logic independently from solver logic. More crates would add interface overhead without benefit at this scale.

---

## Budget Allocation

| Item | Cost | Rationale |
|------|------|-----------|
| RAM upgrade (64GB or 128GB DDR5) | €150–200 | Tabular regret tables are memory-bound. **Run the memory estimate first** (see Memory Budget section) to determine whether 64GB suffices or 128GB is needed. |
| Cloud burst (Vast.ai or similar) | €100–150 | 1 week of high-core-count CPU for final deep training run. Shop for spot instances. |
| Miscellaneous (SSD space, backups) | €50 | Blueprint checkpoints are large. External drive or cloud storage. |
| Buffer | €50 | Unexpected cloud costs. |
| **Total** | **€400** | |

**Cloud burst strategy:** Do all development and validation locally. Only use cloud compute for the final full-scale training run once your solver is verified correct on small games. Running broken code on expensive hardware is the most common budget mistake.

---

## Honest Expectations

**What this system will likely achieve:**
- Solid, balanced preflop ranges
- Reasonable postflop play in common spots
- Handles bet sizes outside the blueprint via resolving
- Probably crushes strong amateurs consistently
- Competitive with many recreational and mid-stakes players

**What it will not achieve:**
- Pluribus-level exploitability (they ran orders of magnitude more iterations)
- Perfect multiway equilibrium (multiplayer CFR has no convergence guarantees — your approximate Nash is empirical, not provable)
- Reliable performance in rare, complex spots (limited by abstraction resolution)

**The honest gap:** The genuine algorithmic improvements since 2019 — DCFR, optimistic updates, public chance sampling, belief-tracking resolving — will save you roughly 30–50% of training compute compared to vanilla 2019 methods. That is meaningful. It does not close the gap between a 14-week solo project and a multi-year research team effort. Manage expectations accordingly.

---

## Reading List

### Essential — Read Before Writing Code

These are the papers that directly determine your architectural decisions. Read them in order.

**1. Libratus: The Superhuman AI for No-Limit Poker**
Brown & Sandholm, IJCAI 2017
The predecessor to Pluribus. Introduces the safe nested resolving concept and the endgame solving framework. Explains why naive depth-limited search is exploitable and what the fix looks like. Required for understanding Phase 5.

**2. Superhuman AI for Multiplayer Poker (Pluribus)**
Brown & Sandholm, Science 2019
The primary reference. Read the main paper and supplementary materials. Pay particular attention to: action abstraction strategy, blueprint training setup, depth-limited search with blueprint leaf values, and the discussion of multiplayer equilibrium approximation.

**3. Solving Imperfect-Information Games via Discounted Regret Minimization (DCFR)**
Brown & Sandholm, AAAI 2019
Your solver algorithm. Short and clear. The discounting schedule is the key practical contribution. Implement this exactly.

**4. Stable-Predictive Optimistic Counterfactual Regret Minimization**
Farina, Kroer, Brown & Sandholm, ICML 2021
The optimistic update that gives you faster last-iterate convergence. The implementation change over DCFR is small. Understand the theory well enough to know when it helps (two-player) vs when gains are more modest (multiplayer).

**5. ReBeL: Combining Deep Reinforcement Learning and Search for Imperfect-Information Games**
Brown, Bakhtin, Lerer & Gong, NeurIPS 2020
The theoretical framework for principled subgame solving via Public Belief States. You will not use the neural network components or the RL training loop. The value is in understanding *why* belief tracking makes subgame solving more principled than ad-hoc depth limiting, and where the guarantees break down without a learned value function.

---

### Important — Read During Implementation

**6. Monte Carlo Sampling for Regret Minimization in Extensive Games**
Lanctot et al., NeurIPS 2009
The foundational paper for MCCFR. Covers external sampling, outcome sampling, and chance sampling. You need to understand why public chance sampling reduces variance in multiplayer settings specifically, and how external sampling interacts with it.

**7. Variance Reduction in Monte Carlo Counterfactual Regret Minimization (VR-MCCFR)**
Schmid et al., AAAI 2019
Baseline subtraction for variance reduction. Explains the theory behind why baselines work for CFR and how to implement them. Directly improves your Phase 3 sampling efficiency.

**8. Regret-Based Pruning in Extensive-Form Games**
Brown & Sandholm, NIPS 2015
Your pruning strategy. Explains safe pruning thresholds and why you should delay pruning to the later phase of training. Note: parameter recommendations are for 2-player HULHE and may need adjustment for 6-player NLHE with different abstraction granularity.

**9. Potential-Aware Imperfect-Recall Abstractions with Earth Mover's Distance in Imperfect-Information Games**
Johanson et al., AAAI 2013
The theoretical basis for potential-aware features (EHS²) in your clustering. Explains why potential-aware abstraction is strictly better than current-street-only equity bucketing.

**10. AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games**
Burch, Johanson & Bowling, 2018
Your evaluation methodology. Dramatically tightens confidence intervals for win-rate estimation. Without this, tournament results between bot versions are mostly noise. Essential for Phase 6.

**11. Finding Optimal Abstract Strategies in Extensive-Form Games**
Johanson, Burch, Valenzano & Bowling, AAAI 2012
Addresses how abstraction pathologies can make your strategy *worse* as you add buckets. Directly relevant to your Phase 2 bucket count decisions — more buckets is not always better.

---

### Background — Read If You Have Time

**12. An Introduction to Counterfactual Regret Minimization**
Neller & Lanctot, 2013
A readable tutorial introduction to CFR. If you are not already comfortable with regret minimization, extensive-form games, and counterfactual values, read this before the papers above. Includes worked examples on Kuhn Poker.

**13. Abstracting Real-World Games**
Brown, Sandholm & Amos, NeurIPS 2015
Information abstraction theory. Explains how to think about the tradeoff between abstraction coarseness and strategy quality. Background for Phase 2 design decisions.

**14. Local Best Response**
Lisy & Bowling, IJCAI 2016 workshop
The evaluation methodology you will use in Phases 4 and 6. Explains how LBR gives you a practical exploitability lower bound without the exponential cost of exact best response computation.

---

### Reference — Keep Open While Coding

**15. OpenSpiel: A Framework for Reinforcement Learning in Games**
Lanctot et al., 2020
Not a paper to read cover-to-cover. Use it as a reference implementation. Their CFR code is clean and well-documented. When debugging your solver, compare behavior against OpenSpiel's CFR on small games.
https://github.com/google-deepmind/open_spiel

**16. Cepheus Poker Project Documentation**
University of Alberta GAMES Group
Reference for heads-up limit poker solving. Different game but the abstraction and CFR implementation details are useful as sanity-check reference.
http://poker.srv.ualberta.ca

---

### A Note on Sources

Do not rely on blog posts, AI-generated summaries, or YouTube explanations for implementation decisions. The papers above are short (most are under 15 pages), clearly written, and are the actual source of truth. When an AI assistant tells you a specific algorithm converges "3x faster" or a library is "the 2026 industry standard," ask for the citation. If there isn't one, treat the claim skeptically.
