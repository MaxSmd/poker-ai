# Poker AI Implementation Plan v3 (2026)
### Post-Pluribus, Budget-Constrained, Theoretically Grounded — Search-Weighted Revision

**Solo Developer · 14 Weeks · €400 Compute Budget**

---

## Changelog from v2

This revision keeps v2's engineering discipline and most of its structure. The changes are concentrated in the solver, the resolver, and how compute is allocated between the two.

1. **Variance reduction is now first-class.** Baseline-corrected sampling (VR-MCCFR) is promoted from the reading list into the core solver architecture. It was the single largest missing compute lever in v2.
2. **DCFR is now fully parameterized.** v2 specified only the α discount. v3 uses the full `(α, β, γ) = (1.5, 0, 2)` and is explicit about extracting the *average* strategy, not the last iterate.
3. **The resolver uses a predictive solver.** The Phase 5 subgame solver switches to Predictive RM⁺ / PCFR⁺, which is dramatically faster in the near-two-player, full-traversal regime that resolving actually operates in. The 6-player blueprint loop stays on sampled DCFR.
4. **Compute is rebalanced toward search.** Following the actual Pluribus lesson, the blueprint targets the *coarser* end of bucket counts so that more of the budget (RAM and cloud) goes to resolving quality. This is framed as a deliberate bet, with v2's blueprint-first conservatism preserved as the fallback.
5. **Phase 2 is cheaper.** Equity distributions are precomputed once and cached by canonical (suit-isomorphic) form; the hand evaluator is benchmarked rather than defaulted.
6. **The cloud burst is warm-started** from a local checkpoint, with an explicit lock-free parallelization strategy so the rented cores aren't idle on a memory-bound workload.
7. **Diagnostics account for imperfect recall**, and the CPU-over-GPU decision is made explicit rather than implicit.

---

## Table of Contents

1. [Guiding Philosophy](#guiding-philosophy)
2. [Memory Budget — Do This First](#memory-budget--do-this-first)
3. [Architecture Overview](#architecture-overview)
4. [Action Abstraction](#action-abstraction)
5. [Phase-by-Phase Plan](#phase-by-phase-plan)
6. [Repository Structure](#repository-structure)
7. [Budget Allocation](#budget-allocation)
8. [Honest Expectations](#honest-expectations)
9. [Reading List](#reading-list)

---

## Guiding Philosophy

You cannot out-scale Pluribus on €400. What you can do is replicate its *engineering discipline* at smaller scale:

- **You have two enemies, not one.** During training, variance kills convergence. At play time, abstraction error is a permanent ceiling on strategy quality. These enemies compete for budget: finer abstractions reduce play-time error but inflate memory and training time, increasing variance pressure. Every design decision is a tradeoff between them — make it explicitly.
- **Attack variance directly, not just by adding iterations.** Throwing more iterations at variance is the expensive way to buy convergence. Control variates (baselines) and the right sampling scheme buy the same convergence for less compute. On a fixed budget, a variance-reduction technique that you implement once is worth more than the cloud hours it saves over the whole project.
- **Search is leverage; the blueprint is the anchor.** Pluribus's decisive lever was real-time search, which let the blueprint stay coarse. A coarse blueprint trains faster, fits in less RAM, and carries less variance; a good resolver recovers quality at the leaves. v3 spends accordingly. The risk — that a weak resolver leaves you with only a coarse blueprint — is real, so the blueprint must still be *correct* before search is layered on.
- **Abstraction quality and solver correctness are co-dependent.** A mediocre solver on good abstractions outperforms a great solver on bad abstractions. But a buggy solver makes a good abstraction *look* bad, sending you on a wild goose chase re-clustering. Invest in abstraction, but only once your solver is verified correct on toy games.
- **Validate on small games first.** Kuhn Poker and Leduc Poker have known exact solutions. If your solver doesn't converge correctly there, it won't converge correctly anywhere.
- **Do not over-engineer.** Unnecessary complexity is fatal for a solo 14-week project. The architecture below is the minimum sufficient system, not a research lab codebase.

---

## Memory Budget — Do This First

Before committing to any abstraction or action abstraction parameters, compute your memory footprint. This determines whether your plan is feasible.

**Formula:**

```
total_memory = num_info_sets × avg_actions_per_set × N × 4 bytes
                                                     ↑
                          regret + strategy_sum + baseline accumulators (f32)
```

Note the change from v2: the per-info-set multiplier is no longer a flat `2`. Adding baseline-corrected sampling (see Phase 3) means storing one additional running accumulator per (info set, action) for the baseline value, so budget for `N = 3` f32 fields, not 2. This is a ~50% increase in the regret-store footprint and is part of why v3 targets coarser bucket counts.

**Info set count drivers:**

| Factor | Multiplier |
|--------|-----------|
| Preflop buckets (per position) | 169 × 6 positions = 1,014 |
| Flop buckets | 500–800 (target the low end; see philosophy) |
| Turn buckets | 500–800 |
| River buckets | 800–1,200 (v2 used 1,000–1,500) |
| Bet sequences per street | Depends on action abstraction — see below |
| Active player configurations | Subsets of 6 players who haven't folded |

The combinatorial explosion across streets, bet sequences, positions, and active player configurations will produce tens to hundreds of millions of info sets. **Run this estimate for your specific action abstraction before buying RAM or finalizing bucket counts.**

**Decision gates (revised toward a search-heavy allocation):**

- The default target is a blueprint that fits comfortably in **64GB** with the baseline accumulators included, *leaving the river at the coarse end and one fewer bet size where the tree is widest*. The resolver is expected to recover the resulting leaf-value coarseness.
- If your estimate only fits in 64–128GB, do **not** reach for 128GB of RAM first. First reduce action abstraction, then river buckets — every euro spent widening the blueprint is a euro not spent on resolving, which buys more strength per euro at this scale.
- Buy the 128GB upgrade only if (a) the 64GB-coarse blueprint validates as *correct but visibly leaky at the leaves in spots the resolver cannot reach*, and (b) the cloud-burst plan still has headroom. In practice this should be rare.
- If total exceeds 128GB even after coarsening, that is a signal your action abstraction is too rich for the budget, not that you need function approximation in the blueprint. Coarsen the tree.

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
   │ Game state  │  │ L2 K-Means │  │  DCFR (α,β,γ)│
   │ Hand eval   │  │ Bucket maps │  │  + optimistic│
   │ Action gen  │  │ EHS² feats  │  │  + baselines │
   └─────────────┘  │ Equity cache│  └──────┬──────┘
                    └─────────────┘         │
                                     ┌──────▼──────┐
                                     │  resolving  │
                                     │             │
                                     │ Depth-ltd   │
                                     │ Predictive  │
                                     │ RM⁺ / PCFR⁺ │
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

- **Pure tabular regret storage (f32, SoA layout) for the blueprint.** No neural components in the training loop. This is the debuggable choice and it does not change in v3.
- **Sampled DCFR with baselines for the blueprint; predictive RM⁺ for the resolver.** These are two different regret minimizers used in two different regimes, deliberately. The blueprint loop is high-variance, sampled, and genuinely 6-player, where sampled DCFR's empirical robustness matters and where predictive methods carry no guarantee. The resolver is a small, depth-limited, near-two-player, full-traversal problem solved against a wall-clock budget, which is exactly where predictive RM⁺ / PCFR⁺ converges fastest. Do not share one solver implementation across both regimes just because the update rules rhyme.
- **Pluggable leaf evaluator at the resolving layer.** The default implementation returns blueprint table lookups. The interface accepts a second implementation backed by a simple MLP, to be trained *after* blueprint is verified, *only if* abstraction coarseness becomes the bottleneck. This is a trait boundary, not a neural network commitment.
- Public chance sampling plus external sampling for opponent hands, with baseline correction layered on top.
- Two crates only: `poker-core` and `poker-ai`. Minimal interface surface.

**On the CPU/GPU question (now explicit):** The blueprint CFR loop stays on CPU, and the cloud burst should rent high-core-count CPU, not a GPU. The regret store is large and randomly accessed — it will not fit in VRAM and its access pattern is the worst case for a GPU. The only components that would benefit from a GPU are the *optional* MLP leaf evaluator and, marginally, K-Means clustering in Phase 2; neither is on the critical path, so neither justifies a GPU rental. If you ever train the MLP leaf evaluator, do it on a short, separate GPU spot rental, not by moving the main pipeline.

**On neural components:** unchanged from v2. The blueprint solver stays fully tabular so that a bad strategy can be traced to exact regret values at an exact info set. The optional MLP leaf evaluator (Phase 5) is a supervised regression onto the blueprint's own values — not RL, not end-to-end — and is a Phase 5 decision, not a Phase 1 one. If resolving is worse with the net than without, fall back to raw blueprint lookups.

---

## Action Abstraction

This is equally important as card abstraction and determines your memory footprint.

**Blueprint bet sizes (fractions of pot):**

| Street | Bet sizes | Raise sizes |
|--------|----------|-------------|
| Preflop | Standard open raises (2.5bb), 3-bet (3×), 4-bet (2.5×) | — |
| Flop | 0.33, 0.67, 1.0 | 0.5, 1.0 of new pot |
| Turn | 0.5, 1.0 | 0.67, 1.0 |
| River | 0.5, 0.75, 1.0, 1.5 (overbet) | 0.5, 1.0 |
| All streets | All-in always available | — |

**What changed from v2 and why:** v2 used three bet sizes on the turn (0.5, 0.75, 1.0). v3 drops the turn to two (0.5, 1.0). The justification is the rebalance toward search: with a *predictive* resolver and explicit belief tracking, off-tree turn sizings are recovered at play time rather than at training time, and the turn is the street where trimming a size buys the most memory back (it multiplies through both turn and river). The river keeps four sizes — river exploitability is the most expensive to leak and the resolver's belief tracking is least reliable there because there is no future street to average over.

**Why these specific sizes:**

- Three sizes is the strategic minimum on streets the resolver leans on least; two is acceptable on the turn specifically because the resolver anchors well there from both the flop and river skeletons around it.
- The river gets four sizes because river play is where most exploitability lives and overbets are strategically important.
- **Each additional bet size approximately doubles the game tree from that street onward.** Tune this against the memory budget calculation above, and remember that with the baseline accumulators the per-info-set cost is higher than in v2.

The resolver (Phase 5) handles off-tree bet sizes. With a predictive subgame solver and three-or-more anchor sizes on flop and river, the resolver has solid anchor points; the two-size turn is the one place to watch in stress testing.

---

## Phase-by-Phase Plan

### Phase 1 — High-Performance Engine (Weeks 1–2)

**Goal:** Fast, correct game state management. Measure throughput on full game tree traversals, not just raw `apply_action` calls.

**Tasks:**

- Represent `GameState` as a packed struct. Use bitmasks for active players, folded players, and board cards.
- Implement `apply_action` and `undo_action` with no heap allocation. Use an undo stack, not cloning.
- **Choose the hand evaluator by benchmark, not reputation.** The Two Plus Two evaluator is correct but is a ~124MB random-access lookup table that thrashes cache inside tight rollout loops. Benchmark it against a perfect-hash evaluator (e.g. `phevaluator` / an OMPEval-style hasher) *inside your actual equity-rollout loop*, not on isolated 7-card lookups. The smaller table usually wins on locality, and Phase 2 runs this loop millions of times. Whatever you pick, ensure you have 5- and 6-card evaluation for flop/turn equity (Monte Carlo rollouts or enumeration), not just final 7-card ranking.
- Write a benchmark binary that measures throughput as **full Kuhn Poker tree traversals per second**, not raw state transitions. The real bottleneck in MCCFR is hand evaluation at terminals and regret-table cache misses.

**What to avoid:**
- `Vec` allocations in the hot path
- `HashMap` for anything touched per-node

**On recursion vs explicit stack:** Don't assume an explicit stack is faster. Recursive traversal with the Rust compiler's optimizations often outperforms explicit stacks because the compiler can keep variables in registers across calls. **Profile both on your actual hardware before committing.** The "explicit stack is always faster" claim is a C-era heuristic that doesn't reliably hold with modern compilers and deep inlining.

**Deliverable:** CLI benchmark tool. Output: full tree traversals/second on Kuhn Poker, plus an evaluator micro-benchmark comparing the two evaluator candidates in-loop.

---

### Phase 1.5 — Heads-Up Validation Pipeline (Week 2–3, overlapping)

**Goal:** A working heads-up (2-player) training and evaluation pipeline before tackling 6-player.

**Why this step matters:**

- In 2-player zero-sum games, exact best response is computationally feasible. You can measure true exploitability, not just the LBR lower bound.
- This separates abstraction errors from multiplayer convergence issues.
- It's roughly one day of additional work: the solver, abstraction, and evaluation code are the same; you just set player count to 2 and use a simpler game tree.
- **Bonus in v3:** heads-up is also the cleanest place to sanity-check the *predictive* resolver against the sampled DCFR blueprint, since predictive RM⁺'s guarantees are strongest in exactly this 2p0s setting. If predictive resolving doesn't beat blueprint-only play heads-up, it won't in 6-max either.

**Tasks:**

- Train a heads-up NLHE blueprint using your Phase 2 abstraction and Phase 3 solver.
- Compute exact best response exploitability (feasible in 2-player).
- Compare your exploitability to known benchmarks (Slumbot's published numbers, if available).
- Fix any issues found before moving to 6-player.

**Deliverable:** Exploitability number for heads-up NLHE. Confidence that solver, abstraction, and resolver are correct before multiplayer complexity enters.

---

### Phase 2 — Information Abstraction (Weeks 3–4)

**Goal:** A serialized abstraction map that the solver loads at startup, built on a cached equity layer that is computed exactly once.

**The core concept:** Group similar information sets into buckets. The quality of this bucketing determines your ceiling more than anything else in the project.

**Equity precomputation and caching (new in v3):** Equity-distribution computation over millions of boards is your most compute-bound non-CFR step and it is embarrassingly parallel. Do it once:

- Canonicalize every (hole cards, board) by suit isomorphism so that strategically identical situations map to a single canonical key.
- Compute the equity histogram for each canonical key once and serialize it to disk.
- Re-clustering passes (turn conditioned on flop, river) read from this cache instead of recomputing rollouts. This turns your most expensive offline phase into a one-time cost and makes the cloud burst cheaper because clustering is no longer rollout-bound.

**Feature engineering:**

Use potential-aware features, not just raw equity:
- `EHS` — Expected Hand Strength against random opponent hands
- `EHS²` — Second moment, captures variance in outcomes
- Draw potential — probability of improving to a strong hand

Cluster on the *discretized equity-distribution histogram*, which captures these features implicitly; EHS/EHS² are useful as inspection diagnostics and as a fallback low-dimensional feature set.

Note on blocker effects: Blockers are inherently combinatorial and hard to encode in a fixed-dimensional feature vector without it becoming very high-dimensional. Rely on the EHS histogram to implicitly capture some blocker information — hands with unusual equity distributions due to blocking effects cluster differently. If you want explicit blocker modeling, do it at resolving time (Phase 5), not in the static abstraction.

**Bucketing strategy:**

| Street   | Buckets | Notes |
|----------|---------|-------|
| Preflop  | 169 × 6 | Canonical hands **per position**. Collapsing positions loses one of the most important strategic dimensions in 6-max. |
| Flop     | 500–800 | K-Means++ with L2 distance on EHS histogram. Target the lower end first; let the resolver carry the difference. |
| Turn     | 500–800 | Re-cluster conditioning on flop bucket (imperfect recall — see Phase 4 caveat). |
| River    | 800–1200 | Highest fidelity. This is where exploitability leaks. v2's 1500 ceiling is relaxed to fund search. |

The river is the one place where, if a 64GB-coarse blueprint validates as leaky in spots the resolver can't reach, you spend recovered budget *before* widening any other street.

**Clustering distance metric:**

Use **L2 distance on discretized EHS histograms**, not raw EMD. K-Means with EMD has no closed-form centroid update — the EMD barycenter requires solving a linear program at each iteration, which is extremely slow. L2 on a fixed-bin histogram is a well-known proxy for EMD: fast, trivial centroid update (average the histograms), and empirically nearly as good for poker abstraction. If L2 proves insufficient after evaluation, try Wasserstein with the Sinkhorn approximation, but start with L2.

**Implementation:**
- K-Means++ initialization
- Discretize equity distributions into 50-bin histograms
- L2 distance between histograms as clustering metric
- Use `ndarray` for vector math
- Serialize the final map with `serde` + `bincode` for fast loading
- Serialize the canonical equity cache separately so it survives re-clustering experiments

**Deliverable:** `abstraction.bin` file plus a serialized `equity_cache`. Validate: inspect bucket contents manually. Each bucket should contain hands that feel intuitively similar.

---

### Phase 3 — DCFR with Baselines and Optimistic Updates (Weeks 5–9)

**Goal:** A converging blueprint strategy. Validate on small games before running on full NLHE.

**Timeline note:** This phase gets 5 weeks, not 4. Debugging multiplayer MCCFR is notoriously slow — bugs manifest as "convergence is weird" rather than crashes. The coarser v3 blueprint shortens the *final training run*, which is part of how the budget is freed for search, but it does not shorten *debugging*, which is why the five weeks stay. The extra week is still borrowed from Phase 5; v3 partly repays Phase 5 with a faster solver rather than more calendar time.

**Algorithm stack:**

1. **DCFR (Brown & Sandholm, 2019), fully parameterized.** Use the authors' recommended `(α, β, γ) = (1.5, 0, 2)`, not α alone:
   - `α = 1.5` discounts cumulative *positive* regret, suppressing early noise.
   - `β = 0` applies a constant 0.5 weight to accumulated *negative* regret, which speeds escaping actions that early noise made look bad.
   - `γ = 2` weights the *strategy-sum* accumulation toward later iterations. This is the parameter v2 omitted, and it is the one that most directly improves the strategy you actually deploy, because **the blueprint is the time-averaged strategy, not the last iterate.** Make sure your checkpoint extraction computes the γ-weighted average correctly.

2. **Baseline-corrected sampling — VR-MCCFR (Schmid et al., 2019).** This is the headline v3 addition and the largest compute lever you have. Maintain a running baseline value per (info set, action) and subtract it from each sampled counterfactual value as a control variate, adding back its expectation so the estimator stays unbiased. Variance — your named convergence enemy in multiplayer — drops, and fewer iterations to a target exploitability is a direct cloud-cost saving. The simplest effective baseline is a running average of observed counterfactual values per (info set, action), stored as the third f32 accumulator budgeted in the Memory section. It composes with external sampling. Measure its effect on Leduc before the full run so you know the iteration budget it buys.

3. **Optimistic updates (Farina et al., 2021)** — add a momentum term: instead of `R_t = R_{t-1} + r_t`, use `R_t = R_{t-1} + 2·r_t − r_{t-1}`. One-line change, accelerates last-iterate convergence. **Caveat, sharpened in v3:** the benefit accrues mainly to the *last* iterate, but you deploy the *average* (γ-weighted) strategy, so the practical gain to the blueprint is real but smaller than the headline. And optimistic MCCFR has no convergence guarantee in multiplayer — DCFR itself only provably converges in 2p0s. In 6-player you target an approximate Nash empirically; oscillating regrets may be inherent rather than a bug.

4. **Public chance sampling + external sampling.** Sample the public board once per iteration, then for each updated player sample a single hand per opponent and traverse that combination. Enumerating all opponent hand combinations across 5 opponents is intractable; external sampling is the only tractable option and your effective scheme is "public chance + external," with baselines layered on to tame the resulting variance.

5. **Regret-Based Pruning (RBP).** After the first 20% of iterations, stop traversing branches whose cumulative regret is below threshold `θ` for `K` consecutive iterations. Keep θ and K as **configurable parameters**; run sensitivity analysis on Leduc before committing.

   **Interaction warnings (now three-way):** RBP interacts with both optimistic updates *and* baselines. Pruning a branch about to receive a large optimistic correction permanently distorts the strategy; and a pruned branch stops updating its baseline, so when it is re-expanded its control variate is stale and briefly *increases* variance. Safeguard: periodically (every N iterations) do a full unpruned traversal to refresh baselines and check whether pruned branches have become relevant.

**Memory layout:**

Structure of Arrays, not Array of Structures, now with the baseline accumulator:

```rust
// cache-friendly for regret updates
struct RegretTable {
    regrets: Vec<f32>,        // flat: [infoset_0_action_0, ...]
    strategy_sum: Vec<f32>,   // same layout, γ-weighted accumulation
    baseline: Vec<f32>,       // running baseline per (infoset, action)
    num_actions: Vec<u8>,     // per info set
    offsets: Vec<u32>,        // start index per info set
}
```

Store as `f32`. If you hit memory limits, consider `bf16`, **not** IEEE `f16`: bf16 keeps f32's exponent range and so handles the large magnitude range of cumulative regrets, whereas f16's narrow exponent range overflows or underflows on them surprisingly fast. Note that the baseline array makes the bf16 fallback more attractive than in v2, since it is the third large array.

**Validation protocol:**

Before running on full NLHE:
1. Implement Kuhn Poker; run to convergence; compare to the known exact solution. Exploitability < 0.001 bb/hand.
2. Repeat for Leduc Poker.
3. Run a baseline-on vs baseline-off comparison on Leduc and record the iteration-count difference — this is your evidence the control variate is wired correctly and your basis for sizing the cloud run.
4. Run heads-up NLHE (Phase 1.5) and verify exploitability against exact best response.

If your solver fails any of these, debug before scaling up.

**Deliverable:** Convergence graphs (exploitability over iterations) on Kuhn and Leduc, with and without baselines. Heads-up NLHE exploitability number. Blueprint checkpoint for 6-player NLHE after the full training run.

---

### Phase 4 — Blueprint Stabilization (Weeks 10–11)

**Goal:** Verify your blueprint is actually converging before enabling resolving.

**Evaluation:**

Self-play win rate is insufficient — a strategy can win at self-play and be heavily exploitable. Use Local Best Response (LBR) for a lower bound on exploitability; it fixes your strategy and computes the best response on a sampled subset of the tree.

**Understand LBR's limits:** LBR measures a *local, single-action* deviation. It does not detect coordinated multi-street exploits (e.g. a check-raise flop that sets up a river bluff). For your budget LBR is the right pragmatic choice, but low LBR is not proof of low exploitability.

**What to look for:**
- **Regret per bucket — but read it correctly.** Your turn buckets are re-clustered conditional on the flop bucket, which makes this an *imperfect-recall* abstraction. CFR on imperfect-recall abstractions can cycle or fail to converge even in 2p0s (Lanctot, Waugh et al.). So a bucket with oscillating rather than decreasing regret may be a consequence of the abstraction *structure*, not a leaky boundary and not a solver bug. Before you go re-clustering on a false alarm, check whether the oscillation correlates with imperfect-recall boundaries. Combined with the genuine multiplayer non-convergence noted in Phase 3, "oscillating regret" now has three possible causes — abstraction leakage, imperfect recall, and multiplayer inherent non-convergence — and your diagnosis has to distinguish them.
- Strategy stability between checkpoints. If 100k iterations differs drastically from 80k, you need more iterations before resolving.
- Sanity checks: preflop ranges should look recognizable. If the solver is 3-betting 72o at 40%, something is wrong.
- **Compare heads-up exploitability (exact BR) to heads-up LBR** to calibrate how much LBR underestimates true exploitability in your abstraction.

**Deliverable:** LBR exploitability estimate. Strategy checkpoint used for resolving.

---

### Phase 5 — Depth-Limited Resolving with a Predictive Subgame Solver (Weeks 12–13)

**Goal:** A bot that handles any bet size, including those not in the blueprint's action abstraction, and that gets more solve-quality per second than v2's plan would have.

**What you're actually building:**

Depth-limited resolving with explicit belief tracking over opponent hands, inspired by ReBeL's Public Belief State framework but with blueprint table lookups in place of ReBeL's learned value function. This is closer to Pluribus-style resolving plus explicit belief tracking. Be precise about what it is so you debug the right things: if resolving produces bad strategies, the first suspect is **leaf value quality** (blueprint accuracy at subgame boundaries), not the belief logic.

**The solver change (v3):** the subgame solver is **Predictive RM⁺ / PCFR⁺**, not the sampled DCFR loop reused from the blueprint. In near-two-player, full-traversal, depth-limited problems — which is what a subgame is once folds collapse the active set — predictive regret matching converges far faster than DCFR or CFR⁺ on most benchmarks, so within a 2–5s wall-clock budget you reach a substantially better strategy. This is the other half of the rebalance: a coarse blueprint is affordable precisely because the resolver is now strong and fast.

**Caveats to validate:**
- Predictive RM⁺'s strong guarantees are a 2p0s result. **Multiway subgames** (several opponents still in) erode them. Validate predictive resolving specifically on multiway pots; if it degrades, fall back to DCFR for multiway subgames and reserve predictive RM⁺ for heads-up and near-heads-up spots. This per-situation fallback is cheap because both solvers consume the same subgame tree.
- Keep the predictive solver isolated in the resolver. Do not let it leak into the blueprint loop, where it has no guarantee and where sampling variance, not solver speed, is the binding constraint.

**Implementation:**

1. Maintain a belief distribution over each opponent's hole cards given observed actions. Use **independent marginals per opponent** — the joint over 5 opponents is intractable. Marginals introduce correlation errors (e.g. holding A♠K♠, the marginals for two opponents can both put mass on A♠) but are the standard tractable approximation.
2. On entering Turn or River, spawn a subgame solver.
3. Solve the subgame with predictive RM⁺, using the blueprint's values as leaf estimates.
4. Depth limit: 1–2 streets is realistic on your compute budget.
5. Time budget: 2–5 seconds per resolving call. Predictive convergence is what makes this budget buy a good strategy rather than a half-solved one.

**Make the leaf evaluator a pluggable trait:**

```rust
trait LeafEvaluator {
    fn evaluate(&self, state: &GameState, beliefs: &BeliefState) -> Vec<f64>;
}

struct BlueprintLeafEval { /* looks up blueprint table */ }
struct NeuralLeafEval { /* optional: small MLP inference */ }
```

Default to `BlueprintLeafEval`. Because v3's blueprint is deliberately coarser, the odds that coarse leaf values become the bottleneck are higher than in v2 — so the `NeuralLeafEval` option is more likely to earn its place here. Train it (if at all) as supervised regression on the blueprint's own values, on a short separate GPU rental, and keep the blueprint-lookup fallback wired in.

**What to avoid:**
- Full subtree recomputation (too slow)
- Resolving on every street from preflop (expensive, rarely worth it at this scale)
- Calling this "ReBeL" in documentation or your mental model — it sets wrong debugging expectations

**Stress tests:**
- Unusual bet sizes (2.3× pot overbet, min-raise, all-in on flop), and specifically the **two-size turn** introduced in v3's action abstraction — verify the resolver recovers off-tree turn sizings cleanly.
- Verify strategy stays balanced and doesn't degenerate.
- Check-raise lines and multiway pots are the most common failure modes — and multiway is also where the predictive fallback matters.

**Deliverable:** Bot that accepts arbitrary bet sizes and produces a coherent strategy within budget, with a recorded comparison of predictive vs DCFR subgame solving on both heads-up and multiway spots.

---

### Phase 6 — Evaluation (Week 14)

**Goal:** Understand exactly how strong your bot is and where it leaks.

**Evaluation protocol:**

- **AIVAT variance-reduced evaluation** (Burch et al., 2018) for all win-rate measurements. Raw poker win rate has enormous variance; even 10,000 hands gives very wide intervals. AIVAT uses the known blueprint as a baseline to tighten them. Without it, tournament results between bot versions are mostly noise.
- Round-robin tournament between blueprint checkpoints (10k, 50k, 200k iterations), measured with AIVAT.
- LBR exploitability on river subgames specifically — where most exploitability lives.
- A specific A/B that v3 makes worth running: **blueprint-only vs blueprint+predictive-resolving**, measured with AIVAT. This quantifies how much the search-heavy bet actually bought, and tells you whether the coarse blueprint was the right call or whether budget should have gone to RAM after all.
- Forced stress scenarios: large overbets on wet boards, unusual 3-bet sizes, multiway all-ins.
- Head-to-head against a random policy (should win overwhelmingly) and a tight-aggressive rule-based bot (should win comfortably).
- If available, evaluate against open-source bots with known strength (e.g. Slumbot API for heads-up) for external calibration.

**What not to do:** Do not evaluate by win rate against yourself in self-play. It tells you almost nothing about exploitability.

**Deliverable:** AIVAT-adjusted win rates, including the blueprint-only vs resolving comparison. LBR exploitability estimate. List of known weaknesses for future iteration.

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
│   ├── memory-budget.md          # Info set counts, 3-field accumulator arithmetic
│   ├── abstraction.md            # Bucketing methodology and validation
│   ├── action-abstraction.md     # Bet sizing choices and rationale
│   ├── cfr-notes.md              # DCFR (α,β,γ), baselines, predictive resolver
│   └── experiments.md            # Training run logs and results
│
├── data/
│   ├── abstraction.bin           # Serialized abstraction map (generated)
│   ├── equity_cache.bin          # Canonical equity histograms (generated, reused)
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
│   │       ├── evaluator.rs       # Hand strength (benchmarked: phevaluator vs 2+2)
│   │       └── undo.rs            # Undo stack for zero-allocation traversal
│   │
│   └── poker-ai/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           │
│           ├── abstraction/
│           │   ├── mod.rs
│           │   ├── features.rs    # EHS, EHS², draw potential
│           │   ├── equity_cache.rs # Canonical (isomorphic) equity precompute + cache
│           │   ├── clustering.rs  # K-Means++ with L2 on EHS histograms
│           │   └── bucket_map.rs  # Info set → bucket lookup
│           │
│           ├── solver/
│           │   ├── mod.rs
│           │   ├── dcfr.rs        # DCFR update rule, full (α,β,γ)
│           │   ├── optimistic.rs  # Optimistic momentum term
│           │   ├── sampling.rs    # Public chance + external sampling
│           │   ├── baseline.rs    # VR-MCCFR control variate (running baseline)
│           │   ├── pruning.rs     # Regret-based pruning (configurable params)
│           │   └── regret_table.rs # SoA: regret + strategy_sum + baseline, f32
│           │
│           ├── resolving/
│           │   ├── mod.rs
│           │   ├── belief_state.rs # Per-opponent marginal hand distributions
│           │   ├── subgame.rs      # Depth-limited subgame solver: predictive RM⁺
│           │   ├── predictive.rs   # Predictive RM⁺ / PCFR⁺ regret minimizer
│           │   ├── leaf_eval.rs    # Pluggable trait: blueprint lookup or MLP
│           │   └── warm_start.rs   # Initialize from blueprint values
│           │
│           ├── evaluation/
│           │   ├── mod.rs
│           │   ├── local_br.rs     # Local Best Response bound
│           │   ├── aivat.rs        # AIVAT variance-reduced win rate
│           │   ├── self_play.rs    # Head-to-head match runner
│           │   └── metrics.rs      # NPS counter, convergence tracking
│           │
│           └── bin/
│               ├── train.rs        # Blueprint training entrypoint
│               ├── benchmark.rs    # Tree traversal + evaluator benchmark
│               ├── cluster.rs      # Run abstraction clustering (reads equity cache)
│               ├── memory_estimate.rs # Info set count for given params
│               └── play.rs         # Interactive CLI play
│
└── scripts/
    ├── plot_convergence.py         # Exploitability over iterations (baseline on/off)
    ├── analyze_buckets.py          # Inspect bucket contents
    └── run_tournament.sh           # Round-robin evaluation script
```

**Why two crates, not eight:** `poker-core` is pure game logic with no solver dependency. Everything else lives in `poker-ai`. This boundary is the only one that matters for a solo developer — it lets you test game logic independently from solver logic. The new modules (`baseline.rs`, `equity_cache.rs`, `predictive.rs`) sit inside the existing crate boundary and add no new interface surface between crates.

---

## Budget Allocation

| Item | Cost | Rationale |
|------|------|-----------|
| RAM (64GB DDR5, default) | €90–130 | The v3 blueprint targets a coarse footprint that fits 64GB *including* baseline accumulators. **Run the memory estimate first.** 128GB is now the exception, not a coin-flip. |
| Cloud burst (Vast.ai or similar) | €150–200 | High-core-count CPU for the final training run, **warm-started** from a local checkpoint. The euros freed by not buying 128GB go here — more burst hours and/or a short separate GPU spot for the optional MLP leaf evaluator. Shop spot instances. |
| Miscellaneous (SSD space, backups) | €50 | Blueprint checkpoints and the equity cache are large. External drive or cloud storage. |
| Buffer | €30–50 | Unexpected cloud costs. |
| **Total** | **~€400** | |

**Cloud burst strategy (revised):**

- Do all development and validation locally. Only use cloud compute once your solver is verified correct on small games and baselines are confirmed wired correctly.
- **Warm-start the burst.** Train a coarse blueprint locally and use its checkpoint to initialize the cloud run's regret and strategy-sum tables. You are then paying cloud rates for refinement, not for cold-starting through the high-variance early iterations that baselines and discounting are designed to suppress on cheap hardware anyway.
- **Parallelize explicitly.** On a memory-bound SoA table, naive mutex locking leaves most rented cores idle. Use lock-free atomic updates to the regret/strategy/baseline arrays (CFR tolerates the occasional benign race) or per-thread accumulation with periodic merges. Decide which *before* you rent the machine and benchmark it locally first.
- Running broken code on expensive hardware is still the most common budget mistake. Warm-starting makes a broken cloud run cheaper to catch, not safe to skip validation.

---

## Honest Expectations

**What this system will likely achieve:**
- Solid, balanced preflop ranges
- Reasonable postflop play in common spots
- Handles bet sizes outside the blueprint via a fast predictive resolver — and v3's search-heavy design should make off-tree play a relative *strength* rather than a patch
- Probably crushes strong amateurs consistently
- Competitive with many recreational and mid-stakes players

**What it will not achieve:**
- Pluribus-level exploitability (orders of magnitude more iterations)
- Perfect multiway equilibrium (multiplayer CFR has no convergence guarantee; your approximate Nash is empirical, and the predictive resolver's guarantees also weaken multiway)
- Reliable performance in rare, complex spots (limited by the deliberately coarse abstraction)

**The honest gap:** The genuine improvements stacked here — DCFR with full discounting, baseline-corrected sampling, optimistic updates, a predictive subgame solver, and belief-tracking resolving — compound to save meaningful training compute versus vanilla 2019 methods, and the baseline correction in particular targets your binding constraint directly. That is real. It does not close the gap between a 14-week solo project and a multi-year research team.

**The specific bet v3 makes:** that a coarser blueprint plus a stronger, faster resolver beats a fatter blueprint plus a weaker resolver, at this budget. The Phase 6 blueprint-only vs resolving A/B is what tells you whether the bet paid off. If it didn't — if resolving adds little and the coarse blueprint leaks in reachable spots — the v2 allocation (more RAM, richer blueprint) was the better call, and you fall back to it for the next iteration. Keep that fallback credible by ensuring the coarse blueprint is *correct* before you lean on search.

---

## Reading List

### Essential — Read Before Writing Code

**1. Libratus: The Superhuman AI for No-Limit Poker** — Brown & Sandholm, IJCAI 2017
Safe nested resolving and endgame solving. Why naive depth-limited search is exploitable and what the fix looks like. Required for Phase 5.

**2. Superhuman AI for Multiplayer Poker (Pluribus)** — Brown & Sandholm, Science 2019
The primary reference, and the source of v3's search-heavy rebalance. Pay attention to how *coarse* the blueprint was relative to the work done by real-time search — that is the design choice v3 leans on.

**3. Solving Imperfect-Information Games via Discounted Regret Minimization (DCFR)** — Brown & Sandholm, AAAI 2019
Your blueprint solver. Implement the full `(α, β, γ) = (1.5, 0, 2)`, not α alone.

**4. Variance Reduction in Monte Carlo Counterfactual Regret Minimization (VR-MCCFR)** — Schmid et al., AAAI 2019
**Promoted to essential in v3.** This is now core solver architecture, not optional reading. Understand the control-variate / baseline formulation well enough to implement the running baseline and to reason about its interaction with external sampling and pruning.

**5. Faster Game Solving via Predictive Blackwell Approachability** — Farina, Kroer, Brown & Sandholm, AAAI 2021
The predictive RM⁺ / PCFR⁺ result behind your resolver. Note carefully that the strong speedups are two-player-zero-sum, full-traversal results — exactly the resolving regime, not the sampled blueprint regime.

**6. ReBeL: Combining Deep RL and Search for Imperfect-Information Games** — Brown, Bakhtin, Lerer & Gong, NeurIPS 2020
The Public Belief State framework. You use the belief-tracking idea, not the neural/RL machinery. The value is understanding *why* belief tracking makes subgame solving principled, and where guarantees break without a learned value function.

### Important — Read During Implementation

**7. Monte Carlo Sampling for Regret Minimization in Extensive Games** — Lanctot et al., NeurIPS 2009
Foundational MCCFR: external, outcome, and chance sampling. Read alongside #4, since baselines layer onto external sampling.

**8. Regret-Based Pruning in Extensive-Form Games** — Brown & Sandholm, NIPS 2015
Your pruning strategy. Parameter recommendations are for 2-player HULHE and need adjustment for 6-max NLHE. Note the new three-way interaction with optimistic updates and baselines (Phase 3).

**9. Potential-Aware Imperfect-Recall Abstractions with EMD** — Johanson et al., AAAI 2013
Basis for potential-aware features. Also read it for the *imperfect-recall* framing that informs the Phase 4 diagnostic caveat.

**10. AIVAT: A New Variance Reduction Technique for Agent Evaluation** — Burch, Johanson & Bowling, 2018
Your evaluation methodology. Without it, version-to-version tournament results are noise. Essential for Phase 6.

**11. Finding Optimal Abstract Strategies in Extensive-Form Games** — Johanson, Burch, Valenzano & Bowling, AAAI 2012
Abstraction pathologies — more buckets is not always better. Directly relevant to v3's choice to target coarse bucket counts.

### Background — Read If You Have Time

**12. An Introduction to Counterfactual Regret Minimization** — Neller & Lanctot, 2013
Readable CFR tutorial with worked Kuhn examples. Read first if you are not already comfortable with regret minimization.

**13. Abstracting Real-World Games** — Brown, Sandholm & Amos, NeurIPS 2015
Information abstraction theory; the coarseness/quality tradeoff that v3 resolves in favor of search.

**14. Local Best Response** — Lisy & Bowling, IJCAI 2016 workshop
The LBR methodology for Phases 4 and 6.

**15. (Optional, currency check) Predictive/discounted CFR since 2021** — e.g. PCFR⁺ refinements and weighted variants such as PDCFR⁺ (IJCAI 2024) and asynchronous predictive step-size work (2024–2025)
All incremental and two-player-zero-sum-focused, so they bear on your *resolver*, not your blueprint. Worth a literature check if you go deep on the predictive subgame solver; nothing in this line overturns the v3 architecture.

### Reference — Keep Open While Coding

**16. OpenSpiel** — Lanctot et al., 2020
Reference CFR implementation. Compare your solver's behavior against it on small games when debugging. https://github.com/google-deepmind/open_spiel

**17. Cepheus Poker Project Documentation** — University of Alberta GAMES Group
Heads-up limit reference; abstraction and CFR details are useful sanity checks. http://poker.srv.ualberta.ca

---

### A Note on Sources

Do not rely on blog posts, AI-generated summaries, or video explanations for implementation decisions. The papers above are short, clearly written, and are the source of truth. When any assistant — including this plan — tells you an algorithm converges "much faster" or a technique is "the standard," check it against the cited paper and against the regime it was measured in. The predictive-solver speedups in #5, for example, are real but two-player-zero-sum and full-traversal; applying that number to your sampled 6-max blueprint would be a category error. The discipline of checking the regime, not just the citation, is what keeps this project honest.
