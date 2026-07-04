# Options Guide

Every configurable choice in this repository, what it's for, and what it costs.
Numbers marked **[measured]** come from benchmarks in this repo (M1 Air, 4P+4E
cores, unless noted); everything else is the literature default. Fully-qualified
paths are given once per section.

The one-paragraph version: **train blueprints with `SoaMccfr` + DCFR + baseline
on the atomic path; validate anything new on Kuhn/Leduc/push-fold with the
exact solvers first; resolve subgames with CFR+ warm-started from the
blueprint; switch to the lean store only when RAM is the binding constraint.**

---

## 1. Solver families

| Solver | Use when | Benefits | Drawbacks |
|---|---|---|---|
| `solver::cfr::Cfr` (full traversal) | Toy/validation games with enumerable chance (Kuhn, Leduc, curated-deal NLHE) | Exact, zero variance — the correctness oracle every sampled result is validated against | Visits the whole tree every iteration; intractable beyond toy games |
| `solver::mccfr::Mccfr` (external-sampling, HashMap store) | Sampled games without a dense index: uncapped betting trees, partial-coverage abstractions, quick experiments | No precomputed layout needed (mints keys on first visit); carries every refinement (baseline / optimistic / RBP) | ~350 B per info set (5 heap `Vec`s per node) — ~15× the SoA store; HashMap lookups on the hot path |
| `solver::mccfr::SoaMccfr` (flat f32 store) | Production blueprint training on an `IndexedGame` (finite raise cap + full-coverage bucket maps) | 24 B/info set for 2 actions; array indexing instead of hashing; checkpoint = contiguous arrays; the only path with atomic training | Requires the dense index up front (full-coverage abstraction); no optimistic/RBP (inert on the games it targets) |
| `solver::mccfr::LeanMccfr` (quantized store) | RAM-bound runs only (deep-stack 6-max, very fine abstractions) | **Half the accumulator bytes** (12 vs 24 B/info set for 2 actions) at equal wall time and equal convergence **[measured: push/fold 1M iters — lean −0.006 bb vs f32 0.051 bb exploitability]** | Must pair with `Discount::LINEAR` (see §2); serial-only so far (no parallel/atomic path); fixed-point ranges sized for bb-scale utilities |
| `solver::predictive::PredictiveSolver` (CFR+) | Subgame re-solving under a per-decision time budget | Strong **last-iterate** (the deployable output in a 2–5 s resolve) **[measured: Leduc 2k iters — CFR+ last-iterate 0.007 beats vanilla average 0.011; subgame @2k: 0.0055 vs DCFR 0.0294 bb]** | Full-traversal only (subgames are small enough); alternating updates required — simultaneous RM+ converges far worse |
| `solver::best_response` | Measuring, not training | Exact exploitability (NashConv/2) on enumerable games | Enumerable chance only; use `evaluation::local_br` otherwise |

## 2. Discount schedules (`Variant` / `Discount`)

All three run through the same solvers; a schedule is one constant.

| Schedule | Use when | Benefits | Drawbacks |
|---|---|---|---|
| `Variant::Vanilla` | Baseline comparisons, theory sanity checks | Simplest; textbook guarantees | Slowest; early-iteration noise never decays |
| `Discount::RECOMMENDED` (DCFR 1.5/0/2) — **default** | Everything, unless quantizing | Fastest average-strategy convergence of the three; β=0 lets actions crushed by early noise recover instantly | β=0 keeps regrets **bounded** ⇒ fundamentally incompatible with 16-bit storage (quantization noise swamps a non-growing signal — proven by the rejected bf16 experiment) |
| `Discount::LINEAR` (LCFR 1/1/1) | Required companion of the lean store | Regrets **grow** with t ⇒ fixed-point error becomes relatively negligible (the Pluribus int-storage regime) | Modest theoretical convergence edge conceded to DCFR (not detectable in our push/fold benchmark, but expect it on larger games) |

Untested middle ground worth a future one-constant experiment: `(1.5, 1, 2)` —
DCFR's positive discounting and γ-averaging with quantization-friendly growing
negatives.

## 3. MCCFR refinements (builder flags)

| Flag | Use when | Benefits | Drawbacks |
|---|---|---|---|
| `.with_baseline()` (VR-MCCFR control variate) | **Always on** for sampled training | Unbiased variance reduction; benefit grows with game size **[measured: Kuhn ~20% / Leduc ~10% lower sd; on by default in every trainer]** | Third accumulator (+4 B/action f32, +2 B lean); baselines must be **per-action** — a per-info-set scalar cancels out algebraically and does nothing |
| `.with_optimistic()` (predictive updates, `R += 2rₜ − r_{t−1}`) | Last-iterate experiments only | Accelerates the last iterate | We deploy the γ-averaged strategy, where it **regressed** on push/fold **[measured: 130.7 vs 111.5 mbb/g]**; extra `prev_inst` array; no multiplayer guarantee |
| `.with_pruning()` (Regret-Based Pruning) | Deep trees under Vanilla/CFR+ where negative regret accumulates | Skips hopeless branches **[measured: Kuhn converged at ~91% of node visits]** | Partly conflicts with DCFR (β=0 keeps regrets above θ ⇒ no-op); θ must be deep relative to the game's regret scale or it breaks *mixed* equilibria (shallow θ on Kuhn: 0.1 expl) |
| `save_checkpoint` / `load_checkpoint` | Any run longer than a coffee | Atomic write (tmp+rename); resume is **bit-identical** to an uninterrupted run; interruption costs ≤ one chunk | None — always checkpoint |

## 4. Training execution paths

| Path | Determinism | Scaling | Use when |
|---|---|---|---|
| `train` / `train_fast` (serial; clone / zero-alloc cursor) | Bit-reproducible per seed; clone and cursor are **bit-identical** to each other | 1 core (~6.8M nodes/s on the cap-2 blueprint) **[measured]** | Debugging, validation, anything needing exact reproducibility |
| `train_parallel` / `train_parallel_fast` (deterministic mini-batch) | Bit-reproducible per (seed, batch) | Merge-bound: 1.09× on the blueprint locally, ~7 effective cores on a 64-core server **[measured]** | Reproducible parallel runs where the ceiling is acceptable |
| `train_atomic` (lock-free CAS, `SoaMccfr` only) | **Not** bit-reproducible (thread interleaving races float order) | Near-linear: 4.5× on 4 P-cores, 5.05× with E-cores **[measured: 34.3M nodes/s vs 7.4M batched]** | **Production blueprint training** — the many-core default (`--atomic`) |

Convergence of the atomic path is gated by exploitability against the serial
reference (per-hand strategy diffs are the wrong gate — two *serial* seeds
differ by the same Monte-Carlo noise, ~0.05 mean at 1M iterations).

## 5. Games (validation ladder)

| Game | Role |
|---|---|
| `games::kuhn` / `games::leduc` | Known-solution gates (value −1/18; 288 info sets, −0.0856) — every solver feature must converge here first |
| `games::nlhe` (curated 4-deal HU) | Proves solver-over-real-engine wiring with enumerable chance |
| `games::push_fold` | First *converging* real-mechanics blueprint (338 info sets, known Nash shove charts) — the standing benchmark game for store/path experiments |
| `games::blueprint::BlueprintHoldem` | The real target: sampled deals, per-street card buckets, raise-capped betting, optional dense indexing |

Trait tiers: `Game` (clone-based, clarity) → `CursorGame` (zero-alloc
apply/undo — 1.31× **[measured]**) → `IndexedGame` (dense info-set index ⇒ SoA
stores). Implement the deepest tier the game can support.

## 6. Card abstraction (offline `cluster` build)

| Option | Use when | Benefits | Drawbacks |
|---|---|---|---|
| Pre-flop: 169 canonical classes | Always (exact, free) | Lossless under suit isomorphism | — |
| Flop/turn: equity-distribution histograms → K-means++ | Always | Exact features via the O(n log n) board sweep (~45× the old MC build **and** noise-free) | K-means is a local optimum (seeded, deterministic) |
| River default: scalar equity → exact 1-D DP (`cluster_1d`) | RAM-constrained builds | Globally optimal buckets for the scalar feature; cheap (river full build in seconds) | Equity-vs-uniform is a lossy 1-D projection — a hard fidelity ceiling no bucket count fixes (99 vs AA conflated on wet boards) |
| River OCHS: 8-dim equity-vs-hand-class (`POKER_AI_RIVER_OCHS=1`) | Server builds (the production choice) | Strictly dominates scalar at every bucket count **[measured: +12% @k=8 → +38% @k=50 held-out RMSE; scalar@50 never reaches OCHS@8]**; solver-side RAM unchanged | Offline equity cache is 8× (≈3.9 GB river full); k-means instead of exact DP |
| `POKER_AI_CLUSTER_MEM_GB` | Sizing the build to the machine | Skips any street whose flat cache exceeds the budget (default 1.5 GB) | Skipped street = unabstracted at train time (info sets won't plateau — that's the signal) |

Bucket counts (edit in `cluster.rs` main): current 500/500/800. Coarse is a
*feature* in this architecture — the resolver re-solves from the flop at play
time, so blueprint buckets are a prior, not the ceiling. Decide on measured
`--expl`, not a priori.

## 7. Betting abstraction (`BlueprintHoldem` policy, not engine)

| Option | Effect | Trade-off |
|---|---|---|
| `with_raise_cap(n)` | Dominant tree-size lever **[measured: HU 20bb — cap-1 1.35M info sets / cap-2 4.63M / cap-3 5.84M; 6-max 20bb — 2.4B / 6.8B / 8.6B]** | Off-tree opponent raises are handled by the resolver (subgame rooted at the actual state), so a low cap costs less than it looks |
| Bet-size menus (`poker-core/betting.rs`) | 3 flop / 3 turn / 4 river pot fractions | Each size multiplies through every later street; trimming `TURN_BET_FRACS` 3→2 is the cheapest standing memory win |
| `abstract_bet_size` (nearest-size mapping) | Maps raw opponent bets onto the abstraction | Nearest-size; pseudo-harmonic mapping (Ganzfried & Sandholm) is the known upgrade, unimplemented |

The engine itself (`poker-core`) is deliberately uncapped and faithful; all
abstraction lives in the game layer. `memory_estimate` prints the exact
footprint for any stack × cap × bucket configuration (2-player and 6-max)
before you commit a server to it.

## 8. Regret storage

| Store | B/info set (2 actions) | Use when | Notes |
|---|---|---|---|
| HashMap `Node` (f64) | ~350 | Toy games, non-indexed games | Correctness-first; carries all refinements |
| `RegretTable` (f32 SoA) — **default** | 24 + 5 index | All current blueprint training | f64 math, f32 store; only store with parallel/atomic/checkpoint paths |
| `LeanTable` (i16/u16 quantized) | 12 + 5 index | RAM-bound only | Requires LCFR; RTN regrets + stochastic-rounded strategy sums + quantized-EMA baselines; saturation halves an info set (strategy-preserving). Rejected sibling: bf16+SR under DCFR (~4× convergence regression — do not revisit without re-reading `lean_table.rs`'s doc) |

Known f32 caveat: absolute `t^γ` strategy weights lose f32 precision past
~30–100M iterations (increments fall below half an ulp and the average
freezes). The lean store's running-discount form avoids this; porting it to
the f32 path is an open follow-up.

## 9. Re-solving (play-time) options

| Option | Use when | Benefits | Drawbacks |
|---|---|---|---|
| `SubgameSolver` + `SolverKind::Predictive` (CFR+) — **default** | Heads-up subgame resolves | Best strategy per unit of time budget (last-iterate) | Heads-up semantics |
| `SolverKind::Dcfr` (average) | The designated multiway fallback | Average-strategy safety when CFR+'s guarantees lapse | Slower to the same quality **[measured: 0.0294 vs 0.0055 bb @2k iters]** |
| `with_warm_start` (blueprint-seeded regrets) | Every resolve | Enormous head start **[measured: 3 iters — cold 3.27 bb vs warm 0.0048 bb]** | Warm-start `scale` must track pot size or it washes out |
| `CheckdownLeafEval` | Local testing; fallback | Exact all-in-equity leaves, no artifacts needed | Assumes check-down — blind to future betting leverage |
| `BlueprintLeafEval` (+ `with_continuations`, K=4) | Production depth limits | The depth-limited-solving fix: opponent picks among K continuations **[measured: K-aware resolve 0.003 bb vs 1.31 bb exploitable in the K=4 game — ~400×]** | Value table must be populated from a trained blueprint (pending) |
| `resolving::gadget` + `continual` (CFV re-solving) | Multi-street play | Provable safety (opponent held to carried CFVs); warm re-entry **[measured: ~2× fewer iters from a coarse carry, up to ~1000× on re-entry]** | Explicit-deal enumeration — fine for narrowed ranges |
| `resolving::vector_cfr` | Full-range (1326×1326) river resolves | Public-tree vectorization **[measured: ~1.1M-deal equivalent in ~1.3 s; agrees with the explicit oracle to 0.0001 bb]** | Complete-board subgames only; depth-limit leaves panic rather than mis-score |

## 10. Evaluation toolkit

| Tool | Use when | Caveat |
|---|---|---|
| `best_response::exploitability` | Enumerable games — the exact gate | Doesn't exist for sampled games |
| `evaluation::exploitability::push_fold_exploitability` | Push/fold benchmarks | Decoupled estimator (removes max-over-noise bias); reads slightly negative within noise near Nash |
| `evaluation::local_br` (sampled BR) | Non-enumerable blueprints (`--expl`) | Lower bound; needs large sample counts to be meaningful; commit argmax per *info set*, never per node (clairvoyance trap) |
| `evaluation::aivat` | Match evaluation | ~3× tighter stderr **[measured on Leduc: 0.0080 vs 0.0247]**, unbiased |
| `evaluation::self_play` | A/B strategy comparison | Seat alternation cancels positional EV |

## 11. CLI & environment quick reference

```
train [iters] [stack_bb] [seed]            push/fold trainer (HashMap path)
      --soa                                flat-store path
      --atomic[=N]                         lock-free training, N threads (default: all)
      --parallel[=BATCH]                   deterministic mini-batch
      --optimistic --rbp                   Phase-3 refinements (serial HashMap path)
      --resume                             continue from checkpoint
      --chunk=N --expl-every=N             progress/checkpoint/eval cadence
train blueprint [iters] [stack] [seed]     full HU blueprint
      --cap=N --soa --atomic --expl --expl-iters=N   (see README §3)
train compare                              before/after table of the refinements
cluster [cap] [seed]                       card abstraction build
      POKER_AI_CLUSTER_MEM_GB=8            per-street cache budget
      POKER_AI_RIVER_OCHS=1                OCHS river feature
memory_estimate [flop turn river]          exact footprint matrix (2p + 6-max)
      POKER_AI_ESTIMATE_STATES=N           betting-tree memo cap
scripts/train_wandb.py -- <train args>     W&B tracking wrapper
POKER_AI_METRICS=1                         emit @wandb metric lines (no-op otherwise)
```
