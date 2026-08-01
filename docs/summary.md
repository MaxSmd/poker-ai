# Poker AI — Technical Report

A heads-up No-Limit Texas Hold'em agent written from scratch in Rust
(~23.9k LoC), following the Libratus/DeepStack architecture: a coarse
**blueprint strategy** trained offline by Monte-Carlo CFR over a card and
betting abstraction, refined at play time by **depth-limited continual
re-solving** with a counterfactual-value gadget. It plays the public
[Slumbot](https://www.slumbot.com) benchmark over its HTTP API at 200 bb and
currently beats it.

## Headline results

| Metric | Result |
|---|---|
| **Slumbot, 10k hands, 200 bb** | **+7.9 ± 26.0 bb/100** (luck-adjusted; raw +11.0 ± 31.1) |
| **Blueprint exploitability** | **2909.8 mbb/hand** vs a card-exact best response (3.85× better than the prior abstraction's 11207.0) |
| Blueprint scale | 464M info sets, 517K betting sequences, 12.45 GB (f32 SoA) |
| Training cost | 3×10⁹ MCCFR iterations / 3.15×10¹² nodes in 13.9 h on 32 cores (**62.9M nodes/s**) |
| Card abstraction | 1500 flop / 1500 turn / 800 river buckets over 1.29M / 13.96M / 123.16M canonical situations |
| Play-time re-solve | ~1–2 s per river decision, full 1326-hand ranges |
| Test suite | 250 tests, 0 failures, 0 clippy warnings; deterministic per seed |

The exploitability figure is measured against a **card-exact** nemesis (it sees
exact hole cards; the blueprint sees only buckets), so it upper-bounds the cost
of abstraction rather than describing play against real opponents — the deployed
agent re-solves the river instead of playing the raw blueprint.

## Key technologies

**Game engine** (`poker-core`) — zero-allocation state machine with
mutate-and-undo for CFR traversal; Cactus-Kev-style lookup-table 7-card
evaluation; pot-fraction action abstraction with configurable raise caps.

**Information abstraction** — exact equity features (scalar equity, equity
histograms, opponent-cluster hand strength / OCHS); suit-isomorphic canonical
hand indexing for dedup; K-means (flop/turn) and an exact 1-D dynamic program
(river) for bucketing; full-coverage precompute parallelised over boards with
per-board deterministic RNG streams.

**Solvers** — vanilla CFR (exact oracle), Discounted CFR (α,β,γ)=(1.5,0,2),
external-sampling MCCFR with a control-variate baseline for variance reduction,
CFR⁺ with alternating regret-matching⁺ and linear averaging, optimistic /
predictive regret updates, and regret-based pruning.

**Storage & parallelism** — flat structure-of-arrays regret table (32 B per
info set vs ~350 B on a hash map, f64 strategy sums for exact long-run
averaging); two parallel training paths, a bit-reproducible mini-batch scheme
and a lock-free atomic CAS scheme measured at **4.5×** the batched path;
atomic checkpointing with resume.

**Continual re-solving** — vectorized public-tree CFR⁺ over full 1326-combo
range vectors with inclusion-exclusion reach folding and pre-sorted showdown
tables; Bayesian range tracking with card removal; CFV-gadget re-solving for
theoretical safety across successive decisions; multi-valued continuation leaves
at depth cuts; explicit river chance nodes on turn re-solves.

**Deployment** — dual-state tracking mirroring the real hand inside the abstract
game, with randomized pseudo-harmonic action translation (Ganzfried & Sandholm)
for off-tree opponent bets; memory-mapped compact policy (~9 GB resident for an
8 GB artifact vs ~25 GB as a hash map).

**Evaluation** — vectorized exact best response over the abstract game,
AIVAT-style chance-only control variates for variance-reduced match scoring,
sampled best response (LBR family) for non-enumerable games, and an exact
betting-tree enumerator for memory budgeting before a run.

## Validation

Every component is gated against an independently derived ground truth.

| Gate | Result |
|---|---|
| Kuhn / Leduc poker | Converge to known game values (−1/18; −0.0856 over 288 info sets) |
| Vectorized best response | Matches an independent scalar oracle to **1e-8** per hand, both seats |
| Full-river turn re-solve | **0.0014 bb** exploitable vs 1.44 bb for a leaf-cut re-solve (~1000×) |
| Multi-valued leaves (K=4) | **0.00037 bb** vs 1.37 bb single-continuation (~3700×) |
| CFR⁺ vs DCFR on a subgame | 0.0055 bb vs 0.0294 bb at equal iteration budget |
| Blueprint warm start | 3 iterations: 3.27 bb cold → **0.0048 bb** warm-started |
| CFV gadget | Opponent best response bounded by the unconstrained bootstrap ± 0.02 bb |
| Luck adjustment | Removes 301.9 bb of card luck over 10k hands; 1.20× tighter CI |

## Findings

- **Info-set count is a poor proxy for exploitability contribution.** River
  buckets are 93% of all info sets, yet tripling *flop and turn* resolution drove
  the 3.85× exploitability improvement: best-response profit is weighted by reach
  frequency × money at stake, and early-street errors compound into every later
  street.
- **The gap to a larger static bot is the betting abstraction, not the card
  abstraction.** Slumbot's published figures are ~6M betting sequences and 5.7B
  info sets against this system's 517K and 464M — 11.6× apart on sequences but
  only ~2.5× on buckets. Matching it by table size implies ~270 GB; real-time
  re-solving closes the same gap inside a 32 GB budget, and does so while winning.
- **Coarse river buckets are cheap when the river is re-solved.** A static bot
  must play the river from its blueprint; this one solves it live, so abstraction
  budget belongs on the streets that are *not* re-solved.
- **Abstraction coarseness manifests as identifiable strategic error.** A
  measured big-blind over-fold (~35% versus a theoretical 20–25%, with an
  inverted hand ordering) was traced to kicker-quality distinctions erased by
  coarse flop/turn buckets, and closed by finer buckets alone — confirmed
  independently by the strategy chart, the exploitability metric, and the match
  result (big-blind preflop −29/−33 → **−11.0 ± 9.4 bb/100**).
- **Algorithmic caveats worth recording:** DCFR's advantage is variance
  suppression and appears only under sampling; CFR⁺ requires alternating
  updates; naive predictive/optimistic CFR⁺ diverged on Leduc and is deferred in
  favour of plain CFR⁺.

## Limits

The headline match result is not statistically significant on its own
(+7.9 ± 26.0 crosses zero), and 62 stack-off hands (0.6%) account for 194% of
net winnings; the preflop sub-lines are the statistically conclusive part. Turn
and flop re-solving are implemented, tested, and wired but off by default —
unmeasured in match play, and the largest known remaining leak sits exactly
there (big blind, flop seen: −102.3 ± 21.4 bb/100). The system is heads-up only;
multiway subgames and a 6-max blueprint are unbuilt.
