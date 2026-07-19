# Poker AI

A No-Limit Texas Hold'em AI in Rust, built on the Pluribus/DeepStack architecture:
a **coarse blueprint strategy** trained offline with Monte-Carlo CFR over a card-
and betting-abstraction, sharpened at play time by **depth-limited continual
re-solving** with CFV-gadget safety.

- **poker-core** — the game engine: state machine, zero-alloc mutate-and-undo,
  LUT 7-card hand evaluation, action abstraction (`crates/poker-core`)
- **poker-ai** — everything on top: information abstraction, solvers
  (DCFR / MCCFR / CFR+), subgame re-solving, evaluation (`crates/poker-ai`)

See [docs/architecture.md](docs/architecture.md) for the system design,
**[docs/options.md](docs/options.md) for every implemented option (solvers,
discount schedules, stores, training paths, abstraction, resolving) with use
cases, measured benefits, and drawbacks**, and [docs/](docs/) for deep dives
(abstraction, CFR notes, memory budget).

## Build & test

```bash
cargo build --release
cargo test                          # fast suite (~1 min, optimized test profile)
cargo test --release -- --ignored   # heavy gates: full enumerations, convergence runs
```

Everything is deterministic per seed: solvers, clustering, and the parallel
training paths all reproduce bit-identical results for a fixed `(seed, batch)`.

## Quick start: a converging blueprint in 20 seconds

Push/fold NLHE needs no card abstraction and converges on a laptop:

```bash
cargo run --release --bin train -- 3000000 20 1
```

This trains a 20 bb heads-up push/fold blueprint with DCFR + variance-reduced
MCCFR (3M iterations, ~20 s), prints the 13×13 SB shove chart plus a measured
exploitability (mbb/hand), and persists the strategy to
`data/blueprint_pushfold.bin`. Flags: `--optimistic`, `--rbp`,
`--parallel[=BATCH]`, `--soa`, `--resume`, `--chunk=N`, `--expl-every=N`,
`--data=DIR` (see `train --help` header in `src/bin/train.rs`).

## Training the headline model (heads-up NLHE blueprint)

The full pipeline is three commands. Steps 1–2 are cheap; step 3 is the long
training run. All long-running steps checkpoint and `--resume`.

### 1. Build the card abstraction (`cluster`)

Buckets every canonical `(hole, board)` situation per street using exact equity
features (scalar / histogram / OCHS), K-means (flop/turn) and an exact 1-D DP
(river), keyed by a dense suit-isomorphic hand index:

```bash
# Laptop (capped: 300 boards/street, turn skipped by the 1.5 GB memory guard)
cargo run --release --bin cluster -- 300 1

# Server, full coverage — required for the real blueprint (--soa needs it)
POKER_AI_CLUSTER_MEM_GB=8 POKER_AI_RIVER_OCHS=1 \
  cargo run --release --bin cluster -- 0 1
```

Both `cluster` and `train` take `--data=DIR` to redirect all artifacts
(caches, maps, checkpoints, blueprints) away from the default `data/` — use it
on quota-limited boxes to point the bulk files at scratch space.

Writes `data/{flop,turn,river}_buckets.bin` (+ equity caches). Full coverage is
flop 1.29M / turn 13.96M / river 123.16M canonical situations; on a 64-core
box the whole build is ~30 min (river OCHS k-means dominates).

### 2. Check the memory footprint (`memory_estimate`)

Enumerates the **exact** abstract betting tree (2-player and 6-max) and prints
info sets, action slots, and regret-table RAM for a stack × raise-cap matrix:

```bash
cargo run --release --bin memory_estimate            # current bucket counts
cargo run --release --bin memory_estimate -- 200 200 200   # what-if buckets
```

Reference points: heads-up 200 bb cap-3 ≈ 299M info sets ≈ 14.5 GB;
6-max 20 bb cap-2 ≈ 6.8B info sets ≈ 204 GB — see the tool's output for the
full matrix before launching anything big.

### 3. Train the blueprint (`train blueprint`)

```bash
# The headline run (production server config — Slumbot-depth 200bb stacks;
# measured: 8.2 h / ~16 GB RSS on 32 cores, 2B iterations, 2.0T nodes):
cargo run --release --bin train -- blueprint 2000000000 200 1 \
    --cap=3 --soa --atomic --resume
```

- `--cap=N` — betting abstraction: max raises per street (the tree-size lever)
- `--soa` — flat structure-of-arrays regret store (32 B/info set vs ~350 B on
  the HashMap path; f64 strategy sums so long-run averaging stays exact).
  Needs the full-coverage abstraction from step 1.
- `--atomic[=THREADS]` — lock-free atomic training (Pluribus-style in-place
  CAS updates; defaults to all cores). Near-linear scaling — measured 4.5×
  over the batched path on 4 performance cores — at the cost of
  bit-determinism (thread interleaving races float updates). Convergence is
  gated against the serial path by exploitability.
- `--parallel[=BATCH]` — deterministic mini-batch parallel MCCFR
  (bit-reproducible per seed+batch, but merge-bound: ~7 effective cores)
- `--resume` — continue from `data/blueprint_holdem_soa.ckpt` (checkpoints are
  atomic and written every `--chunk`; an interruption costs at most one chunk)
- `--chunk=N` — progress/checkpoint cadence (default: line every 1%)
- `--data=DIR` — artifact directory (default `data/`)

There is no in-loop exploitability on the blueprint paths: the sampled
best-response bound is meaningless at any affordable sample count on a tree
this size (it read *negative*) and cost ~25 min per report. Measure the trained
artifact with `play expl` (the vectorized abstract-game best response) as a
milestone metric instead.

Outputs `data/blueprint_holdem.bin` — the average strategy, keyed identically
to the HashMap path, which is what the resolver loads.

### Experiment tracking (Weights & Biases)

Wrap any training command with the W&B logger (`pip install wandb`):

```bash
python scripts/train_wandb.py --name hu-200bb-cap3 -- \
    blueprint 2000000000 200 1 --cap=3 --soa --atomic --resume
```

Metrics (iteration, info sets, nodes/s, exploitability) are parsed from the
trainer's `@wandb` lines and stepped by iteration, so runs of different length
line up. Without the wrapper the trainer's output is unchanged.

## Playing against Slumbot

`bin/play.rs` wires the trained blueprint into a live agent for
[Slumbot](https://www.slumbot.com) (heads-up NLHE, 200 bb, blinds 50/100 — the
standard public benchmark bot):

```bash
# Needs data/blueprint_holdem.bin + the bucket maps from the SAME training run.
cargo run --release --bin play -- slumbot 10000
```

Architecture (`crates/poker-ai/src/play/`):

- **Dual-state tracking** — the real hand is mirrored inside the abstract
  blueprint game; off-tree opponent bets are translated by **randomized
  pseudo-harmonic mapping** (Ganzfried & Sandholm 2013) in pot-fraction space,
  and our abstract raises translate back at the same pot fraction.
- **Bayes range tracking** — both players' ranges are updated at every
  decision with the blueprint's action likelihoods per hand, plus card removal.
- **River re-solving** — each river decision is re-solved from the *actual*
  public state (real pot/stacks, so translation error vanishes where the money
  is deepest) with the vectorized full-range public-tree CFR⁺ solver
  (`resolving/vector_cfr.rs`, ~1–2 s per decision). `--no-resolve` plays the
  blueprint throughout instead.
- The runner prints a running **bb/100 ± 95% CI**, emits `@wandb` metric lines
  (wrap with `scripts/train_wandb.py` to chart a long match), persists the
  session token, and logs every hand to `data/slumbot_results.csv`.

Flags: `--iters=N` (resolve iterations), `--river-cap=N`, `--purify=X`
(drop sub-X action probabilities), `--seed=N`, `--no-resolve`,
`--token=`/`--username=`/`--password=` — see the header of `src/bin/play.rs`.

The rest of the resolving stack (`crates/poker-ai/src/resolving/`) — CFV-gadget
continual re-solving, blueprint warm-starting, multi-valued leaf continuations,
full-river turn resolves — is implemented and tested; turn/flop play-time
resolving is wired into the bot but off by default (`--resolve-turn`
/`--resolve-flop`), and a Slumbot A/B of those arms is the next measurement.

## Evaluation toolkit

- `play expl`: vectorized abstract-game best response — the blueprint quality
  metric (`evaluation/vector_br.rs`)
- `evaluation/exploitability.rs`: exact-style push/fold exploitability (mbb/g)
- `evaluation/local_br.rs`: sampled best response, generic over `Game` — the
  tool for future non-`BlueprintHoldem` (e.g. multiway) games
- `evaluation/aivat.rs`: AIVAT variance-reduced match evaluation (the
  conceptual oracle behind `play/luck.rs`'s live luck adjustment)
- `examples/`: bucket inspector, OCHS-vs-scalar benchmark, continual-resolving
  benchmark

## Repository layout

```
crates/poker-core/          game engine (state, actions, evaluator, undo)
crates/poker-ai/src/
  abstraction/              hand indexing, equity features, clustering, buckets
  games/                    Game/CursorGame/IndexedGame traits; Kuhn, Leduc,
                            push/fold, BlueprintHoldem
  solver/                   CFR, DCFR, MCCFR (+SoA store), CFR+, pruning
  resolving/                subgame, gadget, continual re-solving, vector CFR
  evaluation/               exploitability, LBR, AIVAT, vectorized BR
  bin/                      train, cluster, memory_estimate, benchmark, play
docs/                       architecture & design notes
scripts/                    W&B wrapper, analysis helpers
data/                       generated artifacts (gitignored)
```
