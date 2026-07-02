# Poker AI

A No-Limit Texas Hold'em AI in Rust, built on the Pluribus/DeepStack architecture:
a **coarse blueprint strategy** trained offline with Monte-Carlo CFR over a card-
and betting-abstraction, sharpened at play time by **depth-limited continual
re-solving** with CFV-gadget safety.

- **poker-core** — the game engine: state machine, zero-alloc mutate-and-undo,
  LUT 7-card hand evaluation, action abstraction (`crates/poker-core`)
- **poker-ai** — everything on top: information abstraction, solvers
  (DCFR / MCCFR / CFR+), subgame re-solving, evaluation (`crates/poker-ai`)

See [docs/architecture.md](docs/architecture.md) for the system design and
[docs/](docs/) for deep dives (abstraction, CFR notes, memory budget).

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
`--parallel[=BATCH]`, `--soa`, `--resume`, `--chunk=N`, `--expl-every=N`
(see `train --help` header in `src/bin/train.rs`).

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

Reference points: heads-up 20 bb cap-2 ≈ 4.6M info sets ≈ 0.16 GB (trivial);
6-max 20 bb cap-2 ≈ 6.8B info sets ≈ 204 GB — see the tool's output for the
full matrix before launching anything big.

### 3. Train the blueprint (`train blueprint`)

```bash
# The headline run (production server config):
cargo run --release --bin train -- blueprint 300000000 20 1 \
    --cap=2 --soa --atomic --resume --expl --expl-iters=200000
```

- `--cap=N` — betting abstraction: max raises per street (the tree-size lever)
- `--soa` — flat structure-of-arrays regret store (24 B/info set vs ~350 B on
  the HashMap path; required scale for cap≥2). Needs the full-coverage
  abstraction from step 1.
- `--atomic[=THREADS]` — lock-free atomic training (Pluribus-style in-place
  CAS updates; defaults to all cores). Near-linear scaling — measured 4.5×
  over the batched path on 4 performance cores — at the cost of
  bit-determinism (thread interleaving races float updates). Convergence is
  gated against the serial path by exploitability.
- `--parallel[=BATCH]` — deterministic mini-batch parallel MCCFR
  (bit-reproducible per seed+batch, but merge-bound: ~7 effective cores)
- `--resume` — continue from `data/blueprint_holdem_soa.ckpt` (checkpoints are
  atomic and written every `--chunk`; an interruption costs at most one chunk)
- `--expl` / `--expl-iters=N` — periodic sampled best-response exploitability
  (a lower bound; needs large N to be meaningful)
- `--chunk=N`, `--expl-every=N` — progress/checkpoint cadence (default: line
  every 1%, exploitability every 10 chunks)

Outputs `data/blueprint_holdem.bin` — the average strategy, keyed identically
to the HashMap path, which is what the resolver loads.

### Experiment tracking (Weights & Biases)

Wrap any training command with the W&B logger (`pip install wandb`):

```bash
python scripts/train_wandb.py --name hu-cap2 -- \
    blueprint 300000000 20 1 --cap=2 --soa --parallel=512 --resume --expl
```

Metrics (iteration, info sets, nodes/s, exploitability) are parsed from the
trainer's `@wandb` lines and stepped by iteration, so runs of different length
line up. Without the wrapper the trainer's output is unchanged.

## Play-time re-solving

The blueprint is deliberately coarse; quality at the table comes from the
resolving stack (`crates/poker-ai/src/resolving/`): belief-state tracking,
subgame construction rooted at the actual public state (off-tree bets
included), CFR+ with blueprint warm-starting, multi-valued leaf continuations
(the depth-limited-solving fix), CFV-gadget continual re-solving for safety,
and a vectorized 1326-combo public-tree solver for full-range river resolves.
`bin/play.rs` (the interactive bot wiring blueprint + resolver) is the next
milestone.

## Evaluation toolkit

- `--expl` on the trainer: sampled best-response exploitability (lower bound)
- `evaluation/exploitability.rs`: exact-style push/fold exploitability (mbb/g)
- `evaluation/local_br.rs`: sampled best response for non-enumerable games
- `evaluation/aivat.rs`: AIVAT variance-reduced match evaluation
- `evaluation/self_play.rs`: seat-alternated head-to-head match runner
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
  evaluation/               exploitability, LBR, AIVAT, self-play
  bin/                      train, cluster, memory_estimate, benchmark, play
docs/                       architecture & design notes
scripts/                    W&B wrapper, analysis helpers
data/                       generated artifacts (gitignored)
```
