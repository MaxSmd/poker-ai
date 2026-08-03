# Poker AI

A No-Limit Texas Hold'em AI in Rust, built on the Libratus/DeepStack architecture:
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
./check.sh          # fast lane: build, clippy, docs, tests         (~2 min)
./check.sh gates    # heavy lane: the oracle gates                  (~10 min)
./check.sh all      # both
```

Run the fast lane before you stop for a session — it catches the whole "didn't
compile / obvious breakage" class. Run the gates after touching a solver,
abstraction or resolving path: those are what prove the fast production code
still agrees with its slow oracle (see [Production vs. validation](#production-vs-validation)).

The fast lane also builds the docs with `RUSTDOCFLAGS="-D warnings"`, so a
`[`Foo`]` link that a rename broke fails the same pass as a clippy lint. Docs
are the easiest layer to let rot; this is the gate that stops it.

The gates run at 2 threads under background QoS, because several allocate a few
hundred MB and four at once pushes an 8 GB laptop into swap. `NICE=0 ./check.sh
gates` runs at full speed on a machine that can take it.

Everything is deterministic per seed: solvers, clustering, and the parallel
training paths all reproduce bit-identical results for a fixed `(seed, batch)`.

**Benchmarks are not tests** and deliberately aren't in either lane — they
measure speed, which has no pass/fail and flakes on a throttled machine. Run
them by hand on an idle box:

```bash
cargo run --release --example bench_train_paths     # MCCFR path throughput, parallel scaling
cargo run --release --example bench_rbp             # RBP theta/K sensitivity
cargo run --release --example bench_resolve_cost    # per-decision resolve cost by mode
                                                    #   (add `-- all` for the multi-GB
                                                    #    full-river / flop arms)
cargo run --release --example bench_ochs            # OCHS vs scalar river feature
cargo run --release --example bench_continual       # continual re-solving / warm start
cargo run --release --example inspect_buckets       # what a bucket actually contains
```

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
`--data=DIR` (see `train` header in `src/bin/train/main.rs`).

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

Reference points: heads-up 200 bb cap-3 ≈ 299M info sets ≈ 11 GB;
6-max 20 bb cap-2 ≈ 6.8B info sets ≈ 161 GB — see the tool's output for the
full matrix before launching anything big (it is the authority; these are the
same reference points restated at the current 12 B/slot).

### 3. Train the blueprint (`train blueprint`)

```bash
# The headline run: production server config, Slumbot-depth 200 bb stacks.
cargo run --release --bin train -- blueprint 3000000000 200 1 \
    --cap=3 --soa --atomic --resume
```

Measured cost of that run — 3×10⁹ iterations, 3.15×10¹² nodes, 13.9 h on 32
cores — and everything it produced are in
[docs/summary.md](docs/summary.md#headline-results). Keep the numbers there and
cite them from here; two copies drift.

- `--cap=N` — betting abstraction: max raises per street (the tree-size lever)
- `--soa` — flat structure-of-arrays regret store (12 B per (info set, action)
  slot — 24 B/info set at 2 actions — vs ~350 B/info set on the HashMap path;
  all three accumulators `f32`, the strategy sum accumulated with stochastic
  rounding so long-run averaging cannot freeze, see
  [docs/memory-budget.md](docs/memory-budget.md)). Needs the full-coverage
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
  (`resolving/vector_cfr/`, ~1–2 s per decision). `--no-resolve` plays the
  blueprint throughout instead.
- The runner prints a running **bb/100 ± 95% CI**, emits `@wandb` metric lines
  (wrap with `scripts/train_wandb.py` to chart a long match), persists the
  session token, and logs every hand to `data/slumbot_results.csv`.

Flags: `--iters=N` (resolve iterations), `--river-cap=N`, `--purify=X`
(drop sub-X action probabilities), `--seed=N`, `--no-resolve`,
`--token=`/`--username=`/`--password=` — see the header of `src/bin/play.rs`.

The rest of the resolving stack (`crates/poker-ai/src/validation/resolving/`,
the explicit-deal oracle the vectorized solver is gated against) — CFV-gadget
continual re-solving, blueprint warm-starting, multi-valued leaf continuations,
full-river turn resolves — is implemented and tested; turn/flop play-time
resolving is wired into the bot but off by default (`--resolve-turn`
/`--resolve-flop`), and a Slumbot A/B of those arms is the next measurement.

## Evaluation toolkit

- `play expl`: vectorized abstract-game best response — the blueprint quality
  metric (`evaluation/vector_br/`)
- `evaluation/exploitability.rs`: exact-style push/fold exploitability (mbb/g)
- `validation/evaluation/local_br.rs`: sampled best response, generic over
  `Game` — the tool for future non-`BlueprintHoldem` (e.g. multiway) games
- `validation/evaluation/aivat.rs`: AIVAT variance-reduced match evaluation (the
  conceptual oracle behind `play/luck.rs`'s live luck adjustment)
- `examples/`: bucket inspector, OCHS-vs-scalar benchmark, continual-resolving
  benchmark

## Repository layout

```
crates/poker-core/          game engine (state, actions, evaluator, undo)
crates/poker-ai/src/
  abstraction/              hand indexing, equity features, clustering, buckets
  games/                    Game/CursorGame/IndexedGame traits; push/fold,
                            BlueprintHoldem
  solver/                   MCCFR (+SoA/atomic stores), DCFR, pruning, variant
  resolving/                belief tracking, vectorized public-tree CFR
  evaluation/               push/fold exploitability, vectorized BR
  play/                     the live bot, protocol, Slumbot client, tracker
  util/                     combo bijection, hashing, RNG, CLI validation
  validation/               ORACLES — nothing here ships (see below)
    games/                  Kuhn, Leduc, curated-deal NLHE
    solver/                 full-traversal CFR, exact BR, predictive CFR+
    resolving/              explicit-deal subgame, gadget, continual, leaf eval
    evaluation/             AIVAT, sampled local BR
  bin/                      train, cluster, memory_estimate, benchmark, play
docs/                       architecture & design notes
scripts/                    W&B wrapper, analysis helpers
data/                       generated artifacts (gitignored)
```

<a name="production-vs-validation"></a>
### Production vs. validation

Everything under `validation/` exists **only to check the code above it**, and
none of it links into the `train`, `cluster` or `play` binaries. That split is
what makes "does this ship?" answerable from the path alone — production is
written for scale (flat SoA storage, quantized tables, sampled traversal,
vectorized public trees), validation is written to be obviously correct (full
tree walks, `HashMap` storage, explicit per-deal enumeration) and is allowed to
be arbitrarily slow. Each optimization is admitted only once it reproduces its
slow twin's answer; `src/validation/mod.rs` lists the pairings. The property is
checkable:

```sh
cargo build --release
nm -C target/release/train | grep -c 'poker_ai::validation::'   # expect 0
```

Dependencies run one way — validation imports production, never the reverse
(production references it only from `#[cfg(test)]` blocks and doc links).
