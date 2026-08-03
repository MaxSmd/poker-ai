# Information Abstraction

Built offline by the `cluster` binary
([`bin/cluster.rs`](../crates/poker-ai/src/bin/cluster.rs)), which is the
authority for everything on this page — read its module header for the memory
guard and the per-board sweep, and its `main` for the current defaults.

## Bucketing

| Street  | Feature | Clustering |
|---------|---------|------------|
| Preflop | 169 canonical hand classes | none — the class *is* the bucket |
| Flop    | equity-distribution histogram | L2 K-Means++ ([`clustering::kmeans`](../crates/poker-ai/src/abstraction/clustering.rs)) |
| Turn    | equity-distribution histogram | L2 K-Means++ |
| River   | exact scalar equity, or the 8-dim OCHS vector under `POKER_AI_RIVER_OCHS=1` | exact 1-D DP (`cluster_1d`) for scalar; K-Means++ for OCHS |

Each street is clustered independently on its own feature; there is no
conditioning of the turn on its flop bucket. Bucket counts are a command-line
argument, not a property of the code:

```sh
cluster [cap] [seed] [flop_k] [turn_k] [river_k]
```

`memory_estimate` takes the same three counts in the same order and defaults to
the same values, so a no-arg run of either tool describes the same abstraction.
**Check the footprint in `memory_estimate` before spending the build time in
`cluster`** — see [memory-budget.md](memory-budget.md).

Maps are loaded per street as `BucketMap`s; a street with no map on disk falls
back to its exact suit-canonical key, so a partial build still trains, just at
finer-than-intended resolution on the missing streets.

### Where the resolution belongs

The measured result ([summary.md](summary.md#findings)) is that **info-set count
is a poor proxy for exploitability contribution**. River buckets are ~93% of all
info sets, but tripling *flop and turn* resolution is what produced the 3.85×
exploitability improvement — best-response profit is weighted by reach frequency
× money at stake, and early-street errors compound into every later street.

The river is also the street the bot re-solves live from the actual public
state, so blueprint resolution there buys less than it costs. Spend the budget
on the streets that are not re-solved.

## Features

All features build on `river_equity`
([`abstraction/features/equity.rs`](../crates/poker-ai/src/abstraction/features/equity.rs)):
the exact probability a hand beats a uniformly random opponent on a complete
board, by full enumeration, cached by suit-isomorphic key.

- **Equity-distribution histogram** (`ehs_histogram`) — what the flop/turn
  clusterer consumes
- **OCHS** (`board_ochs`, `OCHS_K = 8`) — equity against 8 opponent hand
  clusters rather than one scalar; strictly better river buckets at equal count
  (`examples/bench_ochs.rs`) at the cost of an 8× larger equity cache
- `ehs` — expected hand strength over future runouts
- `ehs2` — second moment (variance in outcomes)
- `draw_potential`

Features are computed **per board**, not per hand: one board scores all ~1081
holes at once in O(n log n) via the sweep in
[`features/sweep.rs`](../crates/poker-ai/src/abstraction/features/sweep.rs), so
every situation gets an exact feature far more cheaply than per-hand
enumeration. Exact, low-noise features cluster better — sampling noise would
otherwise be the ceiling on bucket quality.

A dense, suit-isomorphic `HandIndexer` maps every canonical `(hole, board)`
bijectively onto a flat integer, so lookup is one index computation plus one
array read rather than a hash.
