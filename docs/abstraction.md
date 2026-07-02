# Information Abstraction

## Bucketing

| Street  | Buckets  | Method                              |
|---------|----------|-------------------------------------|
| Preflop | 169 × 6  | Canonical hands per position        |
| Flop    | 500–800  | K-Means++ on equity-distribution histograms |
| Turn    | 500–800  | Re-cluster conditioning on flop bucket |
| River   | 800–1200 | Highest fidelity — where exploitability leaks |

Bucket counts are abstraction targets (see `poker-ai-plan-v3.md`), loaded per
street as `BucketMap`s; a street with no map loaded falls back to its exact
suit-canonical key. v3 targets the coarser end of these ranges so that more of
the compute/RAM budget goes to the resolver.

## Features

All features build on `river_equity` (`abstraction/features.rs`): the exact
probability a hand beats a uniformly random opponent on a complete board,
computed by full enumeration and cached by suit-isomorphic key.

- **Equity-distribution histogram** — what the clusterer actually consumes
- EHS — Expected Hand Strength over future runouts
- EHS² — second moment (variance in outcomes)
- Draw potential

A dense, suit-isomorphic `HandIndexer` maps every canonical `(hole, board)`
bijectively onto a flat integer, so the abstraction is flat arrays (one index
computation + one array read) rather than hash maps.
