# Architecture

The system design. For the rationale behind each *option* — what it costs, when
to use it, what it measured — see [options.md](options.md); for measured results
see [summary.md](summary.md).

## Crates

- `poker-core`: pure game logic, no solver dependency — game state, hand
  evaluation, action generation, integer pot-fraction bet sizing
- `poker-ai`: abstraction, blueprint solver, resolving, evaluation

## Pipeline

1. **Abstraction** (`abstraction/`) — equity features cached by suit-isomorphic
   key, dense `HandIndexer`, K-Means++ bucketing into per-street `BucketMap`s.
2. **Blueprint solver** (`solver/`) — DCFR over external-sampling MCCFR with
   VR-MCCFR baselines, optional optimistic updates and regret-based pruning,
   stored in a flat SoA regret table (`f32` throughout, the strategy sum
   accumulated with stochastic rounding — see
   [memory-budget.md](memory-budget.md)). Validated on
   Kuhn/Leduc against the full-traversal CFR oracle.
3. **Evaluation** (`evaluation/`) — vectorized abstract-game best response and
   push/fold exploitability, the two metrics cheap enough to run in the loop.
4. **Resolving** (`resolving/`) — belief-state tracking and the vectorized
   public-tree solver the live bot re-solves with.
5. **Validation** (`validation/`) — the oracles all of the above are gated
   against, and the only part of the tree that never ships: Kuhn/Leduc/curated
   NLHE, full-traversal CFR, exact best response, Local Best Response, AIVAT,
   and the explicit-deal re-solving stack (predictive CFR⁺ subgames, leaf
   evaluators, CFV gadget, continual re-solving).

## Key Design Decisions

- Pure tabular regret storage (f32, dense SoA layout) — debuggable, no neural
  components in the training loop
- External sampling + VR-MCCFR baselines as the primary variance lever
- Raise-count cap as the blueprint's betting-abstraction / memory lever
- Pluggable leaf evaluator trait at the resolving layer, with a blueprint-lookup
  fallback always wired in
- Predictive CFR⁺ for subgames (fast last iterate near 2p0s), DCFR fallback for
  multiway
- Public chance sampling supported by pre-dealt board cards in `poker-core`
