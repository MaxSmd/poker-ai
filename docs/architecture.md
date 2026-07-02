# Architecture

See [../poker-ai-plan-v3.md](../poker-ai-plan-v3.md) for the full design rationale.

## Crates

- `poker-core`: pure game logic, no solver dependency — game state, hand
  evaluation, action generation, integer pot-fraction bet sizing
- `poker-ai`: abstraction, blueprint solver, resolving, evaluation

## Pipeline

1. **Abstraction** (`abstraction/`) — equity features cached by suit-isomorphic
   key, dense `HandIndexer`, K-Means++ bucketing into per-street `BucketMap`s.
2. **Blueprint solver** (`solver/`) — DCFR over external-sampling MCCFR with
   VR-MCCFR baselines, optional optimistic updates and regret-based pruning,
   stored in a flat `f32` SoA regret table. Validated on Kuhn/Leduc against the
   full-traversal CFR oracle.
3. **Evaluation** (`evaluation/`) — exact best-response exploitability where
   chance is enumerable, plus Local Best Response (LBR), AIVAT, and self-play.
4. **Resolving** (`resolving/`) — depth-limited real-time subgame solving with a
   predictive (CFR⁺) solver, belief-state tracking, pluggable leaf evaluators,
   and continual re-solving made safe by a CFV gadget.

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
