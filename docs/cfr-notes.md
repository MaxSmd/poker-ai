# CFR Notes

## Algorithm Stack

**Blueprint solver** (sampled, `solver/mccfr/`):

1. **DCFR** (Brown & Sandholm, 2019) — full three-parameter discounting
   `(α, β, γ) = (1.5, 0, 2)`; the deployed strategy is the γ-weighted *average*,
   not the last iterate.
2. **External-sampling MCCFR** (Lanctot et al., 2009) — makes traversal
   tractable on NLHE: one traverser is fully explored, chance and opponents are
   sampled.
3. **VR-MCCFR baselines** (Schmid et al., 2019) — first-class variance lever; a
   running per-(info set, action) baseline as a control variate. Enabled via
   `with_baseline`.
4. **Optimistic updates** (Farina et al., 2021) — momentum term
   (`R_t = R_{t-1} + 2·r_t − r_{t-1}`) for faster last-iterate convergence.
   Enabled via `with_optimistic`. Serial-only.
5. **Regret-Based Pruning** (Brown & Sandholm, 2015) — configurable θ and K,
   enabled after a warm-up fraction of training, with periodic full refresh
   traversals. Serial-only.

Storage is a flat structure-of-arrays of `f32` (arithmetic in `f64`). The
`poker-core` engine pre-deals board cards, which also supports public chance
sampling.

**Subgame resolver.** The algorithm is **CFR⁺ / RM⁺** last-iterate: in the
near-2p0s regime of a subgame it converges fastest per second, which is what a
per-decision time budget rewards. It falls back to DCFR for multiway subgames.
Two implementations, on either side of the production/validation split:

- `resolving/vector_cfr/` — **what the bot runs.** Vectorized public-tree CFR⁺
  over full 1326-combo range vectors.
- `validation/solver/predictive.rs` — the explicit-deal oracle it is gated
  against. Full-traversal, arbitrarily slow, never linked into a binary.

## Validation Protocol

1. Kuhn Poker → exact solution known, exploitability < 0.001 bb/hand
2. Leduc Poker
3. Heads-up NLHE

Multiway is not part of the protocol yet: there is no 6-max blueprint and no
multiway subgame solver. `memory_estimate` will size a 6-max tree, but nothing
trains one.

The same full-traversal `validation/solver/full_cfr.rs` core is the correctness
oracle that validates Kuhn/Leduc before the sampled variants are layered on.
Everything under `validation/` is oracle code and never links into a shipped
binary; see the README's production-vs-validation section.
