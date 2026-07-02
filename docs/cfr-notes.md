# CFR Notes

## Algorithm Stack

**Blueprint solver** (sampled, `solver/mccfr.rs`):

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

**Subgame resolver** (`solver/predictive.rs`): full-traversal **CFR⁺ / RM⁺**
last-iterate, used for depth-limited real-time re-solving where the near-2p0s
regime makes it converge fastest per second. Falls back to DCFR for multiway
subgames.

## Validation Protocol

1. Kuhn Poker → exact solution known, exploitability < 0.001 bb/hand
2. Leduc Poker
3. Heads-up NLHE
4. 6-player NLHE blueprint

The same full-traversal `solver/cfr.rs` core is the correctness oracle that
validates Kuhn/Leduc before the sampled variants are layered on.
