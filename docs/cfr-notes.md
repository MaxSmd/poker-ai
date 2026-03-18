# CFR Notes

## Algorithm Stack

1. **DCFR** (Brown & Sandholm, 2019) — base solver with discounting
2. **Optimistic updates** (Farina et al., 2021) — momentum term for faster convergence
3. **Public chance sampling** — reduce multiplayer variance
4. **Regret-Based Pruning** — configurable θ and K parameters

## Validation Protocol

1. Kuhn Poker → exact solution known, exploitability < 0.001 bb/hand
2. Leduc Poker
3. Heads-up NLHE
4. 6-player NLHE blueprint
