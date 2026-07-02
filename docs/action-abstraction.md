# Action Abstraction

Bets and raises draw from a single per-street schedule of pot fractions
(`abstract_raise_amounts` in `poker-core/src/betting.rs`); a raise is just the
next level computed from the same fractions, so there is no separate raise
schedule. All sizing is integer-only `(numerator, denominator)` arithmetic for
deterministic, cross-platform trees.

| Street  | Sizes (pot fraction)               | Notes                              |
|---------|------------------------------------|------------------------------------|
| Preflop | 0.5, 1.0, 2.0                      | ≈ 2.5bb open / 3.5bb 3-bet / 6bb 4-bet |
| Flop    | 0.33, 0.67, 1.0                    |                                    |
| Turn    | 0.5, 0.75, 1.0                     |                                    |
| River   | 0.5, 0.75, 1.0, 1.5                | 1.5 = overbet                      |

All-in is always available on every street (offered by the caller).

The blueprint additionally caps the number of raises per street via
`BlueprintHoldem::with_raise_cap` — the dominant tree-size / memory lever. The
engine itself caps nothing (it re-offers reraises until stacks deplete), so the
cap is a blueprint-abstraction choice. Off-tree bet sizes at play time are
recovered by the subgame resolver.
