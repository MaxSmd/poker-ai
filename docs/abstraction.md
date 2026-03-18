# Information Abstraction

## Bucketing

| Street  | Buckets  | Method                        |
|---------|----------|-------------------------------|
| Preflop | 169 × 6  | Canonical hands per position  |
| Flop    | 500–800  | K-Means++ on EHS histograms   |
| Turn    | 500–800  | Re-cluster conditioning flop  |
| River   | 1000–1500| Highest fidelity              |

## Features

- EHS — Expected Hand Strength
- EHS² — second moment (variance in outcomes)
- Draw potential
