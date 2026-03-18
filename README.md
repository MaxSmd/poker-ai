# Poker AI

A 6-max No-Limit Texas Hold'em AI based on the Pluribus architecture.

See [docs/architecture.md](docs/architecture.md) for system design, and [poker-ai-plan-v2.md](poker-ai-plan-v2.md) for the full implementation plan.

## Crates

- **poker-core** — game state, hand evaluation, action generation
- **poker-ai** — abstraction, solver (DCFR), resolving, evaluation

## Quick Start

```bash
cargo build --release
cargo run --bin benchmark
```
