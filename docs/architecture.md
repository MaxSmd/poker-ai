# Architecture

See [../poker-ai-plan-v2.md](../poker-ai-plan-v2.md) for the full design rationale.

## Crates

- `poker-core`: pure game logic, no solver dependency
- `poker-ai`: abstraction, blueprint solver, resolving, evaluation

## Key Design Decisions

- Pure tabular regret storage (f32, SoA layout)
- Pluggable leaf evaluator trait at the resolving layer
- Public chance sampling throughout blueprint training
