//! Small, exactly-solvable games with **known** equilibria, plus a curated-deal
//! NLHE bridge — the fixtures every solver is proven on before it is trusted.
//!
//! A bug in a sampled MCCFR loop does not crash; it manifests as "convergence is
//! weird".  The only way to catch that is to first watch the solver reproduce a
//! solution that is known in closed form.  Kuhn's game value and Leduc's
//! exploitability floor are those known answers.
//!
//! These implement the same [`Game`](crate::games::Game) trait as the production
//! blueprint games, so the *same* solver code runs on both — which is the whole
//! point: what is validated here is literally what trains there.
//!
//! * [`kuhn`] — 3-card, 1-street; game value known analytically.
//! * [`leduc`] — 6-card, 2-street; the standard CFR benchmark.
//! * [`nlhe`] — real NLHE mechanics over a *curated* deal set, small enough to
//!   enumerate.  Bridges the toy games and
//!   [`crate::games::blueprint`]: it checks the engine wiring (betting, pots,
//!   showdown) under full-traversal CFR, which the real deal space forbids.

pub mod kuhn;
pub mod leduc;
pub mod nlhe;
