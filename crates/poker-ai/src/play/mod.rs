//! Play-time deployment: turn the trained blueprint + resolving stack into an
//! agent that plays real hands — currently against **Slumbot**
//! (slumbot.com, heads-up NLHE, 200 bb).
//!
//! * [`protocol`] — Slumbot's action-string format, parsing, chip accounting
//! * [`cards`] — card codec between the wire strings and engine encoding
//! * [`equity`] — all-in equity vs. a belief range, the fallback when the
//!   opponent raises past the abstraction's cap and no blueprint node exists
//! * [`luck`] — AIVAT-style chance-node control variate for luck-adjusted
//!   match scoring (pot-weighted equity deltas at each board reveal)
//! * [`policy`] — compact zero-copy view of `data/blueprint_holdem.bin`
//! * [`tracker`] — dual-state (real ↔ abstract) tracking with pseudo-harmonic
//!   action translation
//! * [`bot`] — the decision engine: blueprint policy + Bayes range tracking +
//!   full-range vectorized river re-solving
//! * [`slumbot`] — the HTTP client
//!
//! The match runner lives in `bin/play.rs` (`play slumbot <hands>`).

pub mod bot;
pub mod cards;
pub mod equity;
pub mod luck;
pub mod policy;
pub mod protocol;
pub mod slumbot;
pub mod tracker;

pub use bot::{Bot, BotConfig, HandState};
pub use policy::CompactPolicy;
