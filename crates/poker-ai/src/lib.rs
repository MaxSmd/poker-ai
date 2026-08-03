// Documentation is gated like everything else here: a link that no longer
// resolves is a compile error, not a warning nobody reads.  `check.sh` runs
// `cargo doc` in the fast lane so this fires on the same pass as clippy.
#![deny(rustdoc::broken_intra_doc_links)]

pub mod abstraction;
pub mod evaluation;
pub mod games;
pub mod play;
pub mod resolving;
pub mod solver;
pub mod util;
pub mod validation;
