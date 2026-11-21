//! `play compact` — rewrite a hashed blueprint strategy in the dense format.

use std::path::PathBuf;

use poker_ai::play::policy::compact_policy;
use poker_ai::util::cli::{data_dir, flag};

use super::{cap_flag, fmt_bytes, load_game, load_policy, policy_path, validate};

/// Convert a blueprint strategy stored under hashed keys (every blueprint
/// trained before the dense format, and anything the `HashMap` trainer
/// writes) into the dense format — about a fifth of the size on disk and in
/// `play`'s memory, with identical lookups to within one `u16` step.
///
/// The flags must describe the game the blueprint was trained on, as for any
/// `play` command; the conversion fails, leaving no file, if they do not.
/// The result is read back and every info set compared against the original
/// before it is reported as written.  The original is never touched: play
/// the new file with `--policy=PATH`, or move it into place yourself.
///
///   play compact [--data=DIR --stack-bb=N --cap=N|P,F,T,R --policy=PATH --out=PATH]
///
/// `--out` defaults to `DATA/blueprint_holdem_dense.bin`.  Needs the hashed
/// policy in memory plus the dense one (≈ 16 GB + 3 GB at 595 M info sets),
/// and makes two passes of binary searches over it — minutes, not seconds.
pub(super) fn run(args: &[String]) {
    validate(args, &["data", "stack-bb", "cap", "policy", "out"], 0);
    let dir = data_dir(args);
    let stack_bb: u32 = flag(args, "stack-bb").unwrap_or(200);
    let cap = cap_flag(args);
    let source = policy_path(args, &dir);
    let out = flag::<String>(args, "out").map(PathBuf::from).unwrap_or_else(|| dir.join("blueprint_holdem_dense.bin"));
    if out == source {
        eprintln!("--out is the policy being converted; the original is kept, so name another file");
        std::process::exit(2);
    }

    println!("Loading abstraction from {} ({stack_bb}bb, cap-{cap})", dir.display());
    let game = load_game(&dir, stack_bb, cap).try_with_indexing().unwrap_or_else(|e| {
        eprintln!("cannot build the dense index for this game: {e}");
        std::process::exit(1);
    });
    let policy = load_policy(args, &dir);
    if policy.is_dense() {
        println!("{} is already dense — nothing to do.", source.display());
        return;
    }

    let t0 = std::time::Instant::now();
    let report = compact_policy(&policy, &game, &out, |stage, done, total| {
        eprintln!("  {stage}: {:.0}%  ({:.0}s)", 100.0 * done as f64 / total.max(1) as f64, t0.elapsed().as_secs_f64());
    })
    .unwrap_or_else(|e| {
        eprintln!("conversion failed, nothing written: {e}");
        std::process::exit(1);
    });
    println!(
        "Wrote {} — {} info sets, {} (was {} in memory); read back and compared: \
         every info set present, largest probability change {:.1e}.",
        out.display(),
        report.info_sets,
        fmt_bytes(report.slots as usize * 2),
        fmt_bytes(policy.resident_bytes()),
        report.max_error,
    );
    println!("Play it with --policy={}, or move it over {}.", out.display(), source.display());
}
