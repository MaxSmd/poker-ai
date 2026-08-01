//! Command-line argument validation shared by the binaries.
//!
//! Every tool here hand-scans `argv` for the flags it cares about, which means
//! an unrecognised flag is *silently ignored* by default.  That is a real
//! hazard on this project: the expensive commands are multi-hour training and
//! abstraction runs, and a typo'd `--capp=3` or a stale `--expl` would run
//! happily to completion having trained something other than what was asked
//! for.  [`validate_flags`] turns that into an immediate, explanatory failure.

/// Reject unknown flags and stray positional arguments, exiting with status 2
/// and a message naming the accepted flags.
///
/// `skip` is how many leading `argv` entries to ignore (1 for the program name,
/// 2 when a subcommand follows it).  `allowed` lists flag names *without* the
/// leading dashes; a flag may appear as `--name` or `--name=value`, so only the
/// part before `=` is matched.  `max_positionals` bounds the non-flag arguments
/// the command accepts.
///
/// **Every positional in this project is numeric** (iteration counts, stack
/// depths, seeds, bucket counts), so a non-numeric one is rejected regardless of
/// the budget.  That is the check that catches the dangerous typo: `--data dir`
/// written with a space instead of `=` leaves the flag valueless *and* parks the
/// path in a positional slot, so the tool silently reads and writes the default
/// directory instead of the one that was asked for.
pub fn validate_flags(args: &[String], skip: usize, allowed: &[&str], max_positionals: usize) {
    let mut positionals = 0;
    for a in args.iter().skip(skip) {
        let Some(body) = a.strip_prefix("--") else {
            positionals += 1;
            if a.parse::<f64>().is_err() {
                eprintln!(
                    "unexpected argument `{a}`: positional arguments are numeric \
                     (flags take their value as --name=value, not --name value)"
                );
                std::process::exit(2);
            }
            if positionals > max_positionals {
                eprintln!(
                    "unexpected argument `{a}`: at most {max_positionals} positional \
                     argument(s) here"
                );
                std::process::exit(2);
            }
            continue;
        };
        let name = body.split('=').next().unwrap_or("");
        if !allowed.contains(&name) {
            eprintln!("unknown flag `--{name}`; expected one of: {}", allowed.join(", "));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// Accepted forms must pass: bare flags, `=`-valued flags, and positionals
    /// up to the stated budget.  (Rejection paths call `process::exit`, so they
    /// are exercised by the CLI smoke tests rather than here.)
    #[test]
    fn accepts_known_flags_and_budgeted_positionals() {
        let args = argv(&["train", "1000", "20", "--soa", "--data=/tmp/x"]);
        validate_flags(&args, 1, &["soa", "data"], 3);
    }

    #[test]
    fn flag_value_is_not_counted_as_a_positional() {
        let args = argv(&["cluster", "--data=/tmp/x"]);
        validate_flags(&args, 1, &["data"], 0);
    }

    #[test]
    fn skip_ignores_the_subcommand() {
        let args = argv(&["train", "blueprint", "--cap=3"]);
        validate_flags(&args, 2, &["cap"], 0);
    }

    /// A space-separated flag value parks a non-numeric token in a positional
    /// slot.  Guarded in a subprocess because rejection exits the process — this
    /// is the case that silently wrote to the default directory before the
    /// numeric rule existed.
    #[test]
    fn rejects_a_space_separated_flag_value() {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "--exact",
                "util::cli::tests::space_separated_child",
                "--ignored",
                "--nocapture",
            ])
            .output()
            .expect("run child");
        assert_eq!(out.status.code(), Some(2), "rejection must exit with status 2");
        let text = String::from_utf8_lossy(&out.stderr) + String::from_utf8_lossy(&out.stdout);
        assert!(
            text.contains("positional arguments are numeric"),
            "and must say why, got: {text}"
        );
    }

    /// The child of [`rejects_a_space_separated_flag_value`]; exits(2) by design.
    #[test]
    #[ignore = "subprocess helper: exits the process on purpose"]
    fn space_separated_child() {
        let args = argv(&["cluster", "2", "1", "--data", "/tmp/x"]);
        validate_flags(&args, 1, &["data"], 5);
    }
}
