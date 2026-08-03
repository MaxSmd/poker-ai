#!/bin/sh
# Verification, sized for a fanless laptop rather than a build farm.
#
#   ./check.sh          fast lane  — build, clippy, docs, tests  (~2 min)
#   ./check.sh gates    heavy lane — the oracle gates             (~10 min)
#   ./check.sh all      both
#
# Run the fast lane before you stop for the session; it catches the whole
# "didn't compile / obvious breakage" class. Run the gates when you have touched
# a solver, abstraction or resolving path — they are what actually proves the
# fast production code still agrees with its slow oracle.
#
# Heat: the gates run at 2 threads under background QoS (macOS `taskpolicy -b`,
# which parks the work on efficiency cores). Slower in wall-clock, but the
# machine stays cool and usable — start it and go do something else. Set
# NICE=0 to run at full speed on a machine that can take it.
#
# Benchmarks are deliberately NOT here. They measure speed, which has no
# pass/fail and flakes on a throttled machine; run them by hand on an idle box:
#   cargo run --release --example bench_train_paths     # MCCFR path throughput
#   cargo run --release --example bench_rbp             # RBP theta/K sweep
#   cargo run --release --example bench_resolve_cost    # per-decision resolve cost
#   cargo run --release --example bench_ochs            # OCHS vs scalar river

set -e
cd "$(dirname "$0")"

# Background QoS keeps a fanless machine off its thermal limit. Absent on
# non-macOS, and skippable with NICE=0.
NICE_CMD=""
if [ "${NICE:-1}" != "0" ] && command -v taskpolicy >/dev/null 2>&1; then
    NICE_CMD="taskpolicy -b"
fi

fast() {
    echo "== build =="
    cargo build --all-targets
    echo "== clippy =="
    cargo clippy --all-targets -- -D warnings
    # Docs are gated like the code: a `[`Foo`]` that no longer resolves fails
    # here. Without this the whole doc layer is the one subsystem with no
    # check, and it rots silently under every rename. Covers the bins too,
    # which the crate-level `deny` in lib.rs cannot reach.
    echo "== docs =="
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all
    echo "== tests =="
    cargo test --release
}

gates() {
    echo "== oracle gates (2 threads${NICE_CMD:+, background QoS}) =="
    echo "   ~10 min. These are correctness only; nothing here measures speed."
    # 2 threads, not the default all-cores: several gates allocate a few hundred
    # MB of coverage maps, and four of those at once pushes an 8 GB machine into
    # swap — which is both slower and hotter than the compute it replaces.
    $NICE_CMD cargo test --release -- --ignored --test-threads=2
}

case "${1:-fast}" in
    fast)  fast ;;
    gates) gates ;;
    all)   fast; gates ;;
    *)     awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 2 ;;
esac

echo
echo "OK  ($(date '+%Y-%m-%d %H:%M'))"
