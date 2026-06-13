#!/usr/bin/env python3
"""Run the Rust blueprint trainer under Weights & Biases experiment tracking.

The Rust `train` binary has no W&B SDK (and there is no first-class one for
Rust), so this wrapper drives it as a subprocess: it sets `POKER_AI_METRICS=1`
so the trainer emits machine-readable `@wandb` / `@wandb-config` JSON lines on
stdout (a no-op without the env var, so plain `cargo run --bin train` is
unchanged), streams that stdout through to your terminal, and forwards each
parsed metric to W&B via `wandb.log`.

Usage:
    python scripts/train_wandb.py [--project P] [--entity E] [--name N]
                                  [--mode online|offline|disabled] [--debug]
                                  -- <train args...>

Examples:
    python scripts/train_wandb.py -- 1000000 20 1 --optimistic
    python scripts/train_wandb.py --name rbp-run -- 2000000 20 1 --rbp
    python scripts/train_wandb.py --mode offline -- 500000 20 1 --soa

Any arguments after `--` (or any unrecognised args) are forwarded verbatim to
the `train` binary; see `train.rs` for its flags. Logged metrics are stepped by
training `iteration`, so the W&B charts line up across runs of different length.

Requires `pip install wandb` (run `wandb login` once for online mode).
"""

import argparse
import json
import os
import subprocess
import sys

# Prefixes the Rust trainer uses for its machine-readable lines.
CONFIG_TAG = "@wandb-config "
METRIC_TAG = "@wandb "


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Run the Rust blueprint trainer under W&B tracking.",
        epilog="Arguments after `--` are forwarded to the train binary.",
    )
    p.add_argument("--project", default="poker-ai", help="W&B project (default: poker-ai)")
    p.add_argument("--entity", default=None, help="W&B entity/team (default: your default)")
    p.add_argument("--name", default=None, help="W&B run name (default: auto-generated)")
    p.add_argument(
        "--mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (default: online)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Build/run the trainer in debug instead of --release.",
    )
    # Everything else is forwarded to the trainer. `--` is consumed by argparse.
    args, train_args = p.parse_known_args(argv)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    return args, train_args


def build_command(debug, train_args):
    cmd = ["cargo", "run"]
    if not debug:
        cmd.append("--release")
    cmd += ["--bin", "train", "--", *train_args]
    return cmd


def main():
    args, train_args = parse_args(sys.argv[1:])

    try:
        import wandb
    except ImportError:
        sys.exit("wandb is not installed. Run: pip install wandb")

    cmd = build_command(args.debug, train_args)
    env = {**os.environ, "POKER_AI_METRICS": "1"}

    print(f"[train_wandb] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=None, env=env, text=True, bufsize=1
    )

    run = None

    def ensure_run(config):
        # Lazily start the run so the trainer's `@wandb-config` line seeds the
        # config; if metrics somehow arrive first we still init with the CLI args.
        nonlocal run
        if run is None:
            run = wandb.init(
                project=args.project,
                entity=args.entity,
                name=args.name,
                mode=args.mode,
                config={"train_args": train_args, **config},
            )
        return run

    try:
        for line in proc.stdout:
            sys.stdout.write(line)  # mirror the trainer's output verbatim
            sys.stdout.flush()
            stripped = line.strip()
            try:
                if stripped.startswith(CONFIG_TAG):
                    ensure_run(json.loads(stripped[len(CONFIG_TAG):]))
                elif stripped.startswith(METRIC_TAG):
                    metric = json.loads(stripped[len(METRIC_TAG):])
                    step = metric.get("iteration")
                    ensure_run({}).log(metric, step=step)
            except json.JSONDecodeError:
                # A malformed metric line should never kill a long training run.
                print(f"[train_wandb] skipped unparseable metric: {stripped}", flush=True)
    finally:
        code = proc.wait()
        if run is not None:
            run.summary["exit_code"] = code
            run.finish(exit_code=0 if code == 0 else 1)

    sys.exit(code)


if __name__ == "__main__":
    main()
