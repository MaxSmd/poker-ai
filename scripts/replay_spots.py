#!/usr/bin/env python3
"""Select logged spots for `play replay`, and compare the traces it writes.

    python scripts/replay_spots.py select LOG.jsonl --out=SPOTS.jsonl \\
        [--street=turn --seat=SB --spot="vs bet" --size=40-80% --max=N]
    python scripts/replay_spots.py summarize SPOTS.jsonl NAME=TRACE.jsonl ...

`select` finds every decision of ours in the log matching the spot (keys as
in spot_report.py; `--spot=unopened` ignores --size), keeps a reproducible
random `--max` of them, and writes one line per spot: the logged hand plus
`target`, the offset of that decision in its action string.  A sidecar
`SPOTS.jsonl.meta` records what the spot is judged by — betting frequency
when unopened, folding frequency when facing a bet — and Slumbot's rate in
the same spot of the same log.

`summarize` prints, for each named trace, the agent's average rate there —
overall, and by the made-hand class of our cards — next to the rate actually
logged on the same spots, then applies the decision rule fixed before the
run:

  * the `default` trace must reproduce the logged rate within 5 points, or
    the replay is not faithful and no other row means anything;
  * a setting FIXES the leak if its rate is within 10 points of Slumbot's in
    the same spot — on either side, so trading over-folding for
    over-calling does not count — and, for a fold spot, we fold top pair or
    better at most 10% of the time.
"""
import json
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from spot_report import STREETS, decisions, spot_key  # noqa: E402

RANKS = "23456789TJQKA"
CLASSES = ["flush+", "straight+", "set/trips", "two pair", "overpair", "top pair",
           "weaker pair", "flush draw", "nothing"]
STRONG = set(CLASSES[:6])  # top pair or better


def card(s):
    return RANKS.index(s[0]), s[1]


def straight_in(ranks):
    rs = set(ranks) | ({-1} if 12 in ranks else set())
    return any(all(r + k in rs for k in range(5)) for r in range(-1, 9))


def hand_class(hole, board):
    """Made-hand class of `hole` on `board`, counting only what a hole card makes."""
    h = [card(c) for c in hole]
    b = [card(c) for c in board]
    allc = h + b
    suits = Counter(s for _, s in allc)
    hole_suits = {s for _, s in h}
    if any(n >= 5 and s in hole_suits for s, n in suits.items()):
        return "flush+"
    if straight_in([r for r, _ in allc]) and not straight_in([r for r, _ in b]):
        return "straight+"
    ranks = Counter(r for r, _ in allc)
    hr = [r for r, _ in h]
    if any(ranks[r] >= 3 for r in set(hr)):
        return "set/trips"
    paired = {r for r in hr if ranks[r] >= 2}
    if len(paired) >= 2:
        return "two pair"
    if paired:
        r = next(iter(paired))
        top = max(r for r, _ in b)
        if hr[0] == hr[1] and r > top:
            return "overpair"
        return "top pair" if r == top else "weaker pair"
    if any(n == 4 and s in hole_suits for s, n in suits.items()):
        return "flush draw"
    return "nothing"


def arg(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def stat_of(want):
    """What a spot is judged by: betting frequency when nothing is bet yet,
    folding frequency when facing a bet."""
    return "bet" if want[2] == "unopened" else "fold"


def took(stat, token):
    return token == ("b" if stat == "bet" else "f")


def select(log, out):
    want = (STREETS.index(arg("street", "turn")), arg("seat", "SB"), arg("spot", "vs bet"), arg("size", "40-80%"))
    if want[2] == "unopened":
        want = want[:3] + ("",)
    stat = stat_of(want)
    max_n = int(arg("max", "0"))
    spots, slumbot = [], []
    n_hands = 0
    for ln in open(log):
        if not ln.strip():
            continue
        r = json.loads(ln)
        n_hands += 1
        for d in decisions(r.get("action", "")):
            if spot_key(d) != want:
                continue
            if d["pos"] != r["pos"]:
                slumbot.append(took(stat, d["action"]))
                continue
            if len(r.get("hole") or []) != 2 or len(r.get("board") or []) < [0, 3, 4, 5][want[0]]:
                continue
            spot = {k: r[k] for k in ("i", "pos", "hole", "board", "action")}
            spot["target"] = d["offset"]
            spots.append(spot)
    found = len(spots)
    if max_n and found > max_n:
        spots = random.Random(1).sample(spots, max_n)  # fixed seed: reproducible
    with open(out, "w") as w:
        for spot in spots:
            w.write(json.dumps(spot) + "\n")
    meta = {
        "key": list(want),
        "stat": stat,
        "slumbot_rate": sum(slumbot) / len(slumbot) if slumbot else None,
        "slumbot_n": len(slumbot),
        "log": log,
    }
    with open(out + ".meta", "w") as w:
        json.dump(meta, w)
    sl = f"{meta['slumbot_rate']:.2f}" if slumbot else "n/a"
    print(f"{len(spots)} of {found} spots {want} from {n_hands} hands -> {out}   "
          f"(judged by {stat}; Slumbot {stat}s {sl} here, n={len(slumbot)})")


def summarize(spots_path, named):
    spots = [json.loads(ln) for ln in open(spots_path) if ln.strip()]
    try:
        meta = json.load(open(spots_path + ".meta"))
    except FileNotFoundError:
        sys.exit(f"{spots_path}.meta missing — re-run `select` (it writes the spot's stat and Slumbot's rate)")
    stat, ref = meta["stat"], meta["slumbot_rate"]
    field = "raise" if stat == "bet" else "fold"  # an unopened BetTo is a bet
    logged = sum(took(stat, s["action"][s["target"]]) for s in spots) / len(spots)
    key = " ".join(x for x in [STREETS[meta["key"][0]], *meta["key"][1:]] if x)

    print(f"\n=== {key}: {len(spots)} spots, judged by {stat} frequency ===")
    print(f"Logged {stat} rate on them: {logged:.2f}.   Slumbot in the same spot: "
          f"{ref:.2f} (n={meta['slumbot_n']})\n")
    strong_col = f"{stat}, top pair+"
    hdr = f"{'setting':<22} {'n':>4}  {stat:>5}  {strong_col:>15}  " + "  ".join(f"{c:>11}" for c in CLASSES) + "   paths"
    print(hdr)
    print("-" * len(hdr))
    verdicts = {}
    for name, path in named:
        try:
            rows = [json.loads(ln) for ln in open(path) if ln.strip()]
        except FileNotFoundError:
            print(f"{name:<22} (no trace at {path} — that arm did not run)")
            continue
        if not rows:
            print(f"{name:<22} (empty trace)")
            continue
        by = defaultdict(list)
        for r in rows:
            by[hand_class(r["hole"], r["board"])].append(r[field])
        rate = sum(r[field] for r in rows) / len(rows)
        strong = [f for c in STRONG for f in by[c]]
        strong_rate = sum(strong) / len(strong) if strong else float("nan")
        paths = Counter(r["path"] for r in rows)
        secs = [r["secs"] for r in rows if "secs" in r]
        if secs:
            paths[f"{sum(secs) / len(secs):.2f}s/decision"] = 0
        cells = "  ".join(f"{sum(by[c]) / len(by[c]):>5.2f} n={len(by[c]):<4}" if by[c] else f"{'—':>11}" for c in CLASSES)
        print(f"{name:<22} {len(rows):>4}  {rate:>5.2f}  {strong_rate:>15.2f}  {cells}   "
              + ", ".join(f"{k} {v}" if v else k for k, v in sorted(paths.items())))
        verdicts[name] = (rate, strong_rate, len(rows))

    print("\nDecision rule (fixed before the run):")
    if "default" not in verdicts:
        print("  no `default` trace — cannot check that the replay is faithful.")
        return
    d_rate, _, d_n = verdicts["default"]
    if abs(d_rate - logged) > 0.05:
        print(f"  REPLAY NOT FAITHFUL: default {stat}s {d_rate:.2f}, the log shows {logged:.2f}. "
              "Fix the replay before reading any other row.")
        return
    if d_n < 0.9 * len(spots):
        print(f"  REPLAY INCOMPLETE: default traced {d_n} of {len(spots)} spots — check the errors in its log.")
        return
    print(f"  replay faithful: default {d_rate:.2f} vs logged {logged:.2f}.")
    for name, (rate, strong_rate, _) in verdicts.items():
        near = abs(rate - ref) <= 0.10
        ok = near and (stat == "bet" or strong_rate <= 0.10)
        why = f"{stat} {rate:.2f} vs Slumbot {ref:.2f}"
        if stat == "fold":
            why += f", top pair+ folds {strong_rate:.2f}"
        print(f"  {name:<22} {'FIXES the leak' if ok else 'does not fix it'}   ({why})")


def self_check():
    assert hand_class(["Ah", "Kd"], ["As", "7c", "2d", "9h"]) == "top pair"
    assert hand_class(["Qh", "Qd"], ["Js", "7c", "2d", "9h"]) == "overpair"
    assert hand_class(["5h", "5d"], ["Js", "7c", "2d", "9h"]) == "weaker pair"
    assert hand_class(["Ah", "7d"], ["As", "7c", "2d", "9h"]) == "two pair"
    assert hand_class(["7h", "7d"], ["As", "7c", "2d", "9h"]) == "set/trips"
    assert hand_class(["Th", "8d"], ["Js", "7c", "9d", "2h"]) == "straight+"
    assert hand_class(["Ah", "3h"], ["Kh", "7h", "2h", "9c"]) == "flush+"
    assert hand_class(["Ah", "3h"], ["Kh", "7h", "2d", "9c"]) == "flush draw"
    assert hand_class(["Ah", "3d"], ["Ks", "7c", "2d", "9h"]) == "nothing"
    # A board pair or board straight is not ours.
    assert hand_class(["Ah", "3d"], ["Ks", "Kc", "2d", "9h"]) == "nothing"
    assert hand_class(["2h", "2s"], ["9c", "Tc", "Jd", "Qh", "Kd"]) == "weaker pair"


if __name__ == "__main__":
    self_check()
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) >= 2 and pos[0] == "select":
        select(pos[1], arg("out", "spots.jsonl"))
    elif len(pos) >= 3 and pos[0] == "summarize":
        summarize(pos[1], [tuple(p.split("=", 1)) if "=" in p else (p, p) for p in pos[2:]])
    else:
        print(__doc__)
        sys.exit(2)
