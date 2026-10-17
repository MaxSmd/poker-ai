#!/usr/bin/env python3
"""Where do we play differently from Slumbot?  Spot-by-spot action frequencies.

Win rates need tens of thousands of hands to separate two versions of the
agent; action frequencies need about a thousand occurrences of a spot.  A
10k-hand log has thousands of flop spots, so this is the fast instrument for
finding leaks.  Every decision in the log is rebuilt from the action string
and keyed by

    street, seat (SB/BB), unopened / facing a bet, raise depth, bet size

Both players' decisions are counted, so each spot has our frequencies and
Slumbot's frequencies *in the same seat and spot* side by side.  Slumbot is a
strong reference, not an oracle: a gap is a lead to investigate, not proof of
a leak.  For spots facing a bet the minimum defense frequency (MDF) is shown
as a second, theory-only reference — the defense that makes a pure bluff
break even.

Caveat: the two players reach a spot with different ranges (their preflop and
earlier-street strategies differ), and their bet sizes differ, so the spots
are matched by size *bucket*.  Large, significant gaps in frequent spots are
what this is for.

    python scripts/spot_report.py hands.jsonl [more.jsonl ...] [--min=150] [--all]

Ranking: "importance" ≈ |frequency gap| × average pot (bb) × our occurrences
per 100 hands — a rough bb/100 scale for ordering leads, NOT an estimate of
money lost.  --min sets the minimum occurrences per side (default 150);
--all prints every qualifying spot instead of the top 25.
"""
import json
import math
import sys
from collections import defaultdict

BIG_BLIND = 100
SMALL_BLIND = 50
STACK = 20000
STREETS = ["preflop", "flop", "turn", "river"]
SEAT = {0: "BB", 1: "SB"}  # Slumbot pos: 0 = big blind, 1 = small blind
SIZE_EDGES = [(0.4, "<40%"), (0.8, "40-80%"), (1.25, "80-125%"), (2.5, "125-250%"), (math.inf, ">250%")]


def size_bucket(x, all_in):
    if all_in:
        return "all-in"
    for edge, label in SIZE_EDGES:
        if x < edge:
            return label
    return SIZE_EDGES[-1][1]


def depth_label(street, depth):
    if street == 0:
        return {1: "first in", 2: "vs open", 3: "vs 3-bet"}.get(depth, "vs 4-bet+")
    return {1: "vs bet", 2: "vs raise"}.get(depth, "vs re-raise")


def decisions(action):
    """One dict per decision in a Slumbot action string.

    `pos` is Slumbot's (0 = BB, 1 = SB).  A bet's size `x` uses the agent's own
    pot-fraction convention (`play::tracker::map_real`):
    x = (to − level_before) / (pot_before + level_before).
    """
    out = []
    behind = 0  # chips each player committed on earlier streets (equal)
    base = 0  # offset of this street's substring in `action`
    for street, s in enumerate(action.split("/")):
        if street == 0:
            level = {1: SMALL_BLIND, 0: BIG_BLIND}
            cur, bets, pos = BIG_BLIND, 1, 1
        else:
            level = {0: 0, 1: 0}
            cur, bets, pos = 0, 0, 0
        faced = None  # (x, mdf, all_in) of the last bet on this street
        i = 0
        while i < len(s):
            tok = s[i]
            j = i + 1
            to = None
            if tok == "b":
                while j < len(s) and s[j].isdigit():
                    j += 1
                to = int(s[i + 1:j])
            pot = 2 * behind + level[0] + level[1]
            facing = cur > level[pos]
            d = {
                "street": street,
                "pos": pos,
                "facing": facing,
                "depth": bets,
                "pot_bb": pot / BIG_BLIND,
                "action": tok,
                "offset": base + i,  # where this move starts in `action`
            }
            if facing:
                if faced is None:  # preflop, facing only the big blind
                    d["size"], d["mdf"] = "blind", None
                else:
                    x, mdf, all_in = faced
                    d["size"], d["mdf"] = size_bucket(x, all_in), mdf
            if tok == "b":
                x = (to - cur) / (pot + cur)
                all_in = behind + to >= STACK
                d["bet_x"], d["bet_size"] = x, size_bucket(x, all_in)
                risk = to - level[pos]
                faced = (x, pot / (pot + risk), all_in)
                cur = to
                level[pos] = to
                bets += 1
            elif tok == "c":
                level[pos] = cur
            elif tok not in "kf":
                raise ValueError(f"unknown token {tok!r} in {action!r}")
            out.append(d)
            if tok == "f":
                return out
            pos ^= 1
            i = j
        behind += max(level[0], level[1])
        base += len(s) + 1
    return out


def spot_key(d):
    seat = SEAT[d["pos"]]
    if d["facing"]:
        return (d["street"], seat, depth_label(d["street"], d["depth"]), d["size"])
    return (d["street"], seat, "unopened", "")


class Side:
    def __init__(self):
        self.n = 0
        self.counts = defaultdict(int)
        self.pot = 0.0
        self.mdf = 0.0
        self.mdf_n = 0
        self.bet_x = []

    def add(self, d):
        self.n += 1
        kind = {"f": "fold", "c": "call", "k": "check", "b": "raise" if d["facing"] else "bet"}[d["action"]]
        self.counts[kind] += 1
        self.pot += d["pot_bb"]
        if d.get("mdf") is not None:
            self.mdf += d["mdf"]
            self.mdf_n += 1
        if "bet_x" in d and not d["facing"]:
            self.bet_x.append(d["bet_x"])

    def freq(self, kind):
        return self.counts[kind] / self.n if self.n else 0.0


def z_two_props(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def report(paths, min_n, show_all):
    hands = 0
    sides = defaultdict(lambda: {"us": Side(), "them": Side()})
    for path in paths:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                hands += 1
                for d in decisions(r.get("action", "")):
                    who = "us" if d["pos"] == r["pos"] else "them"
                    sides[spot_key(d)][who].add(d)

    print(f"\n=== {', '.join(paths)}: {hands} hands, spots with ≥{min_n} decisions per side ===")
    rows = []
    for key, s in sides.items():
        us, them = s["us"], s["them"]
        if us.n < min_n or them.n < min_n:
            continue
        main = "fold" if key[2] != "unopened" else "bet"
        if key[3] == "blind":
            main = "fold"  # SB first in: how often it gives up the blind
        gap = us.freq(main) - them.freq(main)
        z = z_two_props(us.freq(main), us.n, them.freq(main), them.n)
        importance = abs(gap) * (us.pot / us.n) * (us.n / hands * 100)
        rows.append((importance, key, main, gap, z, us, them))

    rows.sort(key=lambda r: -r[0])
    if not show_all:
        rows = rows[:25]
    hdr = (f"{'street':<7} {'seat':<4} {'spot':<11} {'size':<8} {'n us':>5} {'n them':>6}  "
           f"{'stat':<5} {'us':>5} {'them':>5} {'gap':>6} {'z':>5}  {'MDF':>4}  "
           f"{'us f/c/r or k/b':<16} {'them':<16} {'imp':>5}")
    print(hdr)
    print("-" * len(hdr))
    for importance, key, main, gap, z, us, them in rows:
        street, seat, spot, size = key
        if spot == "unopened":
            mix_us = f"{us.freq('check'):.2f}/{us.freq('bet'):.2f}"
            mix_them = f"{them.freq('check'):.2f}/{them.freq('bet'):.2f}"
            mdf = "  - "
        else:
            mix_us = f"{us.freq('fold'):.2f}/{us.freq('call'):.2f}/{us.freq('raise'):.2f}"
            mix_them = f"{them.freq('fold'):.2f}/{them.freq('call'):.2f}/{them.freq('raise'):.2f}"
            mdf = f"{us.mdf / us.mdf_n:.2f}" if us.mdf_n else "  - "
        flag = " *" if abs(z) >= 3 else "  "
        print(f"{STREETS[street]:<7} {seat:<4} {spot:<11} {size:<8} {us.n:>5} {them.n:>6}  "
              f"{main:<5} {us.freq(main):>5.2f} {them.freq(main):>5.2f} {gap:>+6.2f} {z:>+5.1f}{flag}"
              f"{mdf:>5}  {mix_us:<16} {mix_them:<16} {importance:>5.1f}")
    print("\n* = |z| ≥ 3 (a gap this size is very unlikely to be sampling noise).")
    print("MDF = minimum defense frequency for the size faced; folding more than 1−MDF")
    print("lets any two cards bluff profitably — a theory reference, not a target.")

    # Bet sizing in unopened spots: the size buckets above only cover bets faced.
    print("\nMedian bet size (pot fraction) when betting unopened:")
    for street in range(1, 4):
        for seat in ("BB", "SB"):
            s = sides.get((street, seat, "unopened", ""))
            if not s or len(s["us"].bet_x) < 30 or len(s["them"].bet_x) < 30:
                continue
            print(f"  {STREETS[street]:<6} {seat}:  us {median(s['us'].bet_x):.2f} (n={len(s['us'].bet_x)})"
                  f"   them {median(s['them'].bet_x):.2f} (n={len(s['them'].bet_x)})")


def self_check():
    # SB opens to 200, BB calls; flop: BB checks, SB bets 100 into 400, BB calls;
    # turn and river checked through.
    ds = decisions("b200c/kb100c/kk/kk")
    assert [d["action"] for d in ds] == ["b", "c", "k", "b", "c", "k", "k", "k", "k"], ds
    sb_open, bb_vs_open, _, sb_cbet, bb_vs_cbet = ds[:5]
    assert sb_open["pos"] == 1 and sb_open["facing"] and sb_open["size"] == "blind"
    assert abs(sb_open["bet_x"] - 0.4) < 1e-9  # (200 − 100) / (150 + 100)
    assert bb_vs_open["pos"] == 0 and bb_vs_open["depth"] == 2 and bb_vs_open["size"] == "40-80%"
    assert sb_cbet["pos"] == 1 and not sb_cbet["facing"] and abs(sb_cbet["bet_x"] - 0.25) < 1e-9
    assert bb_vs_cbet["depth"] == 1 and bb_vs_cbet["size"] == "<40%"
    assert abs(bb_vs_cbet["mdf"] - 0.8) < 1e-9  # 400 / (400 + 100)
    assert abs(bb_vs_cbet["pot_bb"] - 5.0) < 1e-9  # 400 + 100 in the middle

    # Limp, BB checks its option: the option is unopened, not facing.
    ds = decisions("ck/kk/kk/kk")
    assert ds[0]["facing"] and ds[0]["size"] == "blind" and ds[0]["action"] == "c"
    assert not ds[1]["facing"] and ds[1]["pos"] == 0

    # Preflop shove: the BB faces an all-in.
    ds = decisions("b20000c")
    assert ds[1]["size"] == "all-in" and ds[1]["depth"] == 2

    # Flop raise: BB bets 100 into 400, SB raises to 400, BB folds.
    ds = decisions("b200c/b100b400f")
    sb_vs_bet, bb_vs_raise = ds[3], ds[4]
    assert sb_vs_bet["depth"] == 1 and abs(sb_vs_bet["bet_x"] - 0.5) < 1e-9  # (400 − 100) / (500 + 100)
    assert bb_vs_raise["depth"] == 2 and bb_vs_raise["size"] == "40-80%"
    assert abs(bb_vs_raise["mdf"] - 500 / 900) < 1e-9  # risk 400 into 500
    assert bb_vs_raise["action"] == "f" and len(ds) == 5

    # Offsets index the full string: the flop raise above starts at 10.
    assert sb_vs_bet["offset"] == 10 and "b200c/b100b400f"[10:14] == "b400"

    # An all-in run-out leaves empty streets: no decisions there.
    assert len(decisions("b200c/b19800c//")) == 4


if __name__ == "__main__":
    self_check()
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        print(__doc__)
        sys.exit(2)
    min_n = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--min=")), 150)
    report(paths, min_n, "--all" in sys.argv)
