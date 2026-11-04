"""Benchmark harness. Numbers, not vibes.

    python eval.py                                   # 120 runs per condition
    python eval.py --runs 40 --out results.csv --plot results.png
    python eval.py --conditions keeper_on            # single condition
    python eval.py --mode llm --runs 3               # small LLM sample

Every condition runs the SAME seeds on the SAME maps, so a difference
between conditions cannot come from the dungeon layout or from where the
monsters happened to spawn.

Default: 3 maps x 40 seeds = 120 games per condition, 240 games total.

Win rates come with a Wilson 95% confidence interval. With 120 games the
interval is roughly +/- 8 points, which is worth knowing before you claim
that one condition beat another.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from engine import build_state
from sim import run_headless
from world import GameMap

DEFAULT_MAPS = ["maps/warren.txt", "maps/halls.txt", "maps/crypt.txt"]

CONDITIONS = {
    # name          : (enemy commander on?, note for the report)
    "keeper_on": (True, "Dungeon Keeper issues orders to the monsters"),
    "keeper_off": (False, "every monster acts alone, no shared intel"),
}

FIELDS = [
    "condition", "mode", "map", "seed", "outcome", "won", "turns",
    "heroes_lost", "first_death_turn", "deaths_by_role", "chests",
    "total_chests", "survivors", "hp_left", "monsters_killed",
    "orders_issued", "order_mix", "llm_calls", "llm_actions",
    "fallback_actions", "invalid_actions",
]


# --- statistics ------------------------------------------------------------

def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test. Returns (z, two-sided p).

    This, not overlapping confidence intervals, is the right way to compare
    two win rates. Two 95% intervals can overlap while the difference is
    still significant, so judging by eye from the error bars is a mistake
    that would cost you marks.
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # two-sided p from the normal CDF via erf
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Honest at small n, unlike the normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --- running ---------------------------------------------------------------

def run_one(map_path: str, condition: str, mode: str, seed: int,
            monsters: int, max_turns: int) -> dict:
    keeper, _ = CONDITIONS[condition]
    gmap = GameMap.from_file(map_path)
    state = build_state(gmap, seed=seed, n_monsters=monsters,
                        max_turns=max_turns)
    row = run_headless(state, use_llm=(mode == "llm"), enemy_commander=keeper)
    row.update(condition=condition, mode=mode, map=Path(map_path).stem,
               seed=seed)
    return row


# --- reporting -------------------------------------------------------------

def summarize(rows: list[dict], condition: str) -> dict:
    sub = [r for r in rows if r["condition"] == condition]
    if not sub:
        return {}
    n = len(sub)
    wins = [r for r in sub if r["won"]]
    lo, hi = wilson(len(wins), n)
    outcomes = Counter(r["outcome"] for r in sub)
    deaths = Counter()
    for r in sub:
        for role in filter(None, str(r["deaths_by_role"]).split("|")):
            deaths[role] += 1
    first = [r["first_death_turn"] for r in sub
             if isinstance(r["first_death_turn"], int)]
    acts = sum(r["llm_actions"] + r["fallback_actions"] for r in sub)
    return {
        "condition": condition,
        "n": n,
        "wins": len(wins),
        "win_rate": len(wins) / n,
        "ci": (lo, hi),
        "turns_to_win": statistics.mean(r["turns"] for r in wins) if wins
        else float("nan"),
        "turns_sd": statistics.pstdev([r["turns"] for r in wins]) if len(wins) > 1
        else 0.0,
        "heroes_lost": statistics.mean(r["heroes_lost"] for r in sub),
        "chest_rate": statistics.mean(
            r["chests"] / r["total_chests"] for r in sub),
        "monsters_killed": statistics.mean(r["monsters_killed"] for r in sub),
        "wiped": outcomes["wiped"] / n,
        "timeout": outcomes["timeout"] / n,
        "first_death": statistics.mean(first) if first else float("nan"),
        "deaths_by_role": deaths,
        "invalid_rate": (sum(r["invalid_actions"] for r in sub) / acts
                         if acts else 0.0),
        "llm_calls": sum(r["llm_calls"] for r in sub),
    }


def print_report(stats: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("BENCHMARK")
    print("=" * 78)

    head = (f"{'condition':<12}{'n':>5}{'win rate':>18}{'turns to win':>15}"
            f"{'heroes lost':>13}{'chests':>9}")
    print(head)
    print("-" * 78)
    for s in stats:
        lo, hi = s["ci"]
        print(f"{s['condition']:<12}{s['n']:>5}"
              f"{s['win_rate']:>9.1%} [{lo:.0%}-{hi:.0%}]"
              f"{s['turns_to_win']:>14.1f}"
              f"{s['heroes_lost']:>13.2f}"
              f"{s['chest_rate']:>9.0%}")

    print("\nHow the squad lost, and who paid for it")
    print("-" * 78)
    for s in stats:
        roles = s["deaths_by_role"]
        mix = ", ".join(f"{r} {c}" for r, c in roles.most_common()) or "nobody"
        print(f"{s['condition']:<12} wiped {s['wiped']:>5.0%}   "
              f"timeout {s['timeout']:>5.0%}   "
              f"first death turn {s['first_death']:>5.1f}")
        print(f"{'':<12} deaths: {mix}")
        print(f"{'':<12} monsters killed {s['monsters_killed']:.2f}/game   "
              f"invalid actions {s['invalid_rate']:.1%}   "
              f"LLM calls {s['llm_calls']}")

    if len(stats) == 2:
        a, b = stats
        delta = (a["win_rate"] - b["win_rate"]) * 100
        z, pval = two_proportion_test(a["wins"], a["n"], b["wins"], b["n"])
        print("\nAblation")
        print("-" * 78)
        print(f"{a['condition']} minus {b['condition']}: "
              f"{delta:+.1f} percentage points of squad win rate, "
              f"{a['turns_to_win'] - b['turns_to_win']:+.1f} turns to win.")
        print(f"Two-proportion z-test: z = {z:.2f}, p = {pval:.4f} "
              f"({'significant' if pval < 0.05 else 'not significant'} "
              f"at alpha = 0.05).")
        if pval >= 0.05:
            print("Not significant yet. Either the commander genuinely does "
                  "little, or n is too small: raise --runs and rerun.")
        else:
            print("The enemy commander changes the outcome. Note the error "
                  "bars in the plot may still overlap; that is not the test.")


def write_plot(stats: list[dict], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not installed, skipping {path}")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = [s["condition"] for s in stats]

    rates = [s["win_rate"] * 100 for s in stats]
    errs = [[(s["win_rate"] - s["ci"][0]) * 100 for s in stats],
            [(s["ci"][1] - s["win_rate"]) * 100 for s in stats]]
    axes[0].bar(names, rates, yerr=errs, capsize=6, color="#5694d6")
    axes[0].set_title("Squad win rate (95% CI)")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 100)

    wins = [s["win_rate"] * 100 for s in stats]
    wiped = [s["wiped"] * 100 for s in stats]
    timeout = [s["timeout"] * 100 for s in stats]
    axes[1].bar(names, wins, color="#5aa85a", label="win")
    axes[1].bar(names, wiped, bottom=wins, color="#d66060", label="wiped")
    axes[1].bar(names, timeout, bottom=[w + x for w, x in zip(wins, wiped)],
                color="#9a9aa8", label="out of turns")
    axes[1].set_title("How games ended")
    axes[1].set_ylabel("% of games")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)

    width = 0.35
    roles = ["Tank", "DPS", "Healer", "Scout"]
    for i, s in enumerate(stats):
        vals = [s["deaths_by_role"].get(r, 0) / s["n"] for r in roles]
        axes[2].bar([x + i * width for x in range(len(roles))], vals,
                    width, label=s["condition"])
    axes[2].set_xticks([x + width / 2 for x in range(len(roles))])
    axes[2].set_xticklabels(roles)
    axes[2].set_title("Death rate per role")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nPlot written to {path}")
    return True


def load_csv(paths: list[str]) -> list[dict]:
    """Read benchmark CSVs back in, so chunked runs can be merged."""
    ints = {"turns", "heroes_lost", "chests", "total_chests", "survivors",
            "hp_left", "monsters_killed", "orders_issued", "llm_calls",
            "llm_actions", "fallback_actions", "invalid_actions", "seed"}
    rows: list[dict] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                row = dict(raw)
                row["won"] = str(row["won"]).lower() == "true"
                for key in ints:
                    if row.get(key, "") != "":
                        row[key] = int(row[key])
                row["first_death_turn"] = (int(row["first_death_turn"])
                                           if row["first_death_turn"] else "")
                rows.append(row)
    return rows


# --- entrypoint ------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Warband benchmark")
    p.add_argument("--maps", nargs="+", default=DEFAULT_MAPS)
    p.add_argument("--conditions", nargs="+",
                   default=["keeper_on", "keeper_off"],
                   choices=list(CONDITIONS))
    p.add_argument("--mode", choices=["scripted", "llm"], default="scripted")
    p.add_argument("--runs", type=int, default=40,
                   help="seeds per map per condition (3 maps x 40 = 120)")
    p.add_argument("--monsters", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--out", default="results.csv")
    p.add_argument("--plot", default="")
    p.add_argument("--quiet", action="store_true",
                   help="no per-run line, only the summary")
    p.add_argument("--from-csv", nargs="+", default=[],
                   help="skip running; build the report from existing CSVs. "
                        "Pass several to merge chunked runs.")
    args = p.parse_args()

    if args.from_csv:
        rows = load_csv(args.from_csv)
        conditions = sorted({r["condition"] for r in rows},
                            key=lambda c: c != "keeper_on")
        stats = [s for s in (summarize(rows, c) for c in conditions) if s]
        print(f"Loaded {len(rows)} games from "
              f"{', '.join(args.from_csv)}")
        print_report(stats)
        if args.plot:
            write_plot(stats, args.plot)
        return

    total = len(args.conditions) * len(args.maps) * args.runs
    print(f"{total} games: {len(args.conditions)} conditions x "
          f"{len(args.maps)} maps x {args.runs} seeds, mode={args.mode}")
    if args.mode == "llm":
        print("WARNING: LLM mode makes roughly 200 API calls per game. "
              f"{total} games is a large bill. Start small.")

    rows: list[dict] = []
    started = time.time()
    done = 0
    for condition in args.conditions:
        for map_path in args.maps:
            for seed in range(1, args.runs + 1):
                row = run_one(map_path, condition, args.mode, seed,
                              args.monsters, args.max_turns)
                rows.append(row)
                done += 1
                if args.quiet:
                    if done % 20 == 0:
                        print(f"  {done}/{total} ...")
                else:
                    flag = {"win": "W", "wiped": "X", "timeout": "."}[
                        row["outcome"]]
                    print(f"{flag} {condition:<11}{row['map']:<8}"
                          f"seed {seed:<4}{row['turns']:>4}t  "
                          f"{row['chests']}/{row['total_chests']} chests  "
                          f"{row['survivors']}/4 alive")

    elapsed = time.time() - started
    stats = [s for s in (summarize(rows, c) for c in args.conditions) if s]
    print_report(stats)
    print(f"\n{total} games in {elapsed:.1f}s "
          f"({elapsed / total * 1000:.0f} ms/game)")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"CSV written to {args.out}")

    if args.plot:
        write_plot(stats, args.plot)


if __name__ == "__main__":
    main()
