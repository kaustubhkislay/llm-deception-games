#!/usr/bin/env python3
"""Paired werewolf-upgrade analysis on all seeds and on "clean" seeds.

A seed is clean when every starting-werewolf seat (the seats that get the
upgraded model) still holds an evil-team card (WEREWOLF or MINION) at dawn.
Night actions are seed-forced, so cleanliness is fixed by the deal before any
discussion happens and is identical across conditions. Cleanliness is read
from the baseline game for each seed.

On non-clean seeds the upgraded model plays at least partly for the village,
so the werewolf-team win does not measure that model's deception. Restricting
to clean seeds scores the upgraded model only where it plays the evil side.

For each condition and each seed set this reports the paired difference in
werewolf-team win rate with a 95% CI, the exact McNemar p-value, and a
Holm-adjusted p across the upgrade conditions.

Usage:
    python experiments/17_clean_seed_analysis.py --spec suite_spec.json
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"

EVIL_CARDS = {"WEREWOLF", "MINION"}


def load_games(path: str) -> dict[int, dict]:
    with open(RESULTS_DIR / path) as f:
        data = json.load(f)
    return {g["seed"]: g for g in data["games"] if g["winner"] != "ERROR"}


def ww_won(game: dict) -> bool:
    return game["winner"] == "WEREWOLF"


def is_clean(game: dict) -> bool:
    return all(
        p["ending_role"] in EVIL_CARDS
        for p in game["players"]
        if p["starting_role"] == "WEREWOLF"
    )


def paired_stats(cond: dict[int, dict], base: dict[int, dict], seeds: list[int]) -> dict:
    n = len(seeds)
    b = sum(1 for s in seeds if ww_won(cond[s]) and not ww_won(base[s]))
    c = sum(1 for s in seeds if not ww_won(cond[s]) and ww_won(base[s]))
    diff = (b - c) / n
    # Wald CI for a paired difference in proportions
    se = np.sqrt(b + c - (b - c) ** 2 / n) / n
    z = scipy_stats.norm.ppf(0.975)
    p = scipy_stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {
        "n": n,
        "cond_rate": sum(ww_won(cond[s]) for s in seeds) / n,
        "base_rate": sum(ww_won(base[s]) for s in seeds) / n,
        "flips_up": b, "flips_down": c,
        "diff": diff, "ci": [diff - z * se, diff + z * se],
        "mcnemar_p": p,
    }


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="clean_seed_analysis")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = RESULTS_DIR / spec_path
    with open(spec_path) as f:
        spec = json.load(f)

    rows = []
    for cond in spec["conditions"]:
        if not cond["baseline"]:
            continue
        games = load_games(cond["results"])
        base = load_games(cond["baseline"])
        shared = sorted(set(games) & set(base))
        clean = [s for s in shared if is_clean(base[s])]
        rows.append({
            "label": cond["label"],
            "clean_seeds": clean,
            "all": paired_stats(games, base, shared),
            "clean": paired_stats(games, base, clean),
        })

    for key in ("all", "clean"):
        for row, adj in zip(rows, holm([r[key]["mcnemar_p"] for r in rows])):
            row[key]["holm_p"] = adj

    print(f"{'condition':30} {'seeds':5} {'n':>3} {'cond':>5} {'base':>5} "
          f"{'diff [95% CI]':>22} {'flips':>7} {'p':>6} {'holm':>6}")
    for row in rows:
        for key in ("all", "clean"):
            r = row[key]
            print(f"{row['label'][:30]:30} {key:5} {r['n']:3d} "
                  f"{100 * r['cond_rate']:5.0f} {100 * r['base_rate']:5.0f} "
                  f"{100 * r['diff']:+6.1f} [{100 * r['ci'][0]:+5.1f}, {100 * r['ci'][1]:+5.1f}] "
                  f"{'+%d/-%d' % (r['flips_up'], r['flips_down']):>7} "
                  f"{r['mcnemar_p']:6.3f} {r['holm_p']:6.3f}")

    with open(RESULTS_DIR / f"{args.output}.json", "w") as f:
        json.dump({"spec": str(spec_path.name), "rows": rows}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.5))
    y = np.arange(len(rows))
    for offset, key, color in ((-0.15, "all", "#9a9a9a"), (0.15, "clean", "#c0392b")):
        diffs = [100 * r[key]["diff"] for r in rows]
        lo = [100 * (r[key]["diff"] - r[key]["ci"][0]) for r in rows]
        hi = [100 * (r[key]["ci"][1] - r[key]["diff"]) for r in rows]
        label = (f"all seeds (n={rows[0]['all']['n']})" if key == "all"
                 else f"clean seeds (n={rows[0]['clean']['n']})")
        ax.errorbar(diffs, y + offset, xerr=[lo, hi], fmt="o", color=color,
                    capsize=3, label=label)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("paired change in werewolf-team win rate vs baseline (points, 95% CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{args.output}.png", dpi=150)
    print(f"\nwrote results/{args.output}.json and results/{args.output}.png")


if __name__ == "__main__":
    main()
