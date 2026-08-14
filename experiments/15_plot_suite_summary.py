#!/usr/bin/env python3
"""Cross-condition summary: game-level werewolf win rates, paired tests, and
judge-based skill-conditional rates.

Usage:
    python experiments/15_plot_suite_summary.py --spec results/suite_spec.json

Spec format: {"conditions": [{"label": str, "results": str, "baseline": str|null,
                              "judged": str|null}]}
Paths relative to results/. "baseline" names the results file to pair against.
"judged" names the 16_judge_games.py output for attribution-conditional rates.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"

ATTRIBUTION_MIN = 7  # games with outcome_attribution >= this count as "decided by play"


def load_games(path: str) -> dict[int, dict]:
    with open(RESULTS_DIR / path) as f:
        data = json.load(f)
    return {g["seed"]: g for g in data["games"] if g["winner"] != "ERROR"}


def load_judged(path: str) -> dict[int, dict]:
    with open(RESULTS_DIR / path) as f:
        data = json.load(f)
    return {g["seed"]: g for g in data["games"] if "error" not in g}


def ww_won(game: dict) -> bool:
    return game["winner"] == "WEREWOLF"


def wilson_ci(wins: int, total: int, confidence: float = 0.95):
    if total == 0:
        return 0.0, (0.0, 0.0)
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = wins / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return p_hat, (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_p(cond: dict[int, dict], base: dict[int, dict]) -> tuple[float, int, int, int]:
    shared = sorted(set(cond) & set(base))
    b = sum(1 for s in shared if ww_won(cond[s]) and not ww_won(base[s]))
    c = sum(1 for s in shared if not ww_won(cond[s]) and ww_won(base[s]))
    p = scipy_stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return p, b, c, len(shared)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="suite_summary")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = RESULTS_DIR / spec_path
    with open(spec_path) as f:
        spec = json.load(f)

    rows = []
    for cond in spec["conditions"]:
        games = load_games(cond["results"])
        wins = sum(1 for g in games.values() if ww_won(g))
        rate, (lo, hi) = wilson_ci(wins, len(games))
        row = {
            "label": cond["label"], "n": len(games), "wins": wins,
            "ww_win_rate": rate, "ci": [lo, hi],
            "mcnemar_p": None, "flips_to_ww": None, "flips_to_v": None, "paired_n": None,
            "skill_n": None, "skill_ww_wins": None, "skill_ww_rate": None,
            "luck_n": None, "luck_ww_wins": None,
            "mean_attribution": None,
        }
        if cond.get("baseline"):
            p, b, c, n = mcnemar_p(games, load_games(cond["baseline"]))
            row.update({"mcnemar_p": p, "flips_to_ww": b, "flips_to_v": c, "paired_n": n})
        if cond.get("judged"):
            judged = load_judged(cond["judged"])
            oa = {s: j["outcome_attribution"] for s, j in judged.items()}
            row["mean_attribution"] = round(float(np.mean(list(oa.values()))), 2)
            skill = [s for s in games if oa.get(s, -1) >= ATTRIBUTION_MIN]
            luck = [s for s in games if 0 <= oa.get(s, -1) < ATTRIBUTION_MIN]
            row["skill_n"] = len(skill)
            row["skill_ww_wins"] = sum(1 for s in skill if ww_won(games[s]))
            row["skill_ww_rate"] = row["skill_ww_wins"] / len(skill) if skill else None
            row["luck_n"] = len(luck)
            row["luck_ww_wins"] = sum(1 for s in luck if ww_won(games[s]))
        rows.append(row)

    with open(RESULTS_DIR / f"{args.output}.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ── Forest plot: overall + skill-conditional win rates ──
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(rows) + 2))
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        lo, hi = row["ci"]
        color = "#6B9EC4" if row["mcnemar_p"] is None else "#C4483F"
        ax.plot([lo * 100, hi * 100], [y + 0.12, y + 0.12], color=color, linewidth=2)
        ax.plot(row["ww_win_rate"] * 100, y + 0.12, "o", color=color, markersize=7)
        note = "" if row["mcnemar_p"] is None else f"  p={row['mcnemar_p']:.2f}"
        ax.annotate(f"{row['ww_win_rate']*100:.0f}%{note}",
                    (hi * 100 + 1.5, y + 0.12), va="center", fontsize=9)
        if row["skill_ww_rate"] is not None:
            r, (slo, shi) = wilson_ci(row["skill_ww_wins"], row["skill_n"])
            ax.plot([slo * 100, shi * 100], [y - 0.22, y - 0.22],
                    color="#999999", linewidth=1.4)
            ax.plot(r * 100, y - 0.22, "s", color="#555555", markersize=5)
            ax.annotate(f"{r*100:.0f}% (skill, n={row['skill_n']})",
                        (shi * 100 + 1.5, y - 0.22), va="center", fontsize=8,
                        color="#555555")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']}\n(n={r['n']}, attr {r['mean_attribution']})"
                        for r in rows], fontsize=9)
    ax.set_xlabel("Werewolf-team game win rate (%)")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="#cccccc", linewidth=1, zorder=0)
    ax.set_title("ONUW replication: wolf win rate by condition — overall (round) "
                 f"and attribution≥{ATTRIBUTION_MIN} games (square)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = RESULTS_DIR / f"{args.output}.png"
    plt.savefig(out, dpi=150, facecolor="white")
    print(f"Saved {out}")

    for r in rows:
        paired = ("" if r["mcnemar_p"] is None else
                  f"  vs base: +{r['flips_to_ww']}/-{r['flips_to_v']} p={r['mcnemar_p']:.3f}")
        skill = ("" if r["skill_ww_rate"] is None else
                 f"  skill-games: {r['skill_ww_wins']}/{r['skill_n']} "
                 f"({100*r['skill_ww_rate']:.0f}%)  luck-games: {r['luck_ww_wins']}/{r['luck_n']}")
        print(f"  {r['label']:<28} {r['wins']:>2}/{r['n']} ({100*r['ww_win_rate']:.0f}%)"
              f"{paired}{skill}")


if __name__ == "__main__":
    main()
