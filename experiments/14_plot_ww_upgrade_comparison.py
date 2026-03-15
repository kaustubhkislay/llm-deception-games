#!/usr/bin/env python3
"""
Plot comparison of baseline vs experimental (upgraded werewolves).

Left panel: werewolf win rate per instance (bar chart with Wilson CIs).
Right panel: 2×2 paired outcome matrix (game-level, shared seeds).
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"


def wilson_ci(wins, total, confidence=0.95):
    if total == 0:
        return 0.0, (0.0, 0.0)
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = wins / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return p_hat, (max(0, center - margin), min(1, center + margin))


def compute_upgraded_ww_stats(games, config_dir, upgraded_pattern):
    """Compute per-instance werewolf win rate for upgraded players."""
    wins, total = 0, 0
    for game in games:
        if game["winner"] == "ERROR":
            continue
        seed = game["seed"]
        config_path = Path(config_dir) / f"config_{seed:03d}.json"
        with open(config_path) as f:
            config = json.load(f)
        upgraded_players = {
            n for n, m in zip(config["names"], config["models"])
            if upgraded_pattern in m["model"]
        }
        for p in game["players"]:
            if p["name"] in upgraded_players and p["starting_role"] == "WEREWOLF":
                total += 1
                if p["won"]:
                    wins += 1
    return wins, total


def mcnemar_pvalue(b, c):
    """McNemar's test p-value (two-sided) for off-diagonal counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return scipy_stats.binomtest(b, n, 0.5).pvalue


def significance_stars(p):
    if p < 0.001:
        return " ***"
    elif p < 0.01:
        return " **"
    elif p < 0.05:
        return " *"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-analysis", type=str, required=True)
    parser.add_argument("--baseline-batch", type=str, required=True)
    parser.add_argument("--experimental-batch", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--upgraded-pattern", type=str, default="pro",
                        help="Substring to match in model name to identify upgraded players")
    parser.add_argument("--upgraded-label", type=str, default=None,
                        help="Short label for the upgraded model")
    parser.add_argument("--baseline-label", type=str, default=None,
                        help="Short label for the baseline model")
    args = parser.parse_args()

    def resolve(p):
        path = Path(p)
        return path if path.is_absolute() else RESULTS_DIR / p

    with open(resolve(args.baseline_analysis)) as f:
        baseline_analysis = json.load(f)
    with open(resolve(args.baseline_batch)) as f:
        baseline_batch = json.load(f)
    with open(resolve(args.experimental_batch)) as f:
        exp_batch = json.load(f)

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = Path(__file__).parent.parent / config_dir

    upgraded_label = args.upgraded_label or args.upgraded_pattern.title()
    baseline_label = args.baseline_label or "baseline"

    # ── Werewolf win rate per instance ──
    base_ww = baseline_analysis["by_starting_role"]["WEREWOLF"]
    base_wr, (base_cl, base_ch) = wilson_ci(base_ww["wins"], base_ww["total"])
    n_base_games = baseline_analysis["completed_games"]

    exp_wins, exp_total = compute_upgraded_ww_stats(
        exp_batch["games"], config_dir, args.upgraded_pattern)
    exp_wr, (exp_cl, exp_ch) = wilson_ci(exp_wins, exp_total)
    n_exp_games = len([g for g in exp_batch["games"] if g["winner"] != "ERROR"])

    # ── Paired outcome matrix (game-level) ──
    baseline_seeds = {g["seed"]: g["winner"] for g in baseline_batch["games"]
                      if g["winner"] != "ERROR"}
    exp_seeds = {g["seed"]: g["winner"] for g in exp_batch["games"]
                 if g["winner"] != "ERROR"}
    shared_seeds = sorted(set(baseline_seeds) & set(exp_seeds))

    # matrix[row][col]: row = baseline outcome, col = experimental outcome
    #   row 0 = baseline village wins, row 1 = baseline wolf wins
    #   col 0 = experimental village wins, col 1 = experimental wolf wins
    matrix = np.zeros((2, 2), dtype=int)
    for seed in shared_seeds:
        br = 0 if baseline_seeds[seed] == "VILLAGE" else 1
        ec = 0 if exp_seeds[seed] == "VILLAGE" else 1
        matrix[br, ec] += 1

    # McNemar test: off-diagonal = Village→Wolf (matrix[0,1]) vs Wolf→Village (matrix[1,0])
    b, c = int(matrix[0, 1]), int(matrix[1, 0])
    p_val = mcnemar_pvalue(b, c)
    stars = significance_stars(p_val)

    # ── Plot ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    COLOR_BASE = "#7BAFD4"
    COLOR_EXP = "#E8963E"
    COLOR_V = "#B8D4E8"  # light blue (experimental → village)
    COLOR_W = "#D4878F"  # pink/red (experimental → wolf)

    fig, (ax_bar, ax_mat) = plt.subplots(1, 2, figsize=(12, 4.5),
                                          gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.4})
    fig.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.12)

    fig.suptitle(
        f"{upgraded_label} werewolves vs {baseline_label} baseline ({len(shared_seeds)} games)",
        fontsize=16, fontweight="bold", y=0.95,
    )

    # ── Bar chart ──
    bars = ax_bar.bar(
        [0, 1],
        [base_wr * 100, exp_wr * 100],
        color=[COLOR_BASE, COLOR_EXP],
        width=0.55,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax_bar.errorbar(
        [0, 1],
        [base_wr * 100, exp_wr * 100],
        yerr=[
            [base_wr * 100 - base_cl * 100, exp_wr * 100 - exp_cl * 100],
            [base_ch * 100 - base_wr * 100, exp_ch * 100 - exp_wr * 100],
        ],
        fmt="none", color="#333333", capsize=5, capthick=1.2, linewidth=1.2, zorder=3,
    )

    ax_bar.annotate(f"{base_wr*100:.1f}%", (0, base_ch * 100 + 2),
                    ha="center", fontsize=12, fontweight="bold", color="#4A7A9B")
    exp_label_text = f"{exp_wr*100:.1f}%{stars}"
    ax_bar.annotate(exp_label_text, (1, exp_ch * 100 + 2),
                    ha="center", fontsize=12, fontweight="bold", color="#B06A20")

    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels([
        f"All {baseline_label}\n(n={base_ww['total']} instances)",
        f"{upgraded_label} werewolves\n(n={exp_total} instances)",
    ], fontsize=10)
    ax_bar.set_ylabel("Werewolf Win Rate (%)")
    ax_bar.set_title("Werewolf win rate", fontweight="bold", pad=8)
    ax_bar.set_ylim(0, 100)
    ax_bar.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax_bar.yaxis.grid(True, alpha=0.25, linewidth=0.5)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── 2×2 paired outcome matrix ──
    ax_mat.set_title(f"Paired outcome comparison (n={len(shared_seeds)})",
                     fontweight="bold", pad=8)
    ax_mat.set_xlim(-0.6, 1.6)
    ax_mat.set_ylim(-0.6, 1.6)
    ax_mat.invert_yaxis()
    ax_mat.axis("off")

    cell_colors = [
        [COLOR_V, COLOR_W],
        [COLOR_V, COLOR_W],
    ]
    cell_labels = [
        ["Village → Village", "Village → Wolf"],
        ["Wolf → Village", "Wolf → Wolf"],
    ]

    total_shared = len(shared_seeds)
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                 facecolor=cell_colors[i][j], alpha=0.6,
                                 edgecolor="#aaaaaa", linewidth=1, zorder=2)
            ax_mat.add_patch(rect)

            ax_mat.text(j, i - 0.18, cell_labels[i][j],
                        ha="center", va="center", fontsize=9, color="#444444", zorder=3)
            ax_mat.text(j, i + 0.12, str(matrix[i, j]),
                        ha="center", va="center", fontsize=24, fontweight="bold", zorder=3)

    # Row labels
    ax_mat.text(-0.65, 0, "Baseline:\nvillage wins", ha="center", va="center", fontsize=9.5)
    ax_mat.text(-0.65, 1, "Baseline:\nwerewolves win", ha="center", va="center", fontsize=9.5)

    # Column labels
    ax_mat.text(0, 1.65, "Experimental:\nvillage wins", ha="center", va="top", fontsize=9.5)
    ax_mat.text(1, 1.65, "Experimental:\nwerewolves win", ha="center", va="top", fontsize=9.5)

    output_path = RESULTS_DIR / args.output
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")
    print(f"  Baseline WW win rate: {base_wr*100:.1f}% ({base_ww['wins']}/{base_ww['total']})")
    print(f"  Experimental WW win rate: {exp_wr*100:.1f}% ({exp_wins}/{exp_total})")
    print(f"  Shared seeds: {len(shared_seeds)}")
    print(f"  McNemar p={p_val:.4f}{stars}")


if __name__ == "__main__":
    main()
