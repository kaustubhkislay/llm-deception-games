#!/usr/bin/env python3
"""Plot comparison of baseline (all Flash) vs experimental (Pro werewolves)."""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"

ROLE_ORDER = ["WEREWOLF", "MINION", "SEER", "ROBBER", "TROUBLEMAKER", "DRUNK", "VILLAGER"]
ROLE_SHORT = {
    "WEREWOLF": "Werewolf", "MINION": "Minion", "SEER": "Seer",
    "ROBBER": "Robber", "TROUBLEMAKER": "Trouble-\nmaker", "DRUNK": "Drunk", "VILLAGER": "Villager",
}
EVIL_ROLES = {"WEREWOLF", "MINION"}
EMPTY_ROLE = {"win_rate": 0, "ci_low": 0, "ci_high": 0, "total": 0, "wins": 0}


def wilson_ci(wins, total, confidence=0.95):
    if total == 0:
        return 0.0, (0.0, 0.0)
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = wins / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return p_hat, (max(0, center - margin), min(1, center + margin))


def compute_pro_role_stats(games, role_key, config_dir):
    role_stats = defaultdict(lambda: {"wins": 0, "total": 0})
    for game in games:
        if game["winner"] == "ERROR":
            continue
        seed = game["seed"]
        config_path = Path(config_dir) / f"config_{seed:03d}.json"
        with open(config_path) as f:
            config = json.load(f)
        pro_players = {n for n, m in zip(config["names"], config["models"]) if "pro" in m["model"]}
        for p in game["players"]:
            if p["name"] not in pro_players:
                continue
            role = p[role_key]
            role_stats[role]["total"] += 1
            if p["won"]:
                role_stats[role]["wins"] += 1
    result = {}
    for role, s in role_stats.items():
        wr, (cl, ch) = wilson_ci(s["wins"], s["total"])
        result[role] = {"win_rate": wr, "ci_low": cl, "ci_high": ch, "total": s["total"], "wins": s["wins"]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-analysis", type=str, required=True)
    parser.add_argument("--baseline-batch", type=str, required=True)
    parser.add_argument("--experimental-batch", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="13_gem31pro_ww_comparison.png")
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

    base_by_start = baseline_analysis["by_starting_role"]
    base_by_end = baseline_analysis["by_ending_role"]
    exp_by_start = compute_pro_role_stats(exp_batch["games"], "starting_role", config_dir)
    exp_by_end = compute_pro_role_stats(exp_batch["games"], "ending_role", config_dir)

    # Paired outcome matrix
    baseline_seeds = {g["seed"]: g["winner"] for g in baseline_batch["games"]}
    exp_seeds = {g["seed"]: g["winner"] for g in exp_batch["games"] if g["winner"] != "ERROR"}
    shared_seeds = sorted(set(baseline_seeds) & set(exp_seeds))

    matrix = np.zeros((2, 2), dtype=int)
    for seed in shared_seeds:
        bw, ew = baseline_seeds[seed], exp_seeds[seed]
        r = 0 if ew == "VILLAGE" else 1
        c = 1 if bw == "VILLAGE" else 0
        matrix[r, c] += 1

    n_exp = len([g for g in exp_batch["games"] if g["winner"] != "ERROR"])
    n_base = baseline_analysis["completed_games"]

    # ── Styling ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    COLOR_BASE = "#7BAFD4"
    COLOR_EXP = "#E8963E"
    BAR_WIDTH = 0.34

    fig = plt.figure(figsize=(16, 9.5), facecolor="white")
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.8, 1], hspace=0.45, wspace=0.35,
                           left=0.06, right=0.96, top=0.92, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    fig.suptitle(
        f"Gemini 3.1 Pro werewolves vs Flash baseline",
        fontsize=15, fontweight="bold", y=0.97,
    )

    def plot_grouped_bars(ax, base_data, exp_data, title):
        roles = [r for r in ROLE_ORDER if r in base_data]
        x = np.arange(len(roles))

        base_wr = [base_data[r]["win_rate"] * 100 for r in roles]
        base_cl = [base_data[r]["ci_low"] * 100 for r in roles]
        base_ch = [base_data[r]["ci_high"] * 100 for r in roles]
        base_err_lo = [max(0, w - c) for w, c in zip(base_wr, base_cl)]
        base_err_hi = [max(0, c - w) for w, c in zip(base_wr, base_ch)]
        base_n = [base_data[r]["total"] for r in roles]

        has_exp = [exp_data.get(r, EMPTY_ROLE)["total"] > 0 for r in roles]
        exp_wr = [exp_data.get(r, EMPTY_ROLE)["win_rate"] * 100 for r in roles]
        exp_cl = [exp_data.get(r, EMPTY_ROLE)["ci_low"] * 100 for r in roles]
        exp_ch = [exp_data.get(r, EMPTY_ROLE)["ci_high"] * 100 for r in roles]
        exp_err_lo = [max(0, w - c) for w, c in zip(exp_wr, exp_cl)]
        exp_err_hi = [max(0, c - w) for w, c in zip(exp_wr, exp_ch)]
        exp_n = [exp_data.get(r, EMPTY_ROLE)["total"] for r in roles]

        ax.bar(x - BAR_WIDTH / 2, base_wr, BAR_WIDTH, color=COLOR_BASE, alpha=0.85,
               edgecolor="white", linewidth=0.6, zorder=2)
        ax.errorbar(x - BAR_WIDTH / 2, base_wr, yerr=[base_err_lo, base_err_hi],
                    fmt="none", color="#444444", capsize=3, capthick=1, linewidth=1, zorder=3)

        first_exp = True
        for i in range(len(roles)):
            if has_exp[i]:
                ax.bar(i + BAR_WIDTH / 2, exp_wr[i], BAR_WIDTH, color=COLOR_EXP, alpha=0.85,
                       edgecolor="white", linewidth=0.6, zorder=2)
                ax.errorbar(i + BAR_WIDTH / 2, exp_wr[i],
                            yerr=[[exp_err_lo[i]], [exp_err_hi[i]]],
                            fmt="none", color="#444444", capsize=3, capthick=1, linewidth=1, zorder=3)
                ax.annotate(f"n={exp_n[i]}", (i + BAR_WIDTH / 2, 2),
                            ha="center", fontsize=7.5, color="#9A6520")
                first_exp = False

        for i in range(len(roles)):
            ax.annotate(f"n={base_n[i]}", (i - BAR_WIDTH / 2, 2),
                        ha="center", fontsize=7.5, color="#5A8EAF")

        ax.set_xticks(x)
        ax.set_xticklabels([ROLE_SHORT.get(r, r) for r in roles], fontsize=9.5)
        ax.set_ylabel("Win Rate (%)")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylim(0, 108)
        ax.set_xlim(-0.6, len(roles) - 0.4)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_grouped_bars(ax1, base_by_start, exp_by_start, "Win rate by initial role (start of night)")
    plot_grouped_bars(ax2, base_by_end, exp_by_end, "Win rate by final role (end of night)")

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_BASE, alpha=0.85, edgecolor="white",
                       label=f"Baseline — all Flash (n={n_base} games)"),
        mpatches.Patch(facecolor=COLOR_EXP, alpha=0.85, edgecolor="white",
                       label=f"Pro WW players only (n={n_exp} games)"),
    ]
    ax1.legend(handles=legend_handles, fontsize=8.5, loc="upper left",
               framealpha=0.9, edgecolor="#cccccc")

    # ── 2×2 paired-outcome matrix ──
    cell_bg = [
        ["#B5DEAD", "#E0E0E0"],
        ["#E0E0E0", "#E8A0A0"],
    ]
    row_labels = ["Exp: Village wins", "Exp: Wolf wins"]
    col_labels = ["Base: Wolf wins", "Base: Village wins"]
    total = matrix.sum()

    ax3.set_xlim(-1.0, 2.3)
    ax3.set_ylim(-1.0, 2.3)
    ax3.invert_yaxis()

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92,
                                 facecolor=cell_bg[i][j], alpha=0.75,
                                 edgecolor="#888888", linewidth=1.2, zorder=2)
            ax3.add_patch(rect)
            count = matrix[i, j]
            pct = count / total * 100 if total > 0 else 0
            ax3.text(j, i - 0.05, str(count), ha="center", va="center",
                     fontsize=22, fontweight="bold", zorder=3)
            ax3.text(j, i + 0.28, f"({pct:.0f}%)", ha="center", va="center",
                     fontsize=10, color="#555555", zorder=3)

    for j, label in enumerate(col_labels):
        ax3.text(j, -0.7, label, ha="center", va="center", fontsize=9.5, fontweight="bold")
    for i, label in enumerate(row_labels):
        ax3.text(-0.75, i, label, ha="center", va="center", fontsize=9.5, fontweight="bold")

    ax3.set_title("Paired outcome comparison", fontsize=13, fontweight="bold", pad=12)
    ax3.axis("off")

    # Annotation explaining colors
    ax3.text(0.5, 2.0, "Green = Pro WW flipped outcome to Village\nRed = Pro WW flipped outcome to Wolf",
             ha="center", va="center", fontsize=8, color="#666666",
             style="italic", transform=ax3.transData)

    output_path = RESULTS_DIR / args.output
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
