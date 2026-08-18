#!/usr/bin/env python3
"""Compare our replication against Lucassen's published results.

Lucassen's values are read off the charts in
https://jlucassen.com/hidden-role-games-as-a-trusted-model-eval/ (2026-03 runs).
Both sides in werewolf-instance units (starting-wolf players, personal outcomes),
his reporting convention. Our values computed from batch results.
"""
import json
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"

# (label, lucassen_pct, lucassen_n, our_batch_tag)
CONDITIONS = [
    ("mini\nbaseline", 45.1, 133, "u1_b1"),
    ("flash\nbaseline", 41.4, 133, "x1_b2"),
    ("3.1-pro WW\nvs flash", 40.6, 133, "u5_c2"),
    ("opus-4.6 WW\nvs flash", 41.4, 133, "u6_c3"),
    ("flash WW\nvs mini*", 67.7, 133, "u2_c1"),
]

COLOR_LUC = "#4E8FC7"
COLOR_OURS = "#D97E1F"


def wilson(wins, total):
    z = scipy_stats.norm.ppf(0.975)
    p = wins / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return center - margin, center + margin


def our_instance_rate(tag):
    f = sorted(glob.glob(str(RESULTS_DIR / f"08_batch_{tag}*.json")))[-1]
    d = json.load(open(f))
    wins = total = 0
    for g in d["games"]:
        if g["winner"] == "ERROR":
            continue
        for p in g["players"]:
            if p["starting_role"] == "WEREWOLF":
                total += 1
                wins += p["won"]
    return wins, total


fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(CONDITIONS))
w = 0.38

for i, (label, luc_pct, luc_n, tag) in enumerate(CONDITIONS):
    luc_wins = round(luc_pct / 100 * luc_n)
    llo, lhi = wilson(luc_wins, luc_n)
    ax.bar(i - w / 2, luc_pct, w, color=COLOR_LUC, edgecolor="white", linewidth=0.8, zorder=2)
    ax.errorbar(i - w / 2, luc_pct, yerr=[[luc_pct - llo * 100], [lhi * 100 - luc_pct]],
                fmt="none", color="#333333", capsize=4, linewidth=1.1, zorder=3)
    ax.annotate(f"{luc_pct:.0f}", (i - w / 2, lhi * 100 + 1.5), ha="center", fontsize=10,
                fontweight="bold", color="#2F5E8C")

    wins, total = our_instance_rate(tag)
    pct = 100 * wins / total
    olo, ohi = wilson(wins, total)
    ax.bar(i + w / 2, pct, w, color=COLOR_OURS, edgecolor="white", linewidth=0.8, zorder=2)
    ax.errorbar(i + w / 2, pct, yerr=[[pct - olo * 100], [ohi * 100 - pct]],
                fmt="none", color="#333333", capsize=4, linewidth=1.1, zorder=3)
    ax.annotate(f"{pct:.0f}", (i + w / 2, ohi * 100 + 1.5), ha="center", fontsize=10,
                fontweight="bold", color="#96530F")

ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CONDITIONS], fontsize=10)
ax.set_ylabel("Werewolf-instance win rate (%)")
ax.set_ylim(0, 100)
ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR_LUC),
                   plt.Rectangle((0, 0), 1, 1, color=COLOR_OURS)],
          labels=["Lucassen (2026-03)", "ours (2026-08)"], frameon=False, loc="upper left")
ax.set_title("Replication vs original: the five shared conditions\n"
             "(werewolf-instance units, his convention; * = his one significant result)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = RESULTS_DIR / "compare_original.png"
plt.savefig(out, dpi=150, facecolor="white")
print(f"Saved {out}")
