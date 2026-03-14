#!/usr/bin/env python3
"""Plot win rates by starting and ending role as a vertically stacked figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
ANALYSIS_FILE = RESULTS_DIR / "09_analysis_onuw_WwWwMiSeRbTrDrVi_gpt5minimed_50g_20260314_124449.json"

ROLE_ORDER = ["WEREWOLF", "MINION", "SEER", "ROBBER", "TROUBLEMAKER", "DRUNK", "VILLAGER"]
EVIL_ROLES = {"WEREWOLF", "MINION"}

with open(ANALYSIS_FILE) as f:
    data = json.load(f)

n_games = data["completed_games"]
by_start = data["by_starting_role"]
by_end = data["by_ending_role"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

COLOR_VILLAGE = "#5B9BD5"
COLOR_EVIL = "#C55A5A"

def plot_role_bars(ax, role_data, title):
    roles = [r for r in ROLE_ORDER if r in role_data]
    win_rates = [role_data[r]["win_rate"] * 100 for r in roles]
    ci_lows = [role_data[r]["ci_low"] * 100 for r in roles]
    ci_highs = [role_data[r]["ci_high"] * 100 for r in roles]
    totals = [role_data[r]["total"] for r in roles]
    errors_lo = [wr - ci_l for wr, ci_l in zip(win_rates, ci_lows)]
    errors_hi = [ci_h - wr for wr, ci_h in zip(win_rates, ci_highs)]

    colors = [COLOR_EVIL if r in EVIL_ROLES else COLOR_VILLAGE for r in roles]
    x = np.arange(len(roles))

    ax.bar(x, win_rates, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, win_rates, yerr=[errors_lo, errors_hi], fmt="none", color="black",
                capsize=4, capthick=1.2, linewidth=1.2)

    for i, (role, total) in enumerate(zip(roles, totals)):
        ax.annotate(f"n={total}", (i, 3), ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(roles, fontsize=10)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, len(roles) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plot_role_bars(ax1, by_start, "Win rate by initial role (start of night)")
plot_role_bars(ax2, by_end, "Win rate by final role (end of night)")

plt.tight_layout()

output_path = RESULTS_DIR / "11_role_winrates.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
