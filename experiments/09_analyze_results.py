#!/usr/bin/env python3
"""Analyze batch game results with win rates by role and swap count."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def wilson_score_interval(wins: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    More accurate than normal approximation, especially for small samples or extreme proportions.
    """
    if total == 0:
        return (0.0, 0.0)
    
    from scipy import stats
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = wins / total
    
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def load_results(results_file: Path) -> dict:
    """Load results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def analyze_by_starting_role(games: list[dict]) -> dict[str, dict]:
    """
    Calculate win rate by starting role.
    
    Returns dict: role -> {wins, total, win_rate, ci_low, ci_high}
    """
    role_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
    
    for game in games:
        if game["winner"] == "ERROR":
            continue
        
        for player in game["players"]:
            role = player["starting_role"]
            role_stats[role]["total"] += 1
            if player["won"]:
                role_stats[role]["wins"] += 1
    
    # Calculate win rates and confidence intervals
    results = {}
    for role, stats in role_stats.items():
        wins = stats["wins"]
        total = stats["total"]
        win_rate = wins / total if total > 0 else 0
        ci_low, ci_high = wilson_score_interval(wins, total)
        
        results[role] = {
            "wins": wins,
            "total": total,
            "win_rate": win_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    
    return results


def analyze_by_ending_role(games: list[dict]) -> dict[str, dict]:
    """
    Calculate win rate by ending role.
    
    Returns dict: role -> {wins, total, win_rate, ci_low, ci_high}
    """
    role_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
    
    for game in games:
        if game["winner"] == "ERROR":
            continue
        
        for player in game["players"]:
            role = player["ending_role"]
            role_stats[role]["total"] += 1
            if player["won"]:
                role_stats[role]["wins"] += 1
    
    # Calculate win rates and confidence intervals
    results = {}
    for role, stats in role_stats.items():
        wins = stats["wins"]
        total = stats["total"]
        win_rate = wins / total if total > 0 else 0
        ci_low, ci_high = wilson_score_interval(wins, total)
        
        results[role] = {
            "wins": wins,
            "total": total,
            "win_rate": win_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    
    return results


def analyze_by_swap_count(games: list[dict]) -> dict[int, dict]:
    """
    Calculate win rate by number of times a player was swapped.
    
    Returns dict: swap_count -> {wins, total, win_rate, ci_low, ci_high}
    """
    swap_stats: dict[int, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
    
    for game in games:
        if game["winner"] == "ERROR":
            continue
        
        for player in game["players"]:
            swap_count = player["swap_count"]
            swap_stats[swap_count]["total"] += 1
            if player["won"]:
                swap_stats[swap_count]["wins"] += 1
    
    # Calculate win rates and confidence intervals
    results = {}
    for swap_count, stats in sorted(swap_stats.items()):
        wins = stats["wins"]
        total = stats["total"]
        win_rate = wins / total if total > 0 else 0
        ci_low, ci_high = wilson_score_interval(wins, total)
        
        results[swap_count] = {
            "wins": wins,
            "total": total,
            "win_rate": win_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    
    return results


def print_analysis(
    by_starting_role: dict,
    by_ending_role: dict,
    by_swap_count: dict,
):
    """Print formatted analysis results."""
    print("\n" + "="*70)
    print("WIN RATE BY STARTING ROLE")
    print("="*70)
    print(f"{'Role':<15} {'Wins':>6} {'Total':>6} {'Win Rate':>10} {'95% CI':>18}")
    print("-"*70)
    
    for role in sorted(by_starting_role.keys()):
        stats = by_starting_role[role]
        print(f"{role:<15} {stats['wins']:>6} {stats['total']:>6} "
              f"{stats['win_rate']*100:>9.1f}% "
              f"[{stats['ci_low']*100:>5.1f}%, {stats['ci_high']*100:>5.1f}%]")
    
    print("\n" + "="*70)
    print("WIN RATE BY ENDING ROLE")
    print("="*70)
    print(f"{'Role':<15} {'Wins':>6} {'Total':>6} {'Win Rate':>10} {'95% CI':>18}")
    print("-"*70)
    
    for role in sorted(by_ending_role.keys()):
        stats = by_ending_role[role]
        print(f"{role:<15} {stats['wins']:>6} {stats['total']:>6} "
              f"{stats['win_rate']*100:>9.1f}% "
              f"[{stats['ci_low']*100:>5.1f}%, {stats['ci_high']*100:>5.1f}%]")
    
    print("\n" + "="*70)
    print("WIN RATE BY SWAP COUNT")
    print("="*70)
    print(f"{'Swaps':>6} {'Wins':>6} {'Total':>6} {'Win Rate':>10} {'95% CI':>18}")
    print("-"*70)
    
    for swap_count, stats in sorted(by_swap_count.items()):
        print(f"{swap_count:>6} {stats['wins']:>6} {stats['total']:>6} "
              f"{stats['win_rate']*100:>9.1f}% "
              f"[{stats['ci_low']*100:>5.1f}%, {stats['ci_high']*100:>5.1f}%]")


def plot_analysis(
    by_starting_role: dict,
    by_ending_role: dict,
    by_swap_count: dict,
    output_file: Optional[Path] = None,
):
    """Generate plots for analysis results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Win rate by starting role
    ax = axes[0]
    roles = sorted(by_starting_role.keys())
    win_rates = [by_starting_role[r]["win_rate"] * 100 for r in roles]
    ci_lows = [by_starting_role[r]["ci_low"] * 100 for r in roles]
    ci_highs = [by_starting_role[r]["ci_high"] * 100 for r in roles]
    errors = [[wr - ci_l for wr, ci_l in zip(win_rates, ci_lows)],
              [ci_h - wr for wr, ci_h in zip(win_rates, ci_highs)]]
    
    bars = ax.bar(range(len(roles)), win_rates, color='steelblue', alpha=0.7)
    ax.errorbar(range(len(roles)), win_rates, yerr=errors, fmt='none', color='black', capsize=3)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=45, ha='right')
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate by Starting Role")
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.set_ylim(0, 100)
    
    # Add sample size annotations
    for i, role in enumerate(roles):
        n = by_starting_role[role]["total"]
        ax.annotate(f'n={n}', (i, 5), ha='center', fontsize=8, color='gray')
    
    # Plot 2: Win rate by ending role
    ax = axes[1]
    roles = sorted(by_ending_role.keys())
    win_rates = [by_ending_role[r]["win_rate"] * 100 for r in roles]
    ci_lows = [by_ending_role[r]["ci_low"] * 100 for r in roles]
    ci_highs = [by_ending_role[r]["ci_high"] * 100 for r in roles]
    errors = [[wr - ci_l for wr, ci_l in zip(win_rates, ci_lows)],
              [ci_h - wr for wr, ci_h in zip(win_rates, ci_highs)]]
    
    bars = ax.bar(range(len(roles)), win_rates, color='forestgreen', alpha=0.7)
    ax.errorbar(range(len(roles)), win_rates, yerr=errors, fmt='none', color='black', capsize=3)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=45, ha='right')
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate by Ending Role")
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.set_ylim(0, 100)
    
    # Add sample size annotations
    for i, role in enumerate(roles):
        n = by_ending_role[role]["total"]
        ax.annotate(f'n={n}', (i, 5), ha='center', fontsize=8, color='gray')
    
    # Plot 3: Win rate by swap count
    ax = axes[2]
    swap_counts = sorted(by_swap_count.keys())
    win_rates = [by_swap_count[s]["win_rate"] * 100 for s in swap_counts]
    ci_lows = [by_swap_count[s]["ci_low"] * 100 for s in swap_counts]
    ci_highs = [by_swap_count[s]["ci_high"] * 100 for s in swap_counts]
    errors = [[wr - ci_l for wr, ci_l in zip(win_rates, ci_lows)],
              [ci_h - wr for wr, ci_h in zip(win_rates, ci_highs)]]
    
    bars = ax.bar(range(len(swap_counts)), win_rates, color='coral', alpha=0.7)
    ax.errorbar(range(len(swap_counts)), win_rates, yerr=errors, fmt='none', color='black', capsize=3)
    ax.set_xticks(range(len(swap_counts)))
    ax.set_xticklabels([str(s) for s in swap_counts])
    ax.set_xlabel("Number of Swaps")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate by Swap Count")
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.set_ylim(0, 100)
    
    # Add sample size annotations
    for i, s in enumerate(swap_counts):
        n = by_swap_count[s]["total"]
        ax.annotate(f'n={n}', (i, 5), ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze batch game results")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSON file (from 08_batch_from_configs.py)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate plots"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for plot (if not specified, displays interactively)"
    )
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        # Try relative to results dir
        results_file = Path(__file__).parent.parent / "results" / args.results_file
    
    if not results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    print(f"Loading results from: {results_file}")
    data = load_results(results_file)
    
    games = data.get("games", [])
    completed_games = [g for g in games if g["winner"] != "ERROR"]
    
    print(f"\nTotal games: {len(games)}")
    print(f"Completed games: {len(completed_games)}")
    print(f"Errors: {len(games) - len(completed_games)}")
    
    if data.get("rate_limit_errors", 0) > 0:
        print(f"  - Rate limit errors: {data['rate_limit_errors']}")
    if data.get("other_errors", 0) > 0:
        print(f"  - Other errors: {data['other_errors']}")
    
    if not completed_games:
        print("\nNo completed games to analyze!")
        sys.exit(1)
    
    # Run analysis
    by_starting_role = analyze_by_starting_role(games)
    by_ending_role = analyze_by_ending_role(games)
    by_swap_count = analyze_by_swap_count(games)
    
    # Print results
    print_analysis(by_starting_role, by_ending_role, by_swap_count)
    
    # Save analysis to JSON
    analysis_output = {
        "source_file": str(results_file),
        "total_games": len(games),
        "completed_games": len(completed_games),
        "by_starting_role": by_starting_role,
        "by_ending_role": by_ending_role,
        "by_swap_count": {str(k): v for k, v in by_swap_count.items()},
    }
    
    analysis_file = results_file.parent / f"09_analysis_{results_file.stem.replace('08_batch_', '')}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_file}")
    
    # Generate plots if requested
    if args.plot:
        output_file = Path(args.output) if args.output else None
        if output_file and not output_file.is_absolute():
            output_file = Path(__file__).parent.parent / "results" / output_file
        
        plot_analysis(by_starting_role, by_ending_role, by_swap_count, output_file)


if __name__ == "__main__":
    main()
