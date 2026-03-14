#!/usr/bin/env python3
"""
Compare two batch results files and highlight games where the outcome differs.

For each matching seed, prints the outcome in both batches and flags differences.
Outputs viewer links for easy inspection of divergent games.
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def load_games_by_seed(results_file: Path) -> dict[int, dict]:
    with open(results_file) as f:
        data = json.load(f)
    return {g["seed"]: g for g in data["games"]}


def main():
    parser = argparse.ArgumentParser(description="Compare two batch results by seed")
    parser.add_argument("baseline", type=str, help="Path to baseline results JSON")
    parser.add_argument("treatment", type=str, help="Path to treatment results JSON")
    parser.add_argument("--port", type=int, default=8090, help="Viewer port (default 8090)")
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--treatment-label", type=str, default="treatment")

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    treatment_path = Path(args.treatment)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    if not treatment_path.is_absolute():
        treatment_path = PROJECT_ROOT / treatment_path

    baseline = load_games_by_seed(baseline_path)
    treatment = load_games_by_seed(treatment_path)

    shared_seeds = sorted(set(baseline.keys()) & set(treatment.keys()))
    assert shared_seeds, "No shared seeds between the two result files!"

    viewer = f"http://127.0.0.1:{args.port}"

    same_count = 0
    diff_count = 0
    diff_games = []

    print(f"\n{'='*80}")
    print(f"BATCH COMPARISON: {args.baseline_label} vs {args.treatment_label}")
    print(f"{'='*80}")
    print(f"  {args.baseline_label}: {baseline_path.name}")
    print(f"  {args.treatment_label}: {treatment_path.name}")
    print(f"  Shared seeds: {len(shared_seeds)}")
    print()

    header = f"{'Seed':>6}  {'':>12} {'':>12}  {'Match':>7}"
    print(f"{' ':>6}  {args.baseline_label:>12} {args.treatment_label:>12}  {'':>7}")
    print("-" * 60)

    for seed in shared_seeds:
        b = baseline[seed]
        t = treatment[seed]
        bw = b["winner"]
        tw = t["winner"]
        match = bw == tw

        if match:
            same_count += 1
            marker = ""
        else:
            diff_count += 1
            marker = " <-- DIFFERENT"
            diff_games.append((seed, b, t))

        print(f"  {seed:>4}  {bw:>12} {tw:>12}{marker}")

    print("-" * 60)
    print(f"  Same outcome:      {same_count}/{len(shared_seeds)}")
    print(f"  Different outcome:  {diff_count}/{len(shared_seeds)}")

    if diff_games:
        print(f"\n{'='*80}")
        print("GAMES WITH DIFFERENT OUTCOMES")
        print(f"{'='*80}\n")

        for seed, b, t in diff_games:
            b_id = b["game_id"]
            t_id = t["game_id"]
            print(f"--- Seed {seed}: {b['winner']} -> {t['winner']} ---")
            print(f"  {args.baseline_label}: {viewer}/game/{b_id}")
            print(f"  {args.treatment_label}: {viewer}/game/{t_id}")

            b_players = {p["name"]: p for p in b["players"]}
            t_players = {p["name"]: p for p in t["players"]}
            all_names = sorted(set(b_players.keys()) | set(t_players.keys()))

            print(f"\n  {'Player':<10} {'Start Role':<15} {'End Role (B)':<15} {'End Role (T)':<15} {'Won (B)':>8} {'Won (T)':>8}")
            for name in all_names:
                bp = b_players.get(name, {})
                tp = t_players.get(name, {})
                start = bp.get("starting_role", "?")
                b_end = bp.get("ending_role", "?")
                t_end = tp.get("ending_role", "?")
                b_won = "Y" if bp.get("won") else "N"
                t_won = "Y" if tp.get("won") else "N"
                print(f"  {name:<10} {start:<15} {b_end:<15} {t_end:<15} {b_won:>8} {t_won:>8}")

            print(f"\n  Killed ({args.baseline_label}):  {b['killed_players']}")
            print(f"  Killed ({args.treatment_label}): {t['killed_players']}")
            print()
    else:
        print("\nAll games had the same outcome!")

    # Summary stats
    b_ww_wins = sum(1 for s in shared_seeds if baseline[s]["winner"] == "WEREWOLF")
    t_ww_wins = sum(1 for s in shared_seeds if treatment[s]["winner"] == "WEREWOLF")
    n = len(shared_seeds)
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Werewolf win rate ({args.baseline_label}):  {b_ww_wins}/{n} ({b_ww_wins/n*100:.1f}%)")
    print(f"  Werewolf win rate ({args.treatment_label}): {t_ww_wins}/{n} ({t_ww_wins/n*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
