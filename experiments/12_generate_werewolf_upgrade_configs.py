#!/usr/bin/env python3
"""
Generate configs where werewolf players use a different (upgraded) model.

Replicates the SeededGameSetup role-assignment logic to determine which
player indices receive the werewolf role for each seed, then assigns
the upgraded model to those players while keeping everyone else on the
baseline model.
"""

import json
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

ROLES = ["WEREWOLF", "WEREWOLF", "MINION", "SEER", "ROBBER", "TROUBLEMAKER", "DRUNK", "VILLAGER"]
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Edward"]
SORTED_NAMES = sorted(NAMES)
NUM_ROUNDS = 5


def get_werewolf_players(seed: int) -> list[str]:
    """Replicate SeededGameSetup._setup_game to find which players are werewolves."""
    rng = random.Random(seed)
    all_roles = list(ROLES)
    rng.shuffle(all_roles)

    player_roles = all_roles[:len(SORTED_NAMES)]
    return [
        name for name, role in zip(SORTED_NAMES, player_roles)
        if role == "WEREWOLF"
    ]


def generate_configs(
    output_dir: Path,
    baseline_model: str,
    upgraded_model: str,
    reasoning_effort: str,
    num_configs: int,
    start_seed: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for i in range(num_configs):
        seed = start_seed + i
        ww_players = set(get_werewolf_players(seed))

        models = []
        for name in NAMES:
            model_name = upgraded_model if name in ww_players else baseline_model
            models.append({"model": model_name, "reasoning_effort": reasoning_effort})

        config = {
            "models": models,
            "roles": ROLES,
            "names": NAMES,
            "num_rounds": NUM_ROUNDS,
            "seed": seed,
        }

        filepath = output_dir / f"config_{seed:03d}.json"
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        ww_names = sorted(ww_players)
        print(f"  seed={seed:3d}  werewolves={ww_names}  -> {upgraded_model}")
        generated.append(filepath)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate configs with upgraded werewolf models")
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    parser.add_argument("--baseline-model", type=str, default="gpt-5-mini")
    parser.add_argument("--upgraded-model", type=str, default="gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--num-configs", "-n", type=int, default=50)
    parser.add_argument("--start-seed", "-s", type=int, default=0)

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    print(f"\nGenerating {args.num_configs} configs in {output_dir}")
    print(f"  Baseline: {args.baseline_model}, Upgraded WW: {args.upgraded_model}")
    print(f"  Reasoning: {args.reasoning_effort}, Seeds: {args.start_seed}-{args.start_seed + args.num_configs - 1}\n")

    files = generate_configs(
        output_dir=output_dir,
        baseline_model=args.baseline_model,
        upgraded_model=args.upgraded_model,
        reasoning_effort=args.reasoning_effort,
        num_configs=args.num_configs,
        start_seed=args.start_seed,
    )
    print(f"\nGenerated {len(files)} config files")


if __name__ == "__main__":
    main()
