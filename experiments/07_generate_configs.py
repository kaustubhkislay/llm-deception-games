#!/usr/bin/env python3
"""Generate config files for batch experiments."""

import json
import argparse
from pathlib import Path


# Default template config for vanilla ONUW with gpt-5
DEFAULT_TEMPLATE = {
    "models": [
        {"model": "gpt-5", "reasoning_effort": "medium"},
        {"model": "gpt-5", "reasoning_effort": "medium"},
        {"model": "gpt-5", "reasoning_effort": "medium"},
        {"model": "gpt-5", "reasoning_effort": "medium"},
        {"model": "gpt-5", "reasoning_effort": "medium"}
    ],
    "roles": [
        "WEREWOLF",
        "WEREWOLF",
        "MINION",
        "SEER",
        "ROBBER",
        "TROUBLEMAKER",
        "DRUNK",
        "VILLAGER",
    ],
    "names": [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Edward"
    ],
    "num_rounds": 5
}


def generate_configs(
    output_dir: Path,
    num_configs: int = 100,
    start_seed: int = 0,
    template: dict | None = None,
) -> list[Path]:
    """
    Generate config files with sequential seeds.
    
    Args:
        output_dir: Directory to write configs to
        num_configs: Number of config files to generate
        start_seed: Starting seed value
        template: Config template (seed will be overwritten)
    
    Returns:
        List of paths to generated config files
    """
    if template is None:
        template = DEFAULT_TEMPLATE.copy()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    for i in range(num_configs):
        seed = start_seed + i
        config = template.copy()
        config["seed"] = seed
        
        filename = f"config_{seed:03d}.json"
        filepath = output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        
        generated_files.append(filepath)
        print(f"Generated {filename}")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(description="Generate config files for batch experiments")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="settings/onuw_WwWwMiSeRbTrDrVi_gpt5med",
        help="Output directory for config files"
    )
    parser.add_argument(
        "--num-configs", "-n",
        type=int,
        default=100,
        help="Number of config files to generate"
    )
    parser.add_argument(
        "--start-seed", "-s",
        type=int,
        default=0,
        help="Starting seed value"
    )
    parser.add_argument(
        "--template", "-t",
        type=str,
        default=None,
        help="Path to template config file (optional)"
    )
    
    args = parser.parse_args()
    
    # Resolve output directory relative to project root
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    
    # Load template if provided
    template = None
    if args.template:
        with open(args.template) as f:
            template = json.load(f)
    
    print(f"\nGenerating {args.num_configs} configs in {output_dir}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.num_configs - 1}")
    print()
    
    files = generate_configs(
        output_dir=output_dir,
        num_configs=args.num_configs,
        start_seed=args.start_seed,
        template=template,
    )
    
    print(f"\nGenerated {len(files)} config files in {output_dir}")


if __name__ == "__main__":
    main()
