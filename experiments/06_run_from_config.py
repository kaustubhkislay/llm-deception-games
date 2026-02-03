#!/usr/bin/env python3
"""
Run a deterministic game from a GameConfig.

Usage:
    # Run from a config file:
    python experiments/06_run_from_config.py --config path/to/config.json
    
    # Run with inline config:
    python experiments/06_run_from_config.py --seed 12345 --models gpt-5-mini gpt-5-mini gpt-5-mini gpt-5-mini gpt-5-mini
    
    # Run same config twice to verify cache reproducibility:
    python experiments/06_run_from_config.py --seed 12345 --verify-cache

The game will be deterministic based on the seed:
- Role assignments are determined by the seed
- Night action targets are determined by the seed
- LLM responses are cached, so re-running with the same config uses cache
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.onuw import GameConfig, Role, DEFAULT_PLAYER_NAMES, DEFAULT_ROLE_POOL
from infra.game_engine import ONUWGame
from infra.events import reset_broadcaster, set_web_queue
from web.app import app, set_game, get_event_queue


def run_flask(port: int):
    """Run Flask in a separate thread."""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Flask error: {e}")


def load_config(config_path: str) -> GameConfig:
    """Load a GameConfig from a JSON file."""
    with open(config_path, 'r') as f:
        data = json.load(f)
    return GameConfig.from_dict(data)


def save_config(config: GameConfig, config_path: str) -> None:
    """Save a GameConfig to a JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def create_config(
    seed: int,
    models: list[str],
    roles: list[Role] | None = None,
    names: list[str] | None = None,
    num_rounds: int = 5,
) -> GameConfig:
    """Create a GameConfig with the given parameters."""
    num_players = len(models)
    
    if roles is None:
        # Use default role pool, selecting exactly num_players + 3
        roles = DEFAULT_ROLE_POOL[:num_players + 3]
    
    if names is None:
        names = DEFAULT_PLAYER_NAMES[:num_players]
    
    return GameConfig(
        seed=seed,
        models=models,
        roles=roles,
        names=names,
        num_rounds=num_rounds,
    )


async def run_game_from_config(config: GameConfig, name: str | None = None, port: int = 9000) -> tuple[str, dict]:
    """Run a game from a GameConfig and return (winner, stats)."""
    reset_broadcaster()
    
    game = ONUWGame(config=config, name=name)
    set_game(game)
    
    print(f"\n🌐 Web viewer available at: http://localhost:{port}/game/{game.game_id}")
    print(f"   Open this URL in your browser to watch the game!\n")
    
    winner = await game.run_game()
    
    stats = {
        "game_id": game.game_id,
        "winner": winner,
        "usage": game.llm_client.usage_stats,
        "cache": game.llm_client.cache_stats,
    }
    
    return winner, stats


def print_stats(stats: dict, model: str = "gpt-5-mini"):
    """Print game statistics."""
    usage = stats["usage"]
    cache = stats["cache"]
    
    # Pricing estimates (per 1M tokens)
    INPUT_PRICE = 0.15
    OUTPUT_PRICE = 0.60
    
    input_cost = (usage['prompt_tokens'] / 1_000_000) * INPUT_PRICE
    output_cost = (usage['completion_tokens'] / 1_000_000) * OUTPUT_PRICE
    total_cost = input_cost + output_cost
    
    print(f"\n📊 API USAGE STATISTICS")
    print(f"   API calls: {usage['api_calls']}")
    print(f"   Prompt tokens: {usage['prompt_tokens']:,}")
    print(f"   Completion tokens: {usage['completion_tokens']:,}")
    print(f"   Total tokens: {usage['total_tokens']:,}")
    print(f"\n   Cache hits: {int(cache['hits'])} ({cache['hit_rate']*100:.1f}% hit rate)")
    print(f"   Cache misses: {int(cache['misses'])}")
    print(f"\n💰 ESTIMATED COST ({model})")
    print(f"   Input:  ${input_cost:.4f}")
    print(f"   Output: ${output_cost:.4f}")
    print(f"   Total:  ${total_cost:.4f}")


async def verify_cache_reproducibility(config: GameConfig, port: int = 9000) -> bool:
    """Run the same config twice and verify results match."""
    print("\n" + "=" * 60)
    print("CACHE REPRODUCIBILITY TEST")
    print("=" * 60)
    print("Running the same config twice to verify deterministic results...\n")
    
    # First run
    print("--- RUN 1 ---")
    winner1, stats1 = await run_game_from_config(config, name="Reproducibility Test - Run 1", port=port)
    print(f"Run 1 complete: {winner1} wins")
    print_stats(stats1)
    
    # Reset for second run
    print("\n--- RUN 2 ---")
    winner2, stats2 = await run_game_from_config(config, name="Reproducibility Test - Run 2", port=port)
    print(f"Run 2 complete: {winner2} wins")
    print_stats(stats2)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    cache_hit_rate = stats2["cache"]["hit_rate"]
    same_winner = winner1 == winner2
    
    print(f"Run 1 winner: {winner1}")
    print(f"Run 2 winner: {winner2}")
    print(f"Winners match: {'✅ YES' if same_winner else '❌ NO'}")
    print(f"Run 2 cache hit rate: {cache_hit_rate*100:.1f}%")
    
    if cache_hit_rate == 1.0 and same_winner:
        print("\n✅ SUCCESS: Game is fully reproducible from cache!")
        return True
    elif same_winner:
        print(f"\n⚠️ PARTIAL: Same winner, but cache hit rate was {cache_hit_rate*100:.1f}%")
        print("   Some LLM calls were not cached (this is expected on first run).")
        return True
    else:
        print("\n❌ FAILURE: Different winners - game is not reproducible!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic ONUW game from config")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=12345,
        help="Random seed for game setup (default: 12345)"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=["gpt-5-mini"] * 5,
        help="Models for each player (default: 5x gpt-5-mini)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=5,
        help="Number of discussion rounds (default: 5)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port for web viewer (default: 9000)"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save the config to a file (for reuse)"
    )
    parser.add_argument(
        "--verify-cache",
        action="store_true",
        help="Run twice and verify cache reproducibility"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Display name for this game"
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep web server running after game ends"
    )
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = create_config(
            seed=args.seed,
            models=args.models,
            num_rounds=args.rounds,
        )
    
    # Optionally save config
    if args.save_config:
        save_config(config, args.save_config)
        print(f"Saved config to {args.save_config}")
    
    # Print config
    print("=" * 60)
    print("GAME CONFIGURATION")
    print("=" * 60)
    print(f"Seed: {config.seed}")
    print(f"Players: {config.names}")
    print(f"Models: {config.models}")
    print(f"Roles: {[r.value for r in config.roles]}")
    print(f"Rounds: {config.num_rounds}")
    print("=" * 60)
    
    # Connect event system to web queue
    set_web_queue(get_event_queue())
    
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()
    time.sleep(0.5)
    
    try:
        if args.verify_cache:
            success = asyncio.run(verify_cache_reproducibility(config, port=args.port))
            sys.exit(0 if success else 1)
        else:
            winner, stats = asyncio.run(run_game_from_config(config, name=args.name, port=args.port))
            
            print(f"\n{'=' * 60}")
            print(f"FINAL RESULT: {winner} WINS!")
            print(f"{'=' * 60}")
            
            print_stats(stats, args.models[0] if args.models else "gpt-5-mini")
            
            if args.keep_alive:
                print("\nGame complete. Web viewer will remain active.")
                print("Press Ctrl+C to exit.")
                while True:
                    time.sleep(1)
            else:
                print("\nGame complete. Exiting.")
                sys.exit(0)
                
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
