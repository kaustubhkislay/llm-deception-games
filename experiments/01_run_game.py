#!/usr/bin/env python3
"""
Run a full game of One Night Ultimate Werewolf with web viewer.

Usage:
    python experiments/01_run_game.py [--seed SEED] [--rounds N] [--port PORT]
    
The web viewer will be available at http://localhost:9000
Open it in your browser to watch the game live.
"""

import argparse
import asyncio
import random
import threading
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.onuw import DEFAULT_PLAYER_NAMES, DEFAULT_ROLE_POOL, Role, GameConfig
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


def create_game(config: GameConfig, name: str | None = None) -> ONUWGame:
    """Create and register an ONUW game. Returns the game instance."""
    reset_broadcaster()
    game = ONUWGame(config=config, name=name)
    set_game(game)
    return game


async def run_game(game: ONUWGame) -> tuple[str, dict]:
    """Run the ONUW game. Returns (winner, usage_stats)."""
    winner = await game.run_game()
    
    usage_stats = {
        "usage": game.llm_client.usage_stats,
        "cache": game.llm_client.cache_stats,
    }
    
    return winner, usage_stats


def main():
    parser = argparse.ArgumentParser(description="Run an One Night Ultimate Werewolf game with web viewer")
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for deterministic game (default: random)"
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
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use for players (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=5,
        help="Number of players (default: 5)"
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep web server running after game ends (requires Ctrl+C to exit)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Display name for this game (shown in game list)"
    )
    args = parser.parse_args()
    
    # Generate seed if not provided
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    
    # Use default player names, trim to requested count
    player_names = DEFAULT_PLAYER_NAMES[:args.players]
    
    # Use default role pool - need exactly players + 3 roles
    num_roles_needed = args.players + 3
    role_pool = DEFAULT_ROLE_POOL[:num_roles_needed]
    
    # Create config
    config = GameConfig(
        seed=seed,
        models=[args.model] * args.players,
        roles=role_pool,
        names=player_names,
        num_rounds=args.rounds,
    )
    
    print("=" * 60)
    print("ONE NIGHT ULTIMATE WEREWOLF")
    print("=" * 60)
    if args.name:
        print(f"Game name: {args.name}")
    print(f"Seed: {seed}")
    print(f"Model: {args.model}")
    print(f"Players: {args.players}")
    print(f"Discussion rounds: {args.rounds}")
    print(f"Web viewer port: {args.port}")
    print(f"Roles: {[r.value for r in role_pool]}")
    print("=" * 60)
    
    # Connect event system to web queue
    set_web_queue(get_event_queue())
    
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()
    
    time.sleep(0.5)
    
    # Create the game
    game = create_game(config, args.name)
    
    print(f"\n🌐 Web viewer available at: http://localhost:{args.port}/game/{game.game_id}")
    print(f"   Open this URL in your browser to watch the game!\n")
    
    try:
        winner, usage_stats = asyncio.run(run_game(game))
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULT: {winner} WINS!")
        print(f"{'=' * 60}")
        
        # Print usage stats
        usage = usage_stats["usage"]
        cache = usage_stats["cache"]
        
        # Pricing estimates (per 1M tokens) - adjust for your model
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
        print(f"\n💰 ESTIMATED COST ({args.model})")
        print(f"   Input:  ${input_cost:.4f}")
        print(f"   Output: ${output_cost:.4f}")
        print(f"   Total:  ${total_cost:.4f}")
        
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
