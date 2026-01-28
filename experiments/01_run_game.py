#!/usr/bin/env python3
"""
Run a full game of LLM Mafia with web viewer.

Usage:
    python experiments/01_run_game.py [--day-duration SECONDS] [--port PORT]
    
The web viewer will be available at http://localhost:8080
Open it in your browser to watch the game live.
"""

import argparse
import asyncio
import threading
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.mafia import DEFAULT_PLAYER_NAMES, DEFAULT_ROLE_DISTRIBUTION
from infra.game_engine import MafiaGame
from infra.events import reset_broadcaster, set_web_queue
from web.app import app, set_game, get_event_queue


def run_flask(port: int):
    """Run Flask in a separate thread."""
    # Disable Flask's default logging for cleaner output
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Flask error: {e}")


async def run_game(day_duration: int, model: str, turn_limit: int | None) -> str:
    """Run the mafia game."""
    # Reset event broadcaster for clean state
    reset_broadcaster()
    
    # Create game
    game = MafiaGame(
        player_names=DEFAULT_PLAYER_NAMES,
        role_distribution=DEFAULT_ROLE_DISTRIBUTION,
        model=model,
        day_duration_seconds=day_duration,
        turn_limit=turn_limit,
    )
    
    # Set up web app with game reference
    set_game(game)
    
    # Run the game
    winner = await game.run_game()
    return winner


def main():
    parser = argparse.ArgumentParser(description="Run an LLM Mafia game with web viewer")
    parser.add_argument(
        "--day-duration", 
        type=int, 
        default=300,
        help="Duration of each day phase in seconds (default: 300 = 5 minutes)"
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
        default="gpt-5-nano",
        help="OpenAI model to use for players (default: gpt-5-nano)"
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=None,
        help="Limit number of turns (night+day+vote = 1 turn). Default: no limit"
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep web server running after game ends (requires Ctrl+C to exit)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM MAFIA")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Day duration: {args.day_duration} seconds")
    print(f"Turn limit: {args.turn_limit or 'none'}")
    print(f"Web viewer port: {args.port}")
    print("=" * 60)
    
    # Connect event system to web queue
    set_web_queue(get_event_queue())
    
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()
    
    # Give Flask a moment to start
    time.sleep(0.5)
    
    print(f"\n🌐 Web viewer available at: http://localhost:{args.port}")
    print(f"   Open this URL in your browser to watch the game!\n")
    
    # Run the game
    try:
        winner = asyncio.run(run_game(args.day_duration, args.model, args.turn_limit))
        print(f"\n{'=' * 60}")
        print(f"FINAL RESULT: {winner} WINS!")
        print(f"{'=' * 60}")
        
        if args.keep_alive:
            # Keep the server running so users can review the game
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
