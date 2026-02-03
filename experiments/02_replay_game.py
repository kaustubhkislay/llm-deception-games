#!/usr/bin/env python3
"""
Replay a completed ONUW game from its JSON log file.

Usage:
    python experiments/02_replay_game.py <log_file.jsonl> [--port PORT] [--dev]
    
Opens a web viewer showing the game replay.
Use --dev to auto-shutdown after 60 seconds (useful for testing).

Note: This script uses the unified web viewer at web/app.py which supports
both live games and replays. You can also view any game directly at:
    http://localhost:PORT/game/<game_id>
"""

import argparse
import sys
import threading
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.game_engine import load_game_from_log
from web.app import app

# Dev mode auto-shutdown
_shutdown_timer: threading.Timer | None = None


def shutdown_server():
    """Shutdown the Flask server."""
    print("\n⏰ Dev mode: Auto-shutdown triggered")
    import os
    os._exit(0)


def main():
    global _shutdown_timer
    
    parser = argparse.ArgumentParser(description="Replay a completed game from its log file")
    parser.add_argument("log_file", type=str, help="Path to the game log file (.jsonl) or just the game_id")
    parser.add_argument("--port", type=int, default=9000, help="Port for web viewer (default: 9000)")
    parser.add_argument("--dev", action="store_true", help="Dev mode: auto-shutdown after 60 seconds")
    parser.add_argument("--dev-timeout", type=int, default=60, help="Dev mode timeout in seconds (default: 60)")
    args = parser.parse_args()
    
    # Check if file exists - support both file path and game_id
    log_path = Path(args.log_file)
    game_logs_dir = Path(__file__).parent.parent / "game_logs"
    
    if not log_path.exists():
        # Try looking in game_logs directory
        log_path = game_logs_dir / args.log_file
        if not log_path.exists():
            # Try with game_ prefix
            log_path = game_logs_dir / f"game_{args.log_file}.jsonl"
            if not log_path.exists():
                print(f"Error: Log file not found: {args.log_file}")
                print(f"Looked in:")
                print(f"  - {args.log_file}")
                print(f"  - {game_logs_dir / args.log_file}")
                print(f"  - {game_logs_dir / f'game_{args.log_file}.jsonl'}")
                sys.exit(1)
    
    print(f"Loading game from: {log_path}")
    game_state = load_game_from_log(str(log_path))
    
    # Extract game_id from filename or loaded state
    game_id = game_state.get("game_id")
    if not game_id:
        # Extract from filename: game_YYYYMMDD_HHMMSS.jsonl -> YYYYMMDD_HHMMSS
        game_id = log_path.stem.replace("game_", "")
    
    print(f"\n{'='*60}")
    print(f"GAME REPLAY - ONE NIGHT ULTIMATE WEREWOLF")
    print(f"{'='*60}")
    print(f"Game ID: {game_id}")
    if game_state.get("name"):
        print(f"Name: {game_state.get('name')}")
    print(f"Phase: {game_state.get('phase', 'UNKNOWN')}")
    print(f"Winner: {game_state.get('winner', 'N/A')}")
    print(f"\nPlayers:")
    for p in game_state.get('players', []):
        original = p.get('original_role', p.get('role', '?'))
        current = p.get('current_role', p.get('role', '?'))
        changed = ' (CHANGED)' if original != current else ''
        print(f"  {p['name']}: {current}{changed}")
    print(f"\nCenter cards: {game_state.get('center_cards', [])}")
    print(f"Messages: {len(game_state.get('messages', []))}")
    print(f"{'='*60}")
    
    print(f"\n🌐 Web viewer available at: http://localhost:{args.port}/game/{game_id}")
    print(f"   Games list at: http://localhost:{args.port}/")
    
    if args.dev:
        print(f"⚠️  Dev mode: Server will auto-shutdown in {args.dev_timeout} seconds")
        _shutdown_timer = threading.Timer(args.dev_timeout, shutdown_server)
        _shutdown_timer.daemon = True
        _shutdown_timer.start()
    else:
        print(f"   Press Ctrl+C to exit.")
    
    print()
    
    # Suppress Flask startup messages
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if _shutdown_timer:
            _shutdown_timer.cancel()
        sys.exit(0)


if __name__ == "__main__":
    main()
