#!/usr/bin/env python3
"""
Start the web viewer to browse all games in game_logs/.

Usage:
    python experiments/05_viewer.py [--port PORT]
    
Navigate to http://localhost:PORT/ to see the games list.
Click on any game to view its replay.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web.app import app, GAME_LOGS_DIR


def main():
    parser = argparse.ArgumentParser(description="Start the game viewer")
    parser.add_argument("--port", "-p", type=int, default=9000, help="Port for web viewer (default: 9000)")
    args = parser.parse_args()
    
    # Count available games
    if GAME_LOGS_DIR.exists():
        game_count = len(list(GAME_LOGS_DIR.glob("game_*.jsonl")))
    else:
        game_count = 0
    
    print(f"{'='*60}")
    print("ONUW GAME VIEWER")
    print(f"{'='*60}")
    print(f"Games directory: {GAME_LOGS_DIR}")
    print(f"Games available: {game_count}")
    print(f"{'='*60}")
    print(f"\n🌐 Web viewer at: http://localhost:{args.port}/")
    print("   Press Ctrl+C to exit.\n")
    
    # Suppress Flask startup messages
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
