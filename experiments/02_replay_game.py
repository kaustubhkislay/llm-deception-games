#!/usr/bin/env python3
"""
Replay a completed game from its JSON log file.

Usage:
    python experiments/02_replay_game.py <log_file.jsonl> [--port PORT] [--dev]
    
Opens a web viewer showing the final state of the game.
Use --dev to auto-shutdown after 60 seconds (useful for testing).
"""

import argparse
import sys
import threading
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, Response
import json

from infra.game_engine import load_game_from_log

app = Flask(__name__, template_folder="../web/templates", static_folder="../web/static")

# Dev mode auto-shutdown
_shutdown_timer: threading.Timer | None = None

# Global game state from log
_game_state: dict = {}


@app.route("/")
def index():
    """Main game view."""
    return render_template("game.html")


@app.route("/player/<player_name>")
def player_view(player_name: str):
    """Individual player view (limited in replay mode)."""
    return render_template("player.html", player_name=player_name)


@app.route("/api/state")
def get_state():
    """Get the replayed game state."""
    return jsonify(_game_state)


@app.route("/api/players")
def get_players():
    """Get list of all players."""
    return jsonify({
        "players": _game_state.get("players", [])
    })


@app.route("/api/messages")
def get_messages():
    """Get all public messages."""
    return jsonify({
        "messages": _game_state.get("messages", [])
    })


@app.route("/api/player/<player_name>/thoughts")
def get_player_thoughts(player_name: str):
    """Player thoughts not available in replay mode."""
    return jsonify({
        "player_name": player_name,
        "thoughts": [{"role": "system", "content": "Player thoughts not available in replay mode."}]
    })


@app.route("/events")
def events():
    """SSE endpoint - sends initial state only in replay mode."""
    def generate():
        initial_state = json.dumps({
            "event_type": "INITIAL_STATE",
            "timestamp": "replay",
            "data": _game_state
        })
        yield f"data: {initial_state}\n\n"
        
        # Keep connection alive but don't send updates
        while True:
            import time
            time.sleep(30)
            yield ": keepalive\n\n"
    
    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response


@app.route("/player/<player_name>/events")
def player_events(player_name: str):
    """SSE for player thoughts - not available in replay."""
    def generate():
        initial = json.dumps({
            "event_type": "INITIAL_THOUGHTS",
            "timestamp": "replay",
            "data": {
                "player_name": player_name,
                "thoughts": [{"role": "system", "content": "Player thoughts not available in replay mode."}]
            }
        })
        yield f"data: {initial}\n\n"
        
        while True:
            import time
            time.sleep(30)
            yield ": keepalive\n\n"
    
    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response


def shutdown_server():
    """Shutdown the Flask server."""
    print("\n⏰ Dev mode: Auto-shutdown triggered")
    # Use os._exit to force shutdown (Flask doesn't have a clean shutdown mechanism)
    import os
    os._exit(0)


def main():
    global _game_state, _shutdown_timer
    
    parser = argparse.ArgumentParser(description="Replay a completed game from its log file")
    parser.add_argument("log_file", type=str, help="Path to the game log file (.jsonl)")
    parser.add_argument("--port", type=int, default=9000, help="Port for web viewer (default: 9000)")
    parser.add_argument("--dev", action="store_true", help="Dev mode: auto-shutdown after 60 seconds")
    parser.add_argument("--dev-timeout", type=int, default=60, help="Dev mode timeout in seconds (default: 60)")
    args = parser.parse_args()
    
    # Check if file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        # Try looking in game_logs directory
        log_path = Path(__file__).parent.parent / "game_logs" / args.log_file
        if not log_path.exists():
            print(f"Error: Log file not found: {args.log_file}")
            sys.exit(1)
    
    print(f"Loading game from: {log_path}")
    _game_state = load_game_from_log(str(log_path))
    
    print(f"\n{'='*60}")
    print(f"GAME REPLAY")
    print(f"{'='*60}")
    print(f"Phase: {_game_state.get('phase', 'UNKNOWN')}")
    print(f"Day: {_game_state.get('day_number', '?')}")
    print(f"Winner: {_game_state.get('winner', 'N/A')}")
    print(f"\nPlayers:")
    for p in _game_state.get('players', []):
        status = '🟢' if p['is_alive'] else '💀'
        print(f"  {status} {p['name']}: {p['role']}")
    print(f"\nMessages: {len(_game_state.get('messages', []))}")
    print(f"Night actions: {len(_game_state.get('night_actions', []))}")
    print(f"{'='*60}")
    
    print(f"\n🌐 Web viewer available at: http://localhost:{args.port}")
    
    if args.dev:
        print(f"⚠️  Dev mode: Server will auto-shutdown in {args.dev_timeout} seconds")
        _shutdown_timer = threading.Timer(args.dev_timeout, shutdown_server)
        _shutdown_timer.daemon = True
        _shutdown_timer.start()
    else:
        print(f"   Press Ctrl+C to exit.")
    
    print()
    
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if _shutdown_timer:
            _shutdown_timer.cancel()
        sys.exit(0)


if __name__ == "__main__":
    main()
