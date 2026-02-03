"""Flask application with SSE endpoints for real-time game updates."""

import json
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Generator

from flask import Flask, render_template, Response, jsonify

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.onuw import GameEvent
from infra.game_engine import ONUWGame, load_game_from_log

# Directory where game logs are stored
GAME_LOGS_DIR = Path(__file__).parent.parent / "game_logs"

app = Flask(__name__)

# Global game instance (set by run script)
_game: Optional[ONUWGame] = None

# Thread-safe queue for SSE events - shared between game thread and Flask
_event_queue: queue.Queue = queue.Queue(maxsize=1000)

# List of subscriber queues for broadcasting
_sse_queues: list[queue.Queue] = []
_sse_lock = threading.Lock()


def set_game(game: ONUWGame) -> None:
    """Set the global game instance."""
    global _game
    _game = game


def get_event_queue() -> queue.Queue:
    """Get the event queue for the game to push events to."""
    return _event_queue


def _subscribe_to_events() -> queue.Queue:
    """Create a new SSE queue and subscribe to events."""
    q: queue.Queue = queue.Queue(maxsize=100)
    with _sse_lock:
        _sse_queues.append(q)
    return q


def _unsubscribe_from_events(q: queue.Queue) -> None:
    """Remove an SSE queue."""
    with _sse_lock:
        if q in _sse_queues:
            _sse_queues.remove(q)


def broadcast_event(event_data: str) -> None:
    """Broadcast an event to all SSE subscribers."""
    with _sse_lock:
        dead_queues = []
        for q in _sse_queues:
            try:
                q.put_nowait(event_data)
            except queue.Full:
                dead_queues.append(q)
        # Clean up dead queues
        for q in dead_queues:
            _sse_queues.remove(q)


def event_dispatcher_thread() -> None:
    """Thread that dispatches events from game to SSE subscribers."""
    while True:
        try:
            event_data = _event_queue.get(timeout=1)
            broadcast_event(event_data)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Event dispatcher error: {e}")


# Start the dispatcher thread
_dispatcher = threading.Thread(target=event_dispatcher_thread, daemon=True)
_dispatcher.start()


@app.route("/")
def index():
    """Games list view."""
    return render_template("games_list.html")


@app.route("/game/<game_id>")
def game_view(game_id: str):
    """View a specific game."""
    return render_template("game.html", game_id=game_id)


@app.route("/game/<game_id>/player/<player_name>")
def player_view(game_id: str, player_name: str):
    """Individual player thoughts view."""
    return render_template("player.html", player_name=player_name, game_id=game_id)


def _get_game_metadata(log_file: Path) -> dict:
    """Extract basic metadata from a game log file (without loading full state)."""
    game_id = log_file.stem.replace("game_", "")
    name = None
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("event_type") == "GAME_INIT":
                    name = event.get("data", {}).get("name")
                    break
    except Exception:
        pass
    
    return {
        "game_id": game_id,
        "name": name,
    }


@app.route("/api/games")
def list_games():
    """List all available games in game_logs/ directory."""
    games = []
    
    if GAME_LOGS_DIR.exists():
        log_files = sorted(GAME_LOGS_DIR.glob("game_*.jsonl"), reverse=True)
        for log_file in log_files:
            metadata = _get_game_metadata(log_file)
            games.append(metadata)
    
    return jsonify({"games": games})


def _load_game_state(game_id: str) -> Optional[dict]:
    """Load game state - from live game if it matches, otherwise from log file."""
    # Check if this is the currently running live game
    if _game is not None and _game.game_id == game_id:
        return _game.get_game_state_dict()
    
    # Otherwise, try to load from log file
    log_file = GAME_LOGS_DIR / f"game_{game_id}.jsonl"
    if log_file.exists():
        return load_game_from_log(str(log_file))
    
    return None


def _is_live_game(game_id: str) -> bool:
    """Check if the given game_id matches the currently running live game."""
    return _game is not None and _game.game_id == game_id


@app.route("/api/game/<game_id>/state")
def get_game_state(game_id: str):
    """Get state for a specific game."""
    state = _load_game_state(game_id)
    if state is None:
        return jsonify({"error": f"Game {game_id} not found", "phase": "NOT_FOUND", "players": []}), 404
    return jsonify(state)


@app.route("/game/<game_id>/events")
def game_events(game_id: str):
    """SSE endpoint for a specific game's events."""
    def generate() -> Generator[str, None, None]:
        # Check if this is a live game
        is_live = _is_live_game(game_id)
        
        # Send initial state
        state = _load_game_state(game_id)
        if state is None:
            initial_state = json.dumps({
                "event_type": "INITIAL_STATE",
                "timestamp": datetime.now().isoformat(),
                "data": {"error": f"Game {game_id} not found", "phase": "NOT_FOUND", "players": []}
            })
            yield f"data: {initial_state}\n\n"
            return
        
        initial_state = json.dumps({
            "event_type": "INITIAL_STATE",
            "timestamp": datetime.now().isoformat(),
            "data": state
        })
        yield f"data: {initial_state}\n\n"
        
        if is_live:
            # For live games, subscribe to events
            q = _subscribe_to_events()
            try:
                while True:
                    try:
                        event_data = q.get(timeout=15)
                        yield f"data: {event_data}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            except GeneratorExit:
                pass
            finally:
                _unsubscribe_from_events(q)
        else:
            # For replay, just send keepalives (no live updates)
            while True:
                time.sleep(30)
                yield ": keepalive\n\n"
    
    response = Response(
        generate(),
        mimetype="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/state")
def get_state():
    """Get current game state."""
    if _game is None:
        return jsonify({"error": "No game running", "phase": "WAITING", "players": [], "messages": []}), 200
    return jsonify(_game.get_game_state_dict())


@app.route("/api/players")
def get_players():
    """Get list of all players."""
    if _game is None:
        return jsonify({"players": []})
    return jsonify({
        "players": [
            {"name": p.name, "original_role": p.original_role.value, "current_role": p.current_role.value}
            for p in _game.players
        ],
        "center_cards": [r.value for r in _game.state.center_cards]
    })


@app.route("/api/player/<player_name>/thoughts")
def get_player_thoughts(player_name: str):
    """Get a player's internal thoughts."""
    if _game is None:
        return jsonify({"player_name": player_name, "thoughts": []})
    
    thoughts = _game.get_player_thoughts(player_name)
    return jsonify({
        "player_name": player_name,
        "thoughts": [
            {
                "role": t.role,
                "content": t.content,
                "tool_calls": t.tool_calls,
            }
            for t in thoughts
        ]
    })


@app.route("/api/messages")
def get_messages():
    """Get all public messages."""
    if _game is None:
        return jsonify({"messages": []})
    
    return jsonify({
        "messages": [
            {
                "id": m.id,
                "sender": m.sender_name,
                "content": m.content,
                "timestamp": m.timestamp.isoformat()
            }
            for m in _game.state.public_messages
        ]
    })


@app.route("/events")
def events():
    """SSE endpoint for real-time game events."""
    def generate() -> Generator[str, None, None]:
        q = _subscribe_to_events()
        try:
            # Send initial state immediately
            if _game:
                initial_state = json.dumps({
                    "event_type": "INITIAL_STATE",
                    "timestamp": datetime.now().isoformat(),
                    "data": _game.get_game_state_dict()
                })
            else:
                initial_state = json.dumps({
                    "event_type": "INITIAL_STATE",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"phase": "WAITING", "players": [], "messages": []}
                })
            yield f"data: {initial_state}\n\n"
            
            # Stream events
            while True:
                try:
                    event_data = q.get(timeout=15)
                    yield f"data: {event_data}\n\n"
                except queue.Empty:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            _unsubscribe_from_events(q)
    
    response = Response(
        generate(),
        mimetype="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/player/<player_name>/events")
def player_events(player_name: str):
    """SSE endpoint for a specific player's thoughts."""
    def generate() -> Generator[str, None, None]:
        q = _subscribe_to_events()
        try:
            # Send initial state (for timeline)
            if _game:
                initial_state = json.dumps({
                    "event_type": "INITIAL_STATE",
                    "timestamp": datetime.now().isoformat(),
                    "data": _game.get_game_state_dict()
                })
            else:
                initial_state = json.dumps({
                    "event_type": "INITIAL_STATE",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"phase": "WAITING", "players": [], "phase_history": [], "player_thoughts": {}}
                })
            yield f"data: {initial_state}\n\n"
            
            # Stream events
            while True:
                try:
                    event_data = q.get(timeout=15)
                    event = json.loads(event_data)
                    
                    # Only send player thought events for this player
                    if event.get("event_type") == "PLAYER_THOUGHT":
                        if event.get("data", {}).get("player_name") == player_name:
                            yield f"data: {event_data}\n\n"
                    else:
                        # Send all other events (phase changes, etc.)
                        yield f"data: {event_data}\n\n"
                        
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            _unsubscribe_from_events(q)
    
    response = Response(
        generate(),
        mimetype="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/stop", methods=["POST"])
def stop_server():
    """Stop the game server gracefully."""
    import os
    import signal
    
    def shutdown():
        os.kill(os.getpid(), signal.SIGTERM)
    
    # Schedule shutdown after response
    from threading import Timer
    Timer(0.5, shutdown).start()
    
    return jsonify({"status": "stopping", "message": "Server shutting down..."})


def run_app(host: str = "127.0.0.1", port: int = 8080, debug: bool = False) -> None:
    """Run the Flask app."""
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
