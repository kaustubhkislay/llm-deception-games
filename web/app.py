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

from infra.mafia import GameEvent
from infra.game_engine import MafiaGame


app = Flask(__name__)

# Global game instance (set by run script)
_game: Optional[MafiaGame] = None

# Thread-safe queue for SSE events - shared between game thread and Flask
_event_queue: queue.Queue = queue.Queue(maxsize=1000)

# List of subscriber queues for broadcasting
_sse_queues: list[queue.Queue] = []
_sse_lock = threading.Lock()


def set_game(game: MafiaGame) -> None:
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
    """Main game view."""
    return render_template("game.html")


@app.route("/player/<player_name>")
def player_view(player_name: str):
    """Individual player thoughts view."""
    return render_template("player.html", player_name=player_name)


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
            {"name": p.name, "is_alive": p.is_alive}
            for p in _game.players
        ]
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
            # Send initial thoughts
            if _game:
                thoughts = _game.get_player_thoughts(player_name)
                initial_thoughts = json.dumps({
                    "event_type": "INITIAL_THOUGHTS",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "player_name": player_name,
                        "thoughts": [
                            {"role": t.role, "content": t.content, "tool_calls": t.tool_calls}
                            for t in thoughts
                        ]
                    }
                })
            else:
                initial_thoughts = json.dumps({
                    "event_type": "INITIAL_THOUGHTS",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"player_name": player_name, "thoughts": []}
                })
            yield f"data: {initial_thoughts}\n\n"
            
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


def run_app(host: str = "127.0.0.1", port: int = 8080, debug: bool = False) -> None:
    """Run the Flask app."""
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
