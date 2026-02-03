"""Event broadcasting system for ONUW game."""

import asyncio
import json
import queue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .onuw import GameEvent, EventType


# Thread-safe queue for pushing events to web viewer
_web_queue: Optional[queue.Queue] = None


def set_web_queue(q: queue.Queue) -> None:
    """Set the web queue for pushing events."""
    global _web_queue
    _web_queue = q


def push_event_to_web(event: GameEvent) -> None:
    """Push an event to the web viewer (thread-safe)."""
    if _web_queue is not None:
        try:
            event_data = json.dumps(event.to_dict())
            _web_queue.put_nowait(event_data)
        except queue.Full:
            print("Warning: Web event queue full, dropping event")
        except Exception as e:
            print(f"Error pushing event to web: {e}")


@dataclass
class EventBroadcaster:
    """Broadcasts game events to all subscribers."""
    
    _subscribers: dict[str, list[asyncio.Queue]] = field(default_factory=lambda: defaultdict(list))
    _event_history: list[GameEvent] = field(default_factory=list)
    
    def subscribe(self, channel: str = "all") -> asyncio.Queue:
        """Subscribe to events on a channel. Returns a queue that will receive events."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(q)
        return q
    
    def unsubscribe(self, q: asyncio.Queue, channel: str = "all") -> None:
        """Unsubscribe a queue from a channel."""
        if q in self._subscribers[channel]:
            self._subscribers[channel].remove(q)
    
    async def broadcast(self, event: GameEvent, channels: list[str] | None = None) -> None:
        """Broadcast an event to all subscribers on the specified channels."""
        self._event_history.append(event)
        
        # Push to web viewer
        push_event_to_web(event)
        
        if channels is None:
            channels = ["all"]
        
        for channel in channels:
            for q in self._subscribers[channel]:
                await q.put(event)
        
        # Always broadcast to "all" channel if not already included
        if "all" not in channels:
            for q in self._subscribers["all"]:
                await q.put(event)
    
    def get_history(self, event_types: list[EventType] | None = None) -> list[GameEvent]:
        """Get event history, optionally filtered by event type."""
        if event_types is None:
            return self._event_history.copy()
        return [e for e in self._event_history if e.event_type in event_types]


# Global event broadcaster instance
_broadcaster: EventBroadcaster | None = None


def get_broadcaster() -> EventBroadcaster:
    """Get or create the global event broadcaster."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = EventBroadcaster()
    return _broadcaster


def reset_broadcaster() -> None:
    """Reset the global event broadcaster (useful for testing)."""
    global _broadcaster
    _broadcaster = None
