from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Any
from datetime import datetime
import uuid


class Role(Enum):
    MAFIA = "MAFIA"
    DOCTOR = "DOCTOR"
    DETECTIVE = "DETECTIVE"
    TOWN = "TOWN"


class Phase(Enum):
    NIGHT = "NIGHT"
    DAY = "DAY"
    VOTING = "VOTING"
    GAME_OVER = "GAME_OVER"


class ActionType(Enum):
    KILL = "KILL"
    SAVE = "SAVE"
    INVESTIGATE = "INVESTIGATE"
    VOTE = "VOTE"


@dataclass
class ChatMessage:
    """A message in a player's internal LLM conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Player:
    """A player in the game."""
    name: str
    model: str
    role: Role
    is_alive: bool = True
    chat_history: list[ChatMessage] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Player):
            return self.name == other.name
        return False


@dataclass
class Action:
    """A game action (night action or vote)."""
    action_type: ActionType
    actor: Player
    target: Player
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class PublicMessage:
    """A message in the public group chat."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_name: str = ""
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NightResult:
    """The result of a night phase."""
    killed_player: Optional[Player] = None
    saved_player: Optional[Player] = None
    investigation_result: Optional[tuple[Player, bool]] = None  # (target, is_mafia)
    kill_target: Optional[Player] = None  # Who mafia tried to kill


@dataclass
class VoteResult:
    """The result of a voting phase."""
    votes: dict[str, str] = field(default_factory=dict)  # voter_name -> target_name
    lynched_player: Optional[Player] = None
    was_tie: bool = False


@dataclass
class GameState:
    """The current state of the game."""
    players: list[Player] = field(default_factory=list)
    phase: Phase = Phase.NIGHT
    day_number: int = 1
    
    # Chat state
    public_messages: list[PublicMessage] = field(default_factory=list)
    
    # Night state
    current_night_actions: dict[str, Action] = field(default_factory=dict)  # role -> action
    night_results: list[NightResult] = field(default_factory=list)
    
    # Voting state
    current_votes: dict[str, str] = field(default_factory=dict)  # voter_name -> target_name
    vote_results: list[VoteResult] = field(default_factory=list)
    
    # Game result
    winner: Optional[Literal["MAFIA", "TOWN"]] = None
    
    # Timing
    phase_start_time: Optional[datetime] = None
    phase_end_time: Optional[datetime] = None
    
    @property
    def living_players(self) -> list[Player]:
        return [p for p in self.players if p.is_alive]
    
    @property
    def dead_players(self) -> list[Player]:
        return [p for p in self.players if not p.is_alive]
    
    @property
    def living_mafia(self) -> list[Player]:
        return [p for p in self.living_players if p.role == Role.MAFIA]
    
    @property
    def living_town(self) -> list[Player]:
        return [p for p in self.living_players if p.role != Role.MAFIA]
    
    def get_player_by_name(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None


# Event types for broadcasting to web viewer
class EventType(Enum):
    GAME_START = "GAME_START"
    PHASE_CHANGE = "PHASE_CHANGE"
    PUBLIC_MESSAGE = "PUBLIC_MESSAGE"
    NIGHT_ACTION = "NIGHT_ACTION"
    NIGHT_RESULT = "NIGHT_RESULT"
    VOTE_CAST = "VOTE_CAST"
    VOTE_RESULT = "VOTE_RESULT"
    PLAYER_DEATH = "PLAYER_DEATH"
    GAME_END = "GAME_END"
    PLAYER_THOUGHT = "PLAYER_THOUGHT"  # For viewing player's internal state
    MAFIA_CHAT = "MAFIA_CHAT"  # Private mafia communication during night


@dataclass
class GameEvent:
    """An event that can be broadcast to viewers."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


# Default game configuration
DEFAULT_PLAYER_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George"
]

DEFAULT_ROLE_DISTRIBUTION = [
    Role.MAFIA, Role.MAFIA,  # 2 mafia
    Role.DOCTOR,              # 1 doctor
    Role.DETECTIVE,           # 1 detective
    Role.TOWN, Role.TOWN, Role.TOWN  # 3 town
]

DAY_PHASE_DURATION_SECONDS = 300  # 5 minutes
