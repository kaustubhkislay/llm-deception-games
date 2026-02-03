"""One Night Ultimate Werewolf data structures and types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Any
from datetime import datetime
import uuid
import random


class Role(Enum):
    """ONUW roles."""
    # Village Team
    VILLAGER = "VILLAGER"           # No ability
    SEER = "SEER"                   # Look at one player's card OR two center cards
    ROBBER = "ROBBER"               # Swap your card with another player's, see new card
    TROUBLEMAKER = "TROUBLEMAKER"   # Swap two other players' cards (not yours)
    DRUNK = "DRUNK"                 # Swap your card with a center card (don't look)
    INSOMNIAC = "INSOMNIAC"         # Look at your card at end of night (after swaps)
    HUNTER = "HUNTER"               # If you die, your vote target also dies
    
    # Werewolf Team  
    WEREWOLF = "WEREWOLF"           # See other werewolves; if alone, may look at center
    MINION = "MINION"               # Knows werewolves but they don't know you; wins with wolves
    
    # Neutral
    TANNER = "TANNER"               # Wins only if you get lynched


# Team affiliations
VILLAGE_TEAM = {Role.VILLAGER, Role.SEER, Role.ROBBER, Role.TROUBLEMAKER, 
                Role.DRUNK, Role.INSOMNIAC, Role.HUNTER}
WEREWOLF_TEAM = {Role.WEREWOLF, Role.MINION}
NEUTRAL_TEAM = {Role.TANNER}


def get_team(role: Role) -> str:
    """Get the team affiliation for a role."""
    if role in VILLAGE_TEAM:
        return "VILLAGE"
    elif role in WEREWOLF_TEAM:
        return "WEREWOLF"
    else:
        return "NEUTRAL"


# Night action order - roles act in this sequence
NIGHT_ACTION_ORDER = [
    Role.WEREWOLF,      # 1. Werewolves wake, see each other (or peek center if alone)
    Role.MINION,        # 2. Minion sees werewolves
    Role.SEER,          # 3. Seer looks at one player OR two center cards
    Role.ROBBER,        # 4. Robber swaps and looks at new card
    Role.TROUBLEMAKER,  # 5. Troublemaker swaps two other players
    Role.DRUNK,         # 6. Drunk swaps with center (blind)
    Role.INSOMNIAC,     # 7. Insomniac looks at own card
]

# Roles with no night action
PASSIVE_ROLES = {Role.VILLAGER, Role.HUNTER, Role.TANNER}


class Phase(Enum):
    NIGHT = "NIGHT"
    DAY = "DAY"
    VOTING = "VOTING"
    GAME_OVER = "GAME_OVER"


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
    original_role: Role  # The role they started with (used for night actions)
    current_role: Role   # Their current role (may be swapped)
    chat_history: list[ChatMessage] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Player):
            return self.name == other.name
        return False


@dataclass 
class PublicMessage:
    """A message in the public group chat."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_name: str = ""
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    round_number: int = 0  # Which discussion round this message was sent in (0 = GM message)


@dataclass
class NightAction:
    """A night action taken by a player."""
    player_name: str
    original_role: Role  # The role that grants this action
    action_type: str     # "look_player", "look_center", "rob", "swap", "drunk_swap"
    targets: list[str]   # Target player names or center positions
    result: Optional[Any] = None  # What they learned (if anything)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VoteResult:
    """The result of the voting phase."""
    votes: dict[str, str] = field(default_factory=dict)  # voter_name -> target_name
    killed_players: list[str] = field(default_factory=list)  # Players who died (ties = multiple)
    hunter_kill: Optional[str] = None  # Additional kill from Hunter ability


@dataclass
class GameState:
    """The current state of the game."""
    players: list[Player] = field(default_factory=list)
    center_cards: list[Role] = field(default_factory=list)  # 3 cards in center
    
    # Track original assignments (for display and night action resolution)
    original_assignments: dict[str, Role] = field(default_factory=dict)  # player_name -> starting role
    
    # Night action log
    night_actions: list[NightAction] = field(default_factory=list)
    
    phase: Phase = Phase.NIGHT
    
    # Chat state
    public_messages: list[PublicMessage] = field(default_factory=list)
    current_round: int = 0  # Current discussion round (0 = not in discussion)
    total_rounds: int = 0   # Total discussion rounds for this game
    
    # Voting state
    current_votes: dict[str, str] = field(default_factory=dict)  # voter_name -> target_name
    vote_result: Optional[VoteResult] = None
    
    # Game result
    winner: Optional[str] = None  # "VILLAGE", "WEREWOLF", "TANNER", or None
    
    # Timing
    phase_start_time: Optional[datetime] = None
    phase_end_time: Optional[datetime] = None
    
    def get_player_by_name(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None
    
    def get_current_role(self, player_name: str) -> Optional[Role]:
        """Get a player's current role (after any swaps)."""
        player = self.get_player_by_name(player_name)
        return player.current_role if player else None
    
    def get_center_card(self, position: int) -> Optional[Role]:
        """Get a center card by position (0, 1, or 2)."""
        if 0 <= position < len(self.center_cards):
            return self.center_cards[position]
        return None
    
    def swap_player_cards(self, player1_name: str, player2_name: str) -> None:
        """Swap the current roles of two players."""
        p1 = self.get_player_by_name(player1_name)
        p2 = self.get_player_by_name(player2_name)
        if p1 and p2:
            p1.current_role, p2.current_role = p2.current_role, p1.current_role
    
    def swap_player_with_center(self, player_name: str, center_position: int) -> None:
        """Swap a player's card with a center card."""
        player = self.get_player_by_name(player_name)
        if player and 0 <= center_position < len(self.center_cards):
            player.current_role, self.center_cards[center_position] = \
                self.center_cards[center_position], player.current_role
    
    def get_werewolf_players(self) -> list[str]:
        """Get names of players who currently have the Werewolf role."""
        return [p.name for p in self.players if p.current_role == Role.WEREWOLF]
    
    def get_players_with_original_role(self, role: Role) -> list[Player]:
        """Get players who started with a specific role (for night actions)."""
        return [p for p in self.players if p.original_role == role]


# Event types for broadcasting to web viewer
class EventType(Enum):
    GAME_START = "GAME_START"
    PHASE_CHANGE = "PHASE_CHANGE"
    PUBLIC_MESSAGE = "PUBLIC_MESSAGE"
    PLAYER_THOUGHT = "PLAYER_THOUGHT"
    GM_MESSAGE = "GM_MESSAGE"
    
    # ONUW-specific events
    ROLE_ASSIGNMENT = "ROLE_ASSIGNMENT"   # Initial role dealt
    NIGHT_ACTION = "NIGHT_ACTION"         # A player performed their night action
    CARD_SWAP = "CARD_SWAP"               # Cards were swapped
    ROLE_PEEK = "ROLE_PEEK"               # Player looked at a card
    
    # Round-based discussion events
    ROUND_START = "ROUND_START"           # A discussion round is starting
    ROUND_END = "ROUND_END"               # A discussion round has ended
    
    # Voting and game end
    VOTE_CAST = "VOTE_CAST"
    VOTE_RESULT = "VOTE_RESULT"
    PLAYER_DEATH = "PLAYER_DEATH"
    GAME_END = "GAME_END"


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
    "Alice", "Bob", "Charlie", "Diana", "Edward"
]

# Available role pool - select from these for each game
# Need at least num_players + 3 roles
AVAILABLE_ROLES = [
    Role.WEREWOLF, Role.WEREWOLF,  # 2 werewolves
    Role.SEER,
    Role.ROBBER,
    Role.TROUBLEMAKER,
    Role.DRUNK,
    Role.INSOMNIAC,
    Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
    Role.MINION,
    Role.TANNER,
    Role.HUNTER,
]

# Simpler default setup for 5 players (5 + 3 = 8 cards)
DEFAULT_ROLE_POOL = [
    Role.WEREWOLF, Role.WEREWOLF,
    Role.SEER,
    Role.ROBBER,
    Role.TROUBLEMAKER,
    Role.VILLAGER, Role.VILLAGER,
    Role.DRUNK,
]

DEFAULT_DISCUSSION_ROUNDS = 5  # Number of discussion rounds before voting


@dataclass
class GameConfig:
    """Configuration for a deterministic game run."""
    seed: int
    models: list[str]  # One model per player
    roles: list[Role]  # Exactly num_players + 3 roles
    names: Optional[list[str]] = None  # Player names (defaults to DEFAULT_PLAYER_NAMES)
    num_rounds: int = DEFAULT_DISCUSSION_ROUNDS
    
    def __post_init__(self):
        if self.names is None:
            self.names = DEFAULT_PLAYER_NAMES[:len(self.models)]
        
        num_players = len(self.models)
        required_roles = num_players + 3
        
        assert len(self.names) == num_players, \
            f"Number of names ({len(self.names)}) must match number of models ({num_players})"
        assert len(self.roles) == required_roles, \
            f"Need exactly {required_roles} roles for {num_players} players, got {len(self.roles)}"
    
    @property
    def num_players(self) -> int:
        return len(self.models)
    
    def to_dict(self) -> dict:
        """Serialize config for logging/storage."""
        return {
            "seed": self.seed,
            "models": self.models,
            "roles": [r.value for r in self.roles],
            "names": self.names,
            "num_rounds": self.num_rounds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        """Deserialize config from dict."""
        return cls(
            seed=data["seed"],
            models=data["models"],
            roles=[Role(r) for r in data["roles"]],
            names=data.get("names"),
            num_rounds=data.get("num_rounds", DEFAULT_DISCUSSION_ROUNDS),
        )


@dataclass
class NightActionPlan:
    """Pre-determined night action for a player."""
    player_name: str
    original_role: Role
    action_type: str  # "look_player", "look_center", "rob", "swap", "drunk_swap", "werewolf_look", "acknowledge"
    targets: list[str]  # Target player names or "center_0", "center_1", "center_2"


@dataclass
class SeededGameSetup:
    """
    Deterministic game setup based on a seed.
    
    Given a GameConfig, this class uses the seed to deterministically assign:
    - Roles to players and center
    - Night action targets for each role
    """
    config: GameConfig
    
    # Computed assignments
    player_roles: dict[str, Role] = field(default_factory=dict)  # name -> role
    center_roles: list[Role] = field(default_factory=list)  # 3 center cards
    night_action_plan: dict[str, NightActionPlan] = field(default_factory=dict)  # name -> action
    
    def __post_init__(self):
        self._setup_game()
    
    def _setup_game(self) -> None:
        """Use the seed to deterministically set up the game."""
        rng = random.Random(self.config.seed)
        
        # Shuffle roles deterministically
        all_roles = list(self.config.roles)
        rng.shuffle(all_roles)
        
        # Assign to players and center
        num_players = self.config.num_players
        player_role_list = all_roles[:num_players]
        self.center_roles = all_roles[num_players:]
        
        # Sort names alphabetically for consistent ordering
        # names is guaranteed to be set by __post_init__
        assert self.config.names is not None
        sorted_names = sorted(self.config.names)
        
        # Assign roles to sorted names
        self.player_roles = {
            name: role for name, role in zip(sorted_names, player_role_list)
        }
        
        # Plan night actions
        self._plan_night_actions(rng, sorted_names)
    
    def _plan_night_actions(self, rng: random.Random, player_names: list[str]) -> None:
        """Pre-determine all night action targets using the seeded RNG."""
        
        for player_name, role in self.player_roles.items():
            other_players = [n for n in player_names if n != player_name]
            
            if role == Role.SEER:
                # Seer: randomly choose between looking at player or center
                if rng.random() < 0.5:
                    # Look at one player
                    target = rng.choice(other_players)
                    self.night_action_plan[player_name] = NightActionPlan(
                        player_name=player_name,
                        original_role=role,
                        action_type="look_player",
                        targets=[target]
                    )
                else:
                    # Look at two center cards
                    positions = rng.sample([0, 1, 2], 2)
                    self.night_action_plan[player_name] = NightActionPlan(
                        player_name=player_name,
                        original_role=role,
                        action_type="look_center",
                        targets=[f"center_{p}" for p in positions]
                    )
            
            elif role == Role.ROBBER:
                # Robber: choose one other player to rob
                target = rng.choice(other_players)
                self.night_action_plan[player_name] = NightActionPlan(
                    player_name=player_name,
                    original_role=role,
                    action_type="rob",
                    targets=[target]
                )
            
            elif role == Role.TROUBLEMAKER:
                # Troublemaker: choose two other players to swap
                targets = rng.sample(other_players, 2)
                self.night_action_plan[player_name] = NightActionPlan(
                    player_name=player_name,
                    original_role=role,
                    action_type="swap",
                    targets=targets
                )
            
            elif role == Role.DRUNK:
                # Drunk: choose center position
                position = rng.randint(0, 2)
                self.night_action_plan[player_name] = NightActionPlan(
                    player_name=player_name,
                    original_role=role,
                    action_type="drunk_swap",
                    targets=[f"center_{position}"]
                )
            
            elif role == Role.WEREWOLF:
                # Werewolf: Will determine at runtime if lone (then looks at center)
                # Pre-plan the center position in case they're alone
                position = rng.randint(0, 2)
                self.night_action_plan[player_name] = NightActionPlan(
                    player_name=player_name,
                    original_role=role,
                    action_type="werewolf_look",  # May become "acknowledge" if not alone
                    targets=[f"center_{position}"]
                )
            
            elif role == Role.MINION:
                # Minion just acknowledges seeing werewolves
                self.night_action_plan[player_name] = NightActionPlan(
                    player_name=player_name,
                    original_role=role,
                    action_type="acknowledge",
                    targets=[]
                )
            
            elif role == Role.INSOMNIAC:
                # Insomniac just acknowledges seeing their card
                self.night_action_plan[player_name] = NightActionPlan(
                    player_name=player_name,
                    original_role=role,
                    action_type="acknowledge",
                    targets=[]
                )
            
            # Passive roles (VILLAGER, HUNTER, TANNER) have no night action plan


def select_roles_for_game(num_players: int, role_pool: Optional[list[Role]] = None) -> tuple[list[Role], list[Role]]:
    """
    Select and distribute roles for a game.
    
    Args:
        num_players: Number of players in the game
        role_pool: Pool of roles to select from (default: DEFAULT_ROLE_POOL)
    
    Returns:
        Tuple of (player_roles, center_roles) where:
        - player_roles: List of roles for players (length = num_players)
        - center_roles: List of roles for center (length = 3)
    """
    if role_pool is None:
        role_pool = DEFAULT_ROLE_POOL
    
    required = num_players + 3
    assert len(role_pool) >= required, \
        f"Need at least {required} roles for {num_players} players, but only have {len(role_pool)}"
    
    # Shuffle and split
    selected = random.sample(role_pool, required)
    player_roles = selected[:num_players]
    center_roles = selected[num_players:]
    
    return player_roles, center_roles


def determine_winner(state: GameState) -> tuple[str, list[str]]:
    """
    Determine the winner after voting.
    
    Args:
        state: The game state after voting
    
    Returns:
        Tuple of (winner, killed_players) where:
        - winner: "VILLAGE", "WEREWOLF", or "TANNER"
        - killed_players: List of player names who were killed
    
    Note: In ties, ALL tied players die. There is no "no one dies" outcome.
    """
    votes = state.current_votes
    
    # Count votes for each player
    from collections import Counter
    vote_counts = Counter(votes.values())
    
    # Find max votes and who got them (ties = multiple deaths)
    max_votes = max(vote_counts.values())
    killed = [p for p, v in vote_counts.items() if v == max_votes]
    
    # Get the roles of killed players
    killed_roles = [state.get_current_role(p) for p in killed]
    
    # Check for Hunter - if Hunter dies, their vote target also dies
    hunter_additional_kill = None
    for player_name in killed:
        player = state.get_player_by_name(player_name)
        if player and player.current_role == Role.HUNTER:
            # Hunter's vote target also dies
            hunter_target = votes.get(player_name)
            if hunter_target and hunter_target not in killed:
                hunter_additional_kill = hunter_target
                killed.append(hunter_target)
                killed_roles.append(state.get_current_role(hunter_target))
    
    # Check Tanner first (special win condition)
    if Role.TANNER in killed_roles:
        # Tanner wins if they died
        # But if a werewolf also died, village also wins (shared victory in some rules)
        # For simplicity: Tanner win takes precedence
        return "TANNER", killed
    
    # Check if any werewolf was killed
    werewolves_in_game = [p.name for p in state.players if p.current_role == Role.WEREWOLF]
    
    if not werewolves_in_game:
        # No werewolves in game - Minion wins only if they are killed
        if Role.MINION in killed_roles:
            # Minion was killed - Minion/Werewolf team wins
            # (Minion's goal when no werewolves: get themselves killed)
            return "WEREWOLF", killed
        else:
            # Minion survived - village wins
            return "VILLAGE", killed
    
    # Normal case: village wins if at least one werewolf killed
    if Role.WEREWOLF in killed_roles:
        return "VILLAGE", killed
    else:
        return "WEREWOLF", killed
