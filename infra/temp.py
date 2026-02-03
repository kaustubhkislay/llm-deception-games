from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

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

@dataclass
class Player:
    id: str
    name: str
    model: str
    initial_role: Role
    current_role: Role

@dataclass
class GamePhase:
    id: str
    name: str
    description: str
    players: list[Player]

@dataclass
class Game:
    id: str
    name: str
    seed: int
    players: list[Player]
    center_roles: list[Role]
    num_rounds: int
    phases: list[GamePhase]
