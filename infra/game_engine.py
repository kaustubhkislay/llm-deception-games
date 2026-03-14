"""Game engine for One Night Ultimate Werewolf."""

import asyncio
import json
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from collections import Counter

from .onuw import (
    Player, Role, Phase, GameState, PublicMessage, NightAction, VoteResult,
    GameEvent, EventType, ChatMessage,
    NIGHT_ACTION_ORDER, PASSIVE_ROLES, determine_winner, get_team,
    GameConfig, SeededGameSetup, NightActionPlan
)
from .llm_client import get_llm_client, CachedLLMClient
from .player import PlayerAgent
from .events import get_broadcaster, EventBroadcaster


# Set up file-based logging
LOG_DIR = Path(__file__).parent.parent / "game_logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_game_logger(game_id: str) -> logging.Logger:
    """Set up a logger for a specific game that writes to a file."""
    logger = logging.getLogger(f"onuw_game_{game_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    log_file = LOG_DIR / f"game_{game_id}.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class GameEventLog:
    """JSON-lines event log for game replay."""
    
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.log_file = LOG_DIR / f"game_{game_id}.jsonl"
        self._file = open(self.log_file, 'a')
    
    def log_event(self, event_type: str, data: dict) -> None:
        """Log an event to the JSON lines file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()
    
    def close(self) -> None:
        """Close the log file."""
        self._file.close()


def load_game_from_log(log_path: str) -> dict:
    """Load and reconstruct game state from a JSON lines log file."""
    players: dict[str, dict] = {}
    center_cards: list[str] = []
    all_messages: list[dict] = []
    all_votes: dict[str, dict] = {}
    phase = "GAME_OVER"
    winner: Optional[str] = None
    killed_players: list[str] = []
    game_name: Optional[str] = None
    game_id: Optional[str] = None
    phase_history: list[dict] = []
    player_thoughts: dict[str, dict[str, list]] = {}
    current_phase_events: list[dict] = []
    current_phase_messages: list[dict] = []
    current_phase_votes: dict[str, dict] = {}
    current_phase_gm_messages: list[dict] = []
    
    def snapshot_players():
        return [
            {"name": name, "model": info.get("model"), "current_role": info["current_role"], "original_role": info["original_role"]}
            for name, info in players.items()
        ]
    
    def save_phase_snapshot(phase_name: str):
        phase_history.append({
            "phase": phase_name,
            "dayNumber": 1,  # ONUW is always day 1
            "players": snapshot_players(),
            "centerCards": list(center_cards),
            "events": list(current_phase_events),
            "messages": list(current_phase_messages),
            "votes": dict(current_phase_votes),
            "gmMessages": list(current_phase_gm_messages),
        })
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            event = json.loads(line)
            event_type = event.get("event_type", "")
            data = event.get("data", {})
            timestamp = event.get("timestamp", "")
            
            if event_type == "GAME_INIT":
                game_id = data.get("game_id")
                game_name = data.get("name")
                for p in data.get("players", []):
                    players[p["name"]] = {
                        "model": p.get("model"),
                        "original_role": p["original_role"],
                        "current_role": p["current_role"]
                    }
                center_cards = data.get("center_cards", [])
            
            elif event_type == "PHASE_CHANGE":
                if phase != "GAME_OVER" and current_phase_events:
                    pass  # Phase already saved
                
                phase = data.get("phase", phase)
                current_phase_events = []
                current_phase_messages = []
                current_phase_votes = {}
                current_phase_gm_messages = []
                save_phase_snapshot(phase)
            
            elif event_type == "GM_MESSAGE":
                gm_msg = {
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp", timestamp)
                }
                current_phase_gm_messages.append(gm_msg)
                current_phase_events.append({"type": "gm_message", "data": gm_msg})
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
                    phase_history[-1]["gmMessages"] = list(current_phase_gm_messages)
            
            elif event_type == "NIGHT_ACTION":
                action_data = {
                    "type": "night_action",
                    "player": data.get("player"),
                    "role": data.get("role"),
                    "action": data.get("action"),
                    "targets": data.get("targets"),
                    "result": data.get("result"),
                    "timestamp": timestamp
                }
                current_phase_events.append(action_data)
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "CARD_SWAP":
                swap_data = {
                    "type": "card_swap",
                    "location1": data.get("location1"),
                    "location2": data.get("location2"),
                    "timestamp": timestamp
                }
                current_phase_events.append(swap_data)
                
                # Update player roles after swap
                for p_name, p_data in data.get("updated_players", {}).items():
                    if p_name in players:
                        players[p_name]["current_role"] = p_data.get("current_role")
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
                    phase_history[-1]["players"] = snapshot_players()
            
            elif event_type == "PUBLIC_MESSAGE":
                msg = {
                    "type": "message",
                    "id": data.get("id"),
                    "sender": data.get("sender"),
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp", timestamp),
                    "round": data.get("round", 0)
                }
                all_messages.append(msg)
                current_phase_messages.append(msg)
                current_phase_events.append(msg)
                
                if phase_history:
                    phase_history[-1]["messages"] = list(current_phase_messages)
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "VOTE_REASONING":
                voter = data.get("voter")
                target = data.get("target")
                reasoning = data.get("reasoning")
                
                vote_data = {
                    "type": "vote",
                    "voter": voter,
                    "target": target,
                    "reasoning": reasoning,
                    "timestamp": timestamp
                }
                current_phase_votes[voter] = vote_data
                all_votes[voter] = vote_data
                current_phase_events.append(vote_data)
                
                if phase_history:
                    phase_history[-1]["votes"] = dict(current_phase_votes)
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "VOTE_RESULT":
                result_data = {
                    "type": "vote_result",
                    "votes": data.get("votes", {}),
                    "killed": data.get("killed", []),
                    "timestamp": timestamp
                }
                current_phase_events.append(result_data)
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "PLAYER_THOUGHT":
                player_name = data.get("player_name")
                thought_phase = data.get("phase", phase)
                phase_key = f"1_{thought_phase}"
                
                if player_name not in player_thoughts:
                    player_thoughts[player_name] = {}
                if phase_key not in player_thoughts[player_name]:
                    player_thoughts[player_name][phase_key] = []
                
                # If this thought includes the system prompt, prepend it
                # This allows reconstructing the full chat thread by appending all phases
                system_prompt = data.get("system_prompt")
                if system_prompt:
                    player_thoughts[player_name][phase_key].append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                player_thoughts[player_name][phase_key].append({
                    "role": "user",
                    "content": data.get("prompt")
                })
                player_thoughts[player_name][phase_key].append({
                    "role": "assistant", 
                    "content": data.get("response"),
                    "tool_calls": data.get("tool_calls")
                })
            
            elif event_type == "GAME_END":
                winner = data.get("winner")
                killed_players = data.get("killed", [])
                for p in data.get("players", []):
                    if p["name"] in players:
                        players[p["name"]]["current_role"] = p["current_role"]
                
                current_phase_events = []
                current_phase_messages = []
                current_phase_votes = {}
                current_phase_gm_messages = []
                save_phase_snapshot("GAME_OVER")
    
    player_list = snapshot_players()
    
    return {
        "game_id": game_id,
        "name": game_name,
        "phase": phase,
        "day_number": 1,
        "players": player_list,
        "center_cards": center_cards,
        "messages": all_messages,
        "current_votes": {k: v.get("target") for k, v in all_votes.items()},
        "winner": winner,
        "killed": killed_players,
        "phase_history": phase_history,
        "player_thoughts": player_thoughts,
    }


class ONUWGame:
    """Main game engine for One Night Ultimate Werewolf."""
    
    def __init__(
        self,
        config: GameConfig,
        name: Optional[str] = None,
    ):
        """
        Initialize an ONUW game from a GameConfig.
        
        Args:
            config: GameConfig with seed, models, roles, names, num_rounds
            name: Optional display name for this game
        """
        # Set up game logger first - include seed in game_id for uniqueness in parallel runs
        self.game_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.seed}"
        self.logger = setup_game_logger(self.game_id)
        self.event_log = GameEventLog(self.game_id)
        
        self.name = name
        self.llm_client = get_llm_client(use_cache=True)
        self.broadcaster = get_broadcaster()
        
        # Store config and setup
        self.config: GameConfig = config
        self.game_setup: SeededGameSetup = SeededGameSetup(config)
        self.num_rounds = config.num_rounds
        
        # names is guaranteed to be set by GameConfig.__post_init__
        assert config.names is not None
        
        # Get sorted names (SeededGameSetup sorts them)
        sorted_names = sorted(config.names)
        
        # Create players with per-player models and reasoning settings
        self.players: list[Player] = []
        for i, player_name in enumerate(sorted_names):
            role = self.game_setup.player_roles[player_name]
            # Find the model config for this player (models are in original name order)
            original_index = config.names.index(player_name)
            model_config = config.models[original_index]
            
            self.players.append(Player(
                name=player_name,
                model=model_config.model,
                original_role=role,
                current_role=role,
                reasoning_settings=model_config.get_reasoning_settings()
            ))
        
        center_roles = self.game_setup.center_roles
        
        # Initialize game state
        self.state = GameState(
            players=self.players,
            center_cards=center_roles,
            original_assignments={p.name: p.original_role for p in self.players},
            total_rounds=self.num_rounds
        )
        
        # Communication infrastructure
        self._message_queues: dict[str, asyncio.Queue] = {
            p.name: asyncio.Queue() for p in self.players
        }
        
        # Active roles in this game (players + center)
        active_roles = [p.original_role for p in self.players] + list(center_roles)
        
        # Player agents
        self.agents: dict[str, PlayerAgent] = {}
        for player in self.players:
            self.agents[player.name] = PlayerAgent(
                player=player,
                llm_client=self.llm_client,
                message_queue=self._message_queues[player.name],
                active_roles=active_roles,
                log_thought_callback=lambda data: self.event_log.log_event("PLAYER_THOUGHT", data),
            )
        
        # Log game initialization
        self.logger.info(f"Game {self.game_id} initialized with {len(self.players)} players")
        for p in self.players:
            self.logger.info(f"  Player: {p.name} | Model: {p.model} | Role: {p.original_role.value}")
        self.logger.info(f"  Center cards: {[r.value for r in center_roles]}")
        if config:
            self.logger.info(f"  Seed: {config.seed}")
        
        init_data = {
            "game_id": self.game_id,
            "name": self.name,
            "players": [{"name": p.name, "model": p.model, "original_role": p.original_role.value, "current_role": p.current_role.value} for p in self.players],
            "center_cards": [r.value for r in center_roles]
        }
        if config:
            init_data["config"] = config.to_dict()
        self.event_log.log_event("GAME_INIT", init_data)
        
        # Error tracking
        self.errors: list[dict] = []
        
        print(f"Game initialized with {len(self.players)} players:")
        for p in self.players:
            print(f"  - {p.name} ({p.model}): {p.original_role.value}")
        print(f"Center cards: {[r.value for r in center_roles]}")
        if config:
            print(f"Seed: {config.seed}")
    def _track_error(self, error_type: str, details: dict) -> None:
        """Track an error for later analysis."""
        error = {
            "type": error_type,
            "timestamp": datetime.now().isoformat(),
            "phase": self.state.phase.value if self.state.phase else "UNKNOWN",
            **details
        }
        self.errors.append(error)
        self.logger.warning(f"Error tracked: {error_type} - {details}")
        self.event_log.log_event("ERROR", error)
    
    async def _emit_gm_message(self, content: str) -> None:
        """Emit a game master narration message."""
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.GM_MESSAGE,
                data={"content": content, "timestamp": datetime.now().isoformat()}
            )
        )
        self.event_log.log_event("GM_MESSAGE", {"content": content, "timestamp": datetime.now().isoformat()})
    
    async def _broadcast_round_messages(self, round_num: int, messages: list[tuple[str, str]]) -> None:
        """Broadcast all messages from a discussion round simultaneously."""
        timestamp = datetime.now()
        
        for sender_name, content in messages:
            msg = PublicMessage(
                sender_name=sender_name,
                content=content,
                timestamp=timestamp,
                round_number=round_num
            )
            self.state.public_messages.append(msg)
            
            await self.broadcaster.broadcast(
                GameEvent(
                    event_type=EventType.PUBLIC_MESSAGE,
                    data={
                        "id": msg.id,
                        "sender": sender_name,
                        "content": content,
                        "timestamp": timestamp.isoformat(),
                        "round": round_num
                    }
                )
            )
            
            self.event_log.log_event("PUBLIC_MESSAGE", {
                "id": msg.id,
                "sender": sender_name,
                "content": content,
                "timestamp": timestamp.isoformat(),
                "round": round_num
            })
            
            print(f"    {sender_name}: {content}")
    
    def _get_messages(self) -> list[PublicMessage]:
        """Get all public messages."""
        return self.state.public_messages.copy()
    
    async def _execute_night_action(self, player: Player) -> Optional[NightAction]:
        """Execute a single player's night action based on their original role."""
        role = player.original_role
        agent = self.agents[player.name]
        player_names = [p.name for p in self.players]
        other_players = [n for n in player_names if n != player.name]
        
        # Build context for this player's night action
        context: dict[str, Any] = {
            "other_players": other_players,
            "player_names": player_names,
        }
        
        # Role-specific context
        if role == Role.WEREWOLF:
            # Find other werewolves
            other_werewolves = [p.name for p in self.players 
                              if p.original_role == Role.WEREWOLF and p.name != player.name]
            context["other_werewolves"] = other_werewolves
            
        elif role == Role.MINION:
            # Minion sees who the werewolves are
            werewolves = [p.name for p in self.players if p.original_role == Role.WEREWOLF]
            context["werewolves"] = werewolves
            
        elif role == Role.MASON:
            # Mason sees the other mason (if not in center)
            other_masons = [p.name for p in self.players 
                          if p.original_role == Role.MASON and p.name != player.name]
            context["other_masons"] = other_masons
            
        elif role == Role.INSOMNIAC:
            # Insomniac sees their current card (after all swaps)
            context["current_role"] = player.current_role
        
        # Get pre-determined action from seeded setup
        if player.name in self.game_setup.night_action_plan:
            plan = self.game_setup.night_action_plan[player.name]
            
            # Handle werewolf special case: if not alone, just acknowledge
            if role == Role.WEREWOLF and context.get("other_werewolves"):
                action = NightAction(
                    player_name=player.name,
                    original_role=role,
                    action_type="acknowledge",
                    targets=[]
                )
            else:
                action = NightAction(
                    player_name=player.name,
                    original_role=role,
                    action_type=plan.action_type,
                    targets=list(plan.targets)
                )
        else:
            # Passive role with no night action plan
            action = NightAction(
                player_name=player.name,
                original_role=role,
                action_type="none",
                targets=[]
            )
        
        # Execute the action and get result
        result = await self._apply_night_action(player, action)
        action.result = result
        
        # Build context with results for informational prompt
        context["predetermined_action"] = action
        context["action_result"] = result
        
        # Inform the agent about what happened (informational, no choice)
        await agent.run_night_phase(context)
        
        # Log the night action
        self.event_log.log_event("NIGHT_ACTION", {
            "player": player.name,
            "role": role.value,
            "action": action.action_type,
            "targets": action.targets,
            "result": action.result
        })
        
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.NIGHT_ACTION,
                data={
                    "player": player.name,
                    "role": role.value,
                    "action": action.action_type,
                    "targets": action.targets,
                    "result": action.result
                }
            )
        )
        
        self.state.night_actions.append(action)
        return action
    
    async def _apply_night_action(self, player: Player, action: NightAction) -> Optional[Any]:
        """Apply a night action and return the result."""
        agent = self.agents[player.name]
        result = None
        
        if action.action_type == "look_player":
            # Seer looks at a player
            target_name = action.targets[0]
            target_player = self.state.get_player_by_name(target_name)
            if target_player:
                result = target_player.current_role.value
                
        elif action.action_type == "look_center":
            # Seer looks at two center cards
            positions = [int(t.split("_")[1]) for t in action.targets]
            roles = [self.state.center_cards[p].value for p in positions if 0 <= p < 3]
            result = roles
            
        elif action.action_type == "rob":
            # Robber swaps with target and sees new card
            target_name = action.targets[0]
            target_player = self.state.get_player_by_name(target_name)
            if target_player:
                new_role = target_player.current_role
                self.state.swap_player_cards(player.name, target_name)
                result = new_role.value
                await self._log_card_swap(player.name, target_name)
                
        elif action.action_type == "swap":
            # Troublemaker swaps two other players
            player1_name, player2_name = action.targets
            self.state.swap_player_cards(player1_name, player2_name)
            result = f"Swapped {player1_name} and {player2_name}"
            await self._log_card_swap(player1_name, player2_name)
            
        elif action.action_type == "drunk_swap":
            # Drunk swaps with center card (blind)
            position = int(action.targets[0].split("_")[1])
            if 0 <= position < 3:
                self.state.swap_player_with_center(player.name, position)
                result = f"Swapped with center_{position}"
                await self._log_card_swap(player.name, f"center_{position}")
                
        elif action.action_type == "werewolf_look":
            # Lone werewolf looks at center
            position = int(action.targets[0].split("_")[1])
            if 0 <= position < 3:
                result = self.state.center_cards[position].value
        
        return result
    
    async def _log_card_swap(self, location1: str, location2: str) -> None:
        """Log a card swap event."""
        # Build updated player info
        updated_players = {}
        for p in self.players:
            updated_players[p.name] = {"current_role": p.current_role.value}
        
        self.event_log.log_event("CARD_SWAP", {
            "location1": location1,
            "location2": location2,
            "updated_players": updated_players,
            "center_cards": [r.value for r in self.state.center_cards]
        })
        
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.CARD_SWAP,
                data={
                    "location1": location1,
                    "location2": location2,
                    "updated_players": updated_players,
                    "center_cards": [r.value for r in self.state.center_cards]
                }
            )
        )
    
    async def _run_night_phase(self) -> None:
        """Run the night phase with actions in order."""
        print(f"\n{'='*50}")
        print(f"NIGHT PHASE")
        print(f"{'='*50}")
        
        self.state.phase = Phase.NIGHT
        self.state.phase_start_time = datetime.now()
        
        # Broadcast phase change
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PHASE_CHANGE,
                data={"phase": "NIGHT", "day_number": 1}
            )
        )
        self.event_log.log_event("PHASE_CHANGE", {"phase": "NIGHT", "day_number": 1})
        
        # Emit GM message with initial role assignments (God View)
        role_assignments = "\n".join([f"  {p.name}: {p.original_role.value}" for p in self.players])
        center_cards = ", ".join([r.value for r in self.state.center_cards])
        await self._emit_gm_message(
            f"Roles dealt:\n{role_assignments}\n  Center: [{center_cards}]"
        )
        
        await self._emit_gm_message("Everyone, close your eyes.")
        
        # Execute night actions in order
        for role in NIGHT_ACTION_ORDER:
            # Find players with this original role
            players_with_role = self.state.get_players_with_original_role(role)
            
            if not players_with_role:
                continue
            
            # GM narration
            role_name = role.value.title()
            if len(players_with_role) > 1:
                await self._emit_gm_message(f"{role_name}s, wake up.")
            else:
                await self._emit_gm_message(f"{role_name}, wake up.")
            
            print(f"\n  --- {role.value} PHASE ---")
            self.logger.info(f"  {role.value} phase - {[p.name for p in players_with_role]}")
            
            # Execute each player's action
            for player in players_with_role:
                action = await self._execute_night_action(player)
                if action:
                    print(f"    {player.name}: {action.action_type} -> {action.targets} = {action.result}")
            
            # GM closes eyes
            if len(players_with_role) > 1:
                await self._emit_gm_message(f"{role_name}s, close your eyes.")
            else:
                await self._emit_gm_message(f"{role_name}, close your eyes.")
        
        await self._emit_gm_message("Everyone, wake up!")
        
        # Log final state after all swaps
        print(f"\n  --- FINAL ROLES AFTER NIGHT ---")
        for p in self.players:
            changed = " (CHANGED)" if p.current_role != p.original_role else ""
            print(f"    {p.name}: {p.original_role.value} -> {p.current_role.value}{changed}")
    
    async def _run_day_phase(self) -> None:
        """Run the day discussion phase."""
        print(f"\n{'='*50}")
        print(f"DAY PHASE - Discussion ({self.num_rounds} rounds)")
        print(f"{'='*50}")
        
        self.state.phase = Phase.DAY
        self.state.phase_start_time = datetime.now()
        
        # Update agents
        for agent in self.agents.values():
            agent.set_phase("DAY")
        
        # Broadcast phase change
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PHASE_CHANGE,
                data={
                    "phase": "DAY", 
                    "day_number": 1,
                    "num_rounds": self.num_rounds
                }
            )
        )
        self.event_log.log_event("PHASE_CHANGE", {"phase": "DAY", "day_number": 1, "num_rounds": self.num_rounds})
        
        await self._emit_gm_message(
            f"The sun rises. You have {self.num_rounds} rounds of discussion before voting. All messages in each round are revealed simultaneously."
        )
        
        player_names = [p.name for p in self.players]
        
        # Run each discussion round
        for round_num in range(1, self.num_rounds + 1):
            self.state.current_round = round_num
            print(f"\n--- Round {round_num} of {self.num_rounds} ---")
            
            # Broadcast round start
            await self.broadcaster.broadcast(
                GameEvent(
                    event_type=EventType.ROUND_START,
                    data={"round": round_num, "total_rounds": self.num_rounds}
                )
            )
            self.event_log.log_event("ROUND_START", {"round": round_num, "total_rounds": self.num_rounds})
            
            # Collect all players' messages for this round simultaneously
            messages_so_far = list(self.state.public_messages)
            
            tasks = {
                player.name: asyncio.create_task(
                    self.agents[player.name].run_day_round(
                        player_names,
                        messages_so_far,
                        current_round=round_num,
                        total_rounds=self.num_rounds,
                    )
                )
                for player in self.players
            }
            
            # Wait for all players to respond
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Collect successful messages
            round_messages = []
            for player_name, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    print(f"  [{player_name}] Error: {result}")
                    self.logger.error(f"Player {player_name} error in round {round_num}: {result}")
                elif result is not None:
                    round_messages.append((player_name, result))
                else:
                    print(f"  [{player_name}] Passed")
            
            # Broadcast all messages from this round together
            if round_messages:
                print(f"  Messages this round:")
                await self._broadcast_round_messages(round_num, round_messages)
            else:
                print(f"  (No messages this round)")
            
            # Broadcast round end
            await self.broadcaster.broadcast(
                GameEvent(
                    event_type=EventType.ROUND_END,
                    data={"round": round_num, "message_count": len(round_messages)}
                )
            )
            self.event_log.log_event("ROUND_END", {"round": round_num, "message_count": len(round_messages)})
        
        self.state.current_round = 0
        print(f"\nDay phase ended. {len(self.state.public_messages)} messages were sent over {self.num_rounds} rounds.")
    
    async def _run_voting_phase(self) -> VoteResult:
        """Run the voting phase."""
        print(f"\n{'='*50}")
        print(f"VOTING PHASE")
        print(f"{'='*50}")
        
        self.state.phase = Phase.VOTING
        self.state.phase_start_time = datetime.now()
        self.state.current_votes.clear()
        
        # Update agents
        for agent in self.agents.values():
            agent.set_phase("VOTING")
        
        # Broadcast phase change
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PHASE_CHANGE,
                data={"phase": "VOTING", "day_number": 1}
            )
        )
        self.event_log.log_event("PHASE_CHANGE", {"phase": "VOTING", "day_number": 1})
        
        await self._emit_gm_message("Time to vote! Point at who you think is a Werewolf.")
        
        player_names = [p.name for p in self.players]
        valid_targets = set(player_names)
        
        # Collect votes in parallel
        votes: dict[str, str] = {}
        vote_reasonings: dict[str, str] = {}
        
        async def get_player_vote(player: Player) -> tuple[str, Optional[str], Optional[str]]:
            vote, reasoning = await self.agents[player.name].run_voting_phase(player_names, self.state.public_messages)
            return player.name, vote, reasoning
        
        voting_tasks = [get_player_vote(player) for player in self.players]
        results = await asyncio.gather(*voting_tasks)
        
        for player_name, vote, reasoning in results:
            # Validate vote is a valid player (not self-vote allowed, must be another player)
            if vote and vote in valid_targets and vote != player_name:
                votes[player_name] = vote
                vote_reasonings[player_name] = reasoning or ""
                self.state.current_votes[player_name] = vote
                
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.VOTE_CAST,
                        data={
                            "voter": player_name, 
                            "target": vote,
                            "reasoning": reasoning
                        }
                    )
                )
                self.event_log.log_event("VOTE_REASONING", {
                    "voter": player_name, 
                    "target": vote,
                    "reasoning": reasoning
                })
            else:
                self.logger.warning(f"Invalid vote from {player_name}: '{vote}'")
        
        # Determine winner using the ONUW rules
        winner, killed_players = determine_winner(self.state)
        
        result = VoteResult(
            votes=votes,
            killed_players=killed_players
        )
        self.state.vote_result = result
        
        # Broadcast result
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.VOTE_RESULT,
                data={
                    "votes": votes,
                    "killed": killed_players,
                }
            )
        )
        self.event_log.log_event("VOTE_RESULT", {
            "votes": votes,
            "killed": killed_players,
        })
        
        # Announce results
        print(f"\nVotes: {votes}")
        print(f"Killed: {killed_players}")
        killed_str = ", ".join(killed_players)
        if len(killed_players) == 1:
            await self._emit_gm_message(f"The village points... {killed_str} is killed!")
        else:
            await self._emit_gm_message(f"The village points... it's a tie! {killed_str} are all killed!")
        
        return result
    
    async def run_game(self) -> str:
        """Run the full game."""
        print(f"\n{'#'*60}")
        print(f"ONE NIGHT ULTIMATE WEREWOLF")
        print(f"{'#'*60}")
        
        # Broadcast game start
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.GAME_START,
                data={
                    "players": [
                        {"name": p.name, "original_role": p.original_role.value}
                        for p in self.players
                    ],
                    "center_cards": [r.value for r in self.state.center_cards]
                }
            )
        )
        
        # Night phase
        await self._run_night_phase()
        
        # Day phase
        await self._run_day_phase()
        
        # Voting phase
        await self._run_voting_phase()
        
        # Determine winner
        winner, killed = determine_winner(self.state)
        self.state.winner = winner
        self.state.phase = Phase.GAME_OVER
        
        print(f"\n{'#'*60}")
        print(f"GAME OVER - {winner} WINS!")
        print(f"{'#'*60}")
        
        # Reveal all roles
        print("\nFinal roles:")
        for p in self.players:
            changed = f" (was {p.original_role.value})" if p.current_role != p.original_role else ""
            status = "KILLED" if p.name in killed else "alive"
            print(f"  - {p.name}: {p.current_role.value}{changed} [{status}]")
        print(f"Center: {[r.value for r in self.state.center_cards]}")
        
        # Explain win condition
        print(f"\nWin explanation:")
        werewolves = [p.name for p in self.players if p.current_role == Role.WEREWOLF]
        if werewolves:
            print(f"  Werewolf players: {werewolves}")
            if any(p.current_role == Role.WEREWOLF for p in self.players if p.name in killed):
                print(f"  A werewolf was killed -> VILLAGE wins!")
            else:
                print(f"  No werewolf was killed -> WEREWOLF wins!")
        else:
            print(f"  No werewolves among players (all in center)")
            minion_killed = any(p.current_role == Role.MINION for p in self.players if p.name in killed)
            if minion_killed:
                print(f"  Minion was killed -> WEREWOLF team wins!")
            else:
                print(f"  Minion survived -> VILLAGE wins!")
        
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.GAME_END,
                data={
                    "winner": winner,
                    "players": [
                        {"name": p.name, "original_role": p.original_role.value, "current_role": p.current_role.value}
                        for p in self.players
                    ],
                    "center_cards": [r.value for r in self.state.center_cards],
                    "killed": killed
                }
            )
        )
        
        self.event_log.log_event("GAME_END", {
            "winner": winner,
            "players": [
                {"name": p.name, "original_role": p.original_role.value, "current_role": p.current_role.value}
                for p in self.players
            ],
            "center_cards": [r.value for r in self.state.center_cards],
            "killed": killed,
            "errors": self.errors
        })
        self.event_log.close()
        
        if self.errors:
            print(f"\n⚠️ ERRORS ENCOUNTERED ({len(self.errors)} total)")
        
        return winner
    
    def get_game_state_dict(self) -> dict:
        """Get the current game state as a dictionary (for web viewer)."""
        event_history = self.broadcaster.get_history()
        phase_history = []
        current_phase = None
        
        for event in event_history:
            event_dict = event.to_dict()
            event_type = event_dict.get("event_type", "")
            data = event_dict.get("data", {})
            
            if event_type == "PHASE_CHANGE":
                current_phase = {
                    "phase": data.get("phase"),
                    "dayNumber": 1,
                    "players": [
                        {"name": p.name, "model": p.model, "original_role": p.original_role.value, "current_role": p.current_role.value}
                        for p in self.players
                    ],
                    "centerCards": [r.value for r in self.state.center_cards],
                    "events": [],
                    "messages": [],
                    "votes": {},
                    "gmMessages": []
                }
                phase_history.append(current_phase)
            
            elif current_phase is not None:
                if event_type == "GM_MESSAGE":
                    current_phase["events"].append({**data, "type": "gm_message"})
                    current_phase["gmMessages"].append(data)
                elif event_type == "NIGHT_ACTION":
                    current_phase["events"].append({**data, "type": "night_action"})
                elif event_type == "CARD_SWAP":
                    current_phase["events"].append({**data, "type": "card_swap"})
                    current_phase["players"] = [
                        {"name": p.name, "model": p.model, "original_role": p.original_role.value, "current_role": p.current_role.value}
                        for p in self.players
                    ]
                elif event_type == "PUBLIC_MESSAGE":
                    msg = {**data, "type": "message"}
                    current_phase["events"].append(msg)
                    current_phase["messages"].append(msg)
                elif event_type == "VOTE_CAST":
                    current_phase["events"].append({**data, "type": "vote"})
                    current_phase["votes"][data.get("voter")] = data
                elif event_type == "VOTE_RESULT":
                    current_phase["events"].append({**data, "type": "vote_result"})
        
        if not phase_history:
            phase_history.append({
                "phase": self.state.phase.value,
                "dayNumber": 1,
                "players": [
                    {"name": p.name, "model": p.model, "original_role": p.original_role.value, "current_role": p.current_role.value}
                    for p in self.players
                ],
                "centerCards": [r.value for r in self.state.center_cards],
                "events": [],
                "messages": [],
                "votes": {},
                "gmMessages": []
            })
        
        # Build player thoughts
        player_thoughts: dict[str, dict[str, list]] = {}
        for event in event_history:
            event_dict = event.to_dict()
            if event_dict.get("event_type") == "PLAYER_THOUGHT":
                data = event_dict.get("data", {})
                player_name = data.get("player_name")
                phase = data.get("phase", "UNKNOWN")
                phase_key = f"1_{phase}"
                
                if player_name not in player_thoughts:
                    player_thoughts[player_name] = {}
                if phase_key not in player_thoughts[player_name]:
                    player_thoughts[player_name][phase_key] = []
                
                # If this thought includes the system prompt, prepend it
                # This allows reconstructing the full chat thread by appending all phases
                system_prompt = data.get("system_prompt")
                if system_prompt:
                    player_thoughts[player_name][phase_key].append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                player_thoughts[player_name][phase_key].append({
                    "role": "user",
                    "content": data.get("prompt")
                })
                player_thoughts[player_name][phase_key].append({
                    "role": "assistant",
                    "content": data.get("response"),
                    "tool_calls": data.get("tool_calls")
                })
        
        # Get killed players from vote result
        killed = []
        if self.state.vote_result:
            killed = self.state.vote_result.killed_players
        
        return {
            "phase": self.state.phase.value,
            "day_number": 1,
            "players": [
                {"name": p.name, "model": p.model, "original_role": p.original_role.value, "current_role": p.current_role.value}
                for p in self.players
            ],
            "center_cards": [r.value for r in self.state.center_cards],
            "messages": [
                {
                    "id": m.id,
                    "sender": m.sender_name,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.state.public_messages
            ],
            "current_votes": self.state.current_votes,
            "winner": self.state.winner,
            "killed": killed,
            "phase_history": phase_history,
            "player_thoughts": player_thoughts,
        }
