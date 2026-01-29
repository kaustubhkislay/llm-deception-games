"""Game engine with night/day/voting phases and win conditions."""

import asyncio
import json
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import Counter

from .mafia import (
    Player, Role, Phase, GameState, PublicMessage, 
    NightResult, VoteResult, Action, ActionType,
    GameEvent, EventType, ChatMessage,
    DEFAULT_PLAYER_NAMES, DEFAULT_ROLE_DISTRIBUTION, DAY_PHASE_DURATION_SECONDS
)
from .llm_client import get_llm_client, CachedLLMClient
from .player import PlayerAgent
from .events import get_broadcaster, EventBroadcaster


# Set up file-based logging
LOG_DIR = Path(__file__).parent.parent / "game_logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_game_logger(game_id: str) -> logging.Logger:
    """Set up a logger for a specific game that writes to a file."""
    logger = logging.getLogger(f"mafia_game_{game_id}")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler - writes immediately (no buffering)
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
    """
    Load and reconstruct game state from a JSON lines log file.
    
    Returns a dict compatible with the web viewer's expected format,
    including phase_history for timeline navigation with all events grouped by phase.
    """
    players: dict[str, dict] = {}  # name -> {role, is_alive}
    all_messages: list[dict] = []
    all_votes: dict[str, dict] = {}  # voter -> {target, reasoning}
    phase = "GAME_OVER"
    day_number = 1
    winner: Optional[str] = None
    
    # Track phase history for timeline
    phase_history: list[dict] = []
    
    # Current phase accumulator
    current_phase_events: list[dict] = []
    current_phase_messages: list[dict] = []
    current_phase_votes: dict[str, dict] = {}
    current_phase_gm_messages: list[dict] = []
    
    def snapshot_players():
        return [
            {"name": name, "role": info["role"], "is_alive": info["is_alive"]}
            for name, info in players.items()
        ]
    
    def save_phase_snapshot(phase_name: str, day_num: int):
        phase_history.append({
            "phase": phase_name,
            "dayNumber": day_num,
            "players": snapshot_players(),
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
                for p in data.get("players", []):
                    players[p["name"]] = {"role": p["role"], "is_alive": True}
            
            elif event_type == "PHASE_CHANGE":
                # Save current phase before starting new one (if we have one)
                if phase != "GAME_OVER" and (current_phase_events or current_phase_messages or current_phase_votes):
                    # Don't save empty phases
                    pass
                
                # Start new phase
                phase = data.get("phase", phase)
                day_number = data.get("day_number", day_number)
                
                # Clear accumulators for new phase
                current_phase_events = []
                current_phase_messages = []
                current_phase_votes = {}
                current_phase_gm_messages = []
                
                # Save the phase entry point
                save_phase_snapshot(phase, day_number)
            
            elif event_type == "GM_MESSAGE":
                gm_msg = {
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp", timestamp)
                }
                current_phase_gm_messages.append(gm_msg)
                current_phase_events.append({"type": "gm_message", "data": gm_msg})
                
                # Update the last phase snapshot
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
                    phase_history[-1]["gmMessages"] = list(current_phase_gm_messages)
            
            elif event_type == "NIGHT_REASONING":
                reasoning_data = {
                    "type": "night_reasoning",
                    "role": data.get("role"),
                    "player": data.get("player"),
                    "target": data.get("target"),
                    "reasoning": data.get("reasoning"),
                    "result": data.get("result"),
                    "timestamp": timestamp
                }
                current_phase_events.append(reasoning_data)
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "MAFIA_CHAT":
                msg_data = {
                    "type": "mafia_chat",
                    "sender": data.get("sender"),
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp", timestamp)
                }
                current_phase_events.append(msg_data)
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "MAFIA_KILL_INTENTION":
                intention_data = {
                    "type": "kill_intention",
                    "player": data.get("player"),
                    "target": data.get("target"),
                    "timestamp": timestamp
                }
                current_phase_events.append(intention_data)
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "PLAYER_DEATH":
                player_name = data.get("player")
                if player_name and player_name in players:
                    players[player_name]["is_alive"] = False
                
                death_data = {
                    "type": "death",
                    "player": player_name,
                    "cause": data.get("cause"),
                    "was_mafia": data.get("was_mafia"),
                    "timestamp": timestamp
                }
                current_phase_events.append(death_data)
                
                if phase_history:
                    phase_history[-1]["players"] = snapshot_players()
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "PUBLIC_MESSAGE":
                msg = {
                    "type": "message",
                    "id": data.get("id"),
                    "sender": data.get("sender"),
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp", timestamp)
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
                    "lynched": data.get("lynched"),
                    "was_tie": data.get("was_tie"),
                    "timestamp": timestamp
                }
                current_phase_events.append(result_data)
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "NIGHT_RESULT":
                result_data = {
                    "type": "night_result",
                    "killed": data.get("killed"),
                    "saved": data.get("saved"),
                    "timestamp": timestamp
                }
                current_phase_events.append(result_data)
                
                if phase_history:
                    phase_history[-1]["events"] = list(current_phase_events)
            
            elif event_type == "GAME_END":
                winner = data.get("winner")
                # Update final player states from game end data
                for p in data.get("players", []):
                    if p["name"] in players:
                        players[p["name"]]["is_alive"] = p["is_alive"]
                        players[p["name"]]["role"] = p["role"]
                
                # Add final GAME_OVER phase
                current_phase_events = []
                current_phase_messages = []
                current_phase_votes = {}
                current_phase_gm_messages = []
                save_phase_snapshot("GAME_OVER", day_number)
    
    # Build final state
    player_list = snapshot_players()
    
    return {
        "phase": phase,
        "day_number": day_number,
        "players": player_list,
        "living_players": [p["name"] for p in player_list if p["is_alive"]],
        "dead_players": [{"name": p["name"], "role": p["role"]} for p in player_list if not p["is_alive"]],
        "messages": all_messages,
        "current_votes": {k: v.get("target") for k, v in all_votes.items()},
        "winner": winner,
        "phase_history": phase_history,
    }


class MafiaGame:
    """Main game engine for LLM Mafia."""
    
    def __init__(
        self,
        player_names: list[str] = DEFAULT_PLAYER_NAMES,
        role_distribution: list[Role] = DEFAULT_ROLE_DISTRIBUTION,
        model: str = "gpt-5-nano",
        day_duration_seconds: int = DAY_PHASE_DURATION_SECONDS,
        turn_limit: Optional[int] = None,  # None means no limit
    ):
        assert len(player_names) == len(role_distribution), \
            f"Player count ({len(player_names)}) must match role count ({len(role_distribution)})"
        
        # Set up game logger
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_game_logger(self.game_id)
        self.event_log = GameEventLog(self.game_id)
        
        self.model = model
        self.day_duration_seconds = day_duration_seconds
        self.turn_limit = turn_limit
        self.llm_client = get_llm_client(use_cache=True)
        self.broadcaster = get_broadcaster()
        
        # Sort player names alphabetically for consistent display order
        sorted_names = sorted(player_names)
        
        # Shuffle roles for random assignment
        shuffled_roles = list(role_distribution)
        random.shuffle(shuffled_roles)
        
        # Create players with alphabetical names and random roles
        self.players: list[Player] = [
            Player(name=name, model=model, role=role)
            for name, role in zip(sorted_names, shuffled_roles)
        ]
        
        # Initialize game state
        self.state = GameState(players=self.players)
        
        # Communication infrastructure
        self._message_queues: dict[str, asyncio.Queue] = {
            p.name: asyncio.Queue() for p in self.players
        }
        self._new_message_event = asyncio.Event()
        self._phase_end_event = asyncio.Event()
        
        # Player agents
        self.agents: dict[str, PlayerAgent] = {}
        for player in self.players:
            self.agents[player.name] = PlayerAgent(
                player=player,
                llm_client=self.llm_client,
                message_queue=self._message_queues[player.name],
                send_message_callback=self._send_message,
                new_message_event=self._new_message_event,
            )
        
        # Log game initialization
        self.logger.info(f"Game {self.game_id} initialized with {len(self.players)} players")
        for p in self.players:
            self.logger.info(f"  Player: {p.name} | Role: {p.role.value}")
        
        # JSON event log for replay
        self.event_log.log_event("GAME_INIT", {
            "game_id": self.game_id,
            "players": [{"name": p.name, "role": p.role.value} for p in self.players]
        })
        
        # Error tracking
        self.errors: list[dict] = []  # List of error records
        
        print(f"Game initialized with {len(self.players)} players:")
        for p in self.players:
            print(f"  - {p.name}: {p.role.value}")
    
    def _track_error(self, error_type: str, details: dict) -> None:
        """Track an error for later analysis."""
        error = {
            "type": error_type,
            "timestamp": datetime.now().isoformat(),
            "day_number": self.state.day_number,
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
    
    async def _send_message(self, sender_name: str, content: str) -> None:
        """Callback for players to send messages to the group chat."""
        msg = PublicMessage(
            sender_name=sender_name,
            content=content,
            timestamp=datetime.now()  # Timestamp when message is sent (after inference)
        )
        self.state.public_messages.append(msg)
        
        # Notify all players that a new message arrived
        self._new_message_event.set()
        self._new_message_event.clear()
        
        # Broadcast to web viewers
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PUBLIC_MESSAGE,
                data={
                    "id": msg.id,
                    "sender": sender_name,
                    "content": content,
                    "timestamp": msg.timestamp.isoformat()
                }
            )
        )
        
        # Log for replay
        self.event_log.log_event("PUBLIC_MESSAGE", {
            "id": msg.id,
            "sender": sender_name,
            "content": content,
            "timestamp": msg.timestamp.isoformat()
        })
        
        # Show elapsed time since phase start
        elapsed_str = ""
        if self.state.phase_start_time:
            elapsed = (msg.timestamp - self.state.phase_start_time).total_seconds()
            mins, secs = divmod(int(elapsed), 60)
            elapsed_str = f"{mins}:{secs:02d}"
        else:
            elapsed_str = msg.timestamp.strftime('%H:%M:%S')
        print(f"  [{elapsed_str}] {sender_name}: {content}")
    
    def _get_messages(self) -> list[PublicMessage]:
        """Get all public messages (used by player agents)."""
        return self.state.public_messages.copy()
    
    def _check_win_condition(self) -> Optional[str]:
        """Check if the game has ended. Returns 'MAFIA' or 'TOWN' or None."""
        living_mafia = len(self.state.living_mafia)
        living_town = len(self.state.living_town)
        
        if living_mafia == 0:
            return "TOWN"
        if living_mafia >= living_town:
            return "MAFIA"
        return None
    
    def _update_agents_phase(self, phase: str, day: int) -> None:
        """Update all agents with the current phase info."""
        for agent in self.agents.values():
            agent.set_phase(phase, day)
    
    async def _run_mafia_coordination(self, living_names: list[str]) -> Optional[str]:
        """Run the mafia coordination phase. Returns kill target or None."""
        living_mafia = [p for p in self.state.living_players if p.role == Role.MAFIA]
        
        if not living_mafia:
            return None
        
        if len(living_mafia) == 1:
            # Single mafia - pick a target but still show reasoning
            mafia = living_mafia[0]
            print(f"\n  --- MAFIA COORDINATION (single mafia) ---")
            target, reasoning = await self.agents[mafia.name].run_night_phase(living_names)
            
            # Broadcast the mafia's reasoning as a chat message
            if reasoning:
                print(f"    [MAFIA CHAT] {mafia.name}: {reasoning}")
                self.logger.info(f"    MAFIA CHAT: {mafia.name}: {reasoning}")
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.MAFIA_CHAT,
                        data={
                            "sender": mafia.name,
                            "content": reasoning,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                )
                self.event_log.log_event("MAFIA_CHAT", {
                    "sender": mafia.name,
                    "content": reasoning,
                    "timestamp": datetime.now().isoformat()
                })
            
            if target:
                print(f"    [MAFIA INTENTION] {mafia.name} → {target}")
                self.logger.info(f"  MAFIA {mafia.name} targets: {target}")
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.MAFIA_KILL_INTENTION,
                        data={"player": mafia.name, "target": target}
                    )
                )
                self.event_log.log_event("MAFIA_KILL_INTENTION", {"player": mafia.name, "target": target})
            return target
        
        # Multiple mafia - run coordination phase with integrated voting
        # Minimum 30 seconds for mafia discussion to allow for LLM response times
        discussion_duration = max(30, self.day_duration_seconds // 2)
        print(f"\n  --- MAFIA COORDINATION ({discussion_duration}s) ---")
        self.logger.info(f"  Mafia coordination phase - {len(living_mafia)} mafia members")
        
        # Shared message list for mafia private chat (includes intentions)
        mafia_messages: list[dict] = []
        mafia_message_event = asyncio.Event()
        
        # Track kill intentions (can be updated during discussion)
        kill_intentions: dict[str, str] = {}
        
        async def send_mafia_message(sender: str, content: str) -> None:
            msg = {
                "type": "chat",
                "sender": sender,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            mafia_messages.append(msg)
            mafia_message_event.set()
            mafia_message_event.clear()
            print(f"    [MAFIA CHAT] {sender}: {content}")
            self.logger.info(f"    MAFIA CHAT: {sender}: {content}")
            
            # Broadcast mafia chat to web viewers
            await self.broadcaster.broadcast(
                GameEvent(
                    event_type=EventType.MAFIA_CHAT,
                    data=msg
                )
            )
            self.event_log.log_event("MAFIA_CHAT", msg)
        
        async def set_kill_intention(sender: str, target: str) -> None:
            """Callback when a mafia member sets their kill intention."""
            kill_intentions[sender] = target
            
            # Add intention to messages so other mafia can see it
            intention_msg = {
                "type": "intention",
                "sender": sender,
                "target": target,
                "timestamp": datetime.now().isoformat()
            }
            mafia_messages.append(intention_msg)
            mafia_message_event.set()
            mafia_message_event.clear()
            
            print(f"    [MAFIA INTENTION] {sender} → {target}")
            self.logger.info(f"    MAFIA INTENTION: {sender} → {target}")
            
            # Broadcast intention to web viewers
            await self.broadcaster.broadcast(
                GameEvent(
                    event_type=EventType.MAFIA_KILL_INTENTION,
                    data={"player": sender, "target": target}
                )
            )
            self.event_log.log_event("MAFIA_KILL_INTENTION", {"player": sender, "target": target})
        
        def get_mafia_messages() -> list[dict]:
            return mafia_messages.copy()
        
        # Set up callbacks for each mafia member
        mafia_names = [p.name for p in living_mafia]
        for mafia in living_mafia:
            self.agents[mafia.name].set_mafia_callbacks(
                send_mafia_message, 
                mafia_message_event,
                set_kill_intention
            )
        phase_end_event = asyncio.Event()
        
        # Start all mafia discussion tasks
        discussion_tasks = []
        for mafia in living_mafia:
            other_mafia = [n for n in mafia_names if n != mafia.name]
            task = asyncio.create_task(
                self.agents[mafia.name].run_mafia_discussion(
                    living_names, other_mafia, get_mafia_messages, phase_end_event
                )
            )
            discussion_tasks.append((mafia.name, task))
        
        # Wait for discussion duration
        await asyncio.sleep(discussion_duration)
        phase_end_event.set()
        mafia_message_event.set()  # Wake up any waiting mafia
        
        # Wait for tasks to finish gracefully
        for name, task in discussion_tasks:
            task.cancel()
            try:
                result = await task
                # If task returned an intention, use it (fallback)
                if result and name not in kill_intentions:
                    kill_intentions[name] = result
            except asyncio.CancelledError:
                pass
        
        print(f"  --- MAFIA FINAL INTENTIONS: {kill_intentions} ---")
        
        # Use collected intentions to determine kill target
        mafia_votes = list(kill_intentions.values())
        
        if mafia_votes:
            vote_counts = Counter(mafia_votes)
            max_votes = max(vote_counts.values())
            top_voted = [target for target, count in vote_counts.items() if count == max_votes]
            
            if len(top_voted) == 1:
                final_target = top_voted[0]
                self.logger.info(f"  Mafia final target (votes: {dict(vote_counts)}): {final_target}")
                return final_target
            else:
                # Tie - random selection
                final_target = random.choice(top_voted)
                self.logger.info(f"  Mafia vote tied (votes: {dict(vote_counts)}) - randomly selected: {final_target}")
                return final_target
        
        return None
    
    async def _run_night_phase(self) -> NightResult:
        """Run the night phase."""
        print(f"\n{'='*50}")
        print(f"NIGHT {self.state.day_number}")
        print(f"{'='*50}")
        
        self.state.phase = Phase.NIGHT
        self.state.phase_start_time = datetime.now()
        self.state.current_night_actions.clear()
        
        # Update agents with phase info
        self._update_agents_phase("NIGHT", self.state.day_number)
        
        # Broadcast phase change
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PHASE_CHANGE,
                data={"phase": "NIGHT", "day_number": self.state.day_number}
            )
        )
        
        # Log for replay
        self.event_log.log_event("PHASE_CHANGE", {"phase": "NIGHT", "day_number": self.state.day_number})
        
        living_names = [p.name for p in self.state.living_players]
        self.logger.info(f"Night {self.state.day_number} - Living players: {living_names}")
        
        doctor_target: Optional[str] = None
        detective_target: Optional[str] = None
        detective_result: Optional[bool] = None
        
        # 1. DETECTIVE phase
        await self._emit_gm_message("Night falls over the town. Detective, open your eyes.")
        
        for player in self.state.living_players:
            if player.role == Role.DETECTIVE:
                target, reasoning = await self.agents[player.name].run_night_phase(living_names)
                
                # Validate target
                if target and target not in living_names:
                    self._track_error("INVALID_NIGHT_ACTION", {
                        "player": player.name,
                        "role": "DETECTIVE",
                        "target": target,
                        "valid_targets": living_names,
                        "reason": "Target not alive"
                    })
                    target = None  # Invalidate the target
                
                detective_target = target
                self.logger.info(f"  DETECTIVE {player.name} investigates: {target}")
                
                # Get investigation result immediately
                if target:
                    target_player = self.state.get_player_by_name(target)
                    if target_player:
                        detective_result = target_player.role == Role.MAFIA
                
                # Emit reasoning event
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.NIGHT_REASONING,
                        data={
                            "role": "DETECTIVE", 
                            "player": player.name, 
                            "target": target,
                            "reasoning": reasoning,
                            "result": f"{target} {'IS' if detective_result else 'is NOT'} Mafia" if target else None
                        }
                    )
                )
                self.event_log.log_event("NIGHT_REASONING", {
                    "role": "DETECTIVE", 
                    "player": player.name, 
                    "target": target,
                    "reasoning": reasoning,
                    "result": f"{target} {'IS' if detective_result else 'is NOT'} Mafia" if target else None
                })
                break
        
        # 2. DOCTOR phase
        await self._emit_gm_message("Detective, close your eyes. Doctor, open your eyes.")
        
        for player in self.state.living_players:
            if player.role == Role.DOCTOR:
                target, reasoning = await self.agents[player.name].run_night_phase(living_names)
                
                # Validate target
                if target and target not in living_names:
                    self._track_error("INVALID_NIGHT_ACTION", {
                        "player": player.name,
                        "role": "DOCTOR",
                        "target": target,
                        "valid_targets": living_names,
                        "reason": "Target not alive"
                    })
                    target = None  # Invalidate the target
                
                doctor_target = target
                self.logger.info(f"  DOCTOR {player.name} protects: {target}")
                
                # Emit reasoning event
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.NIGHT_REASONING,
                        data={
                            "role": "DOCTOR", 
                            "player": player.name, 
                            "target": target,
                            "reasoning": reasoning
                        }
                    )
                )
                self.event_log.log_event("NIGHT_REASONING", {
                    "role": "DOCTOR", 
                    "player": player.name, 
                    "target": target,
                    "reasoning": reasoning
                })
                break
        
        # 3. MAFIA phase
        await self._emit_gm_message("Doctor, close your eyes. Mafia, open your eyes.")
        
        # Run mafia coordination
        mafia_target_name = await self._run_mafia_coordination(living_names)
        
        # 4. End of night
        await self._emit_gm_message("Mafia, close your eyes. The sun rises over the town.")
        
        # Town members just wait (no action needed)
        for player in self.state.living_players:
            if player.role == Role.TOWN:
                await self.agents[player.name].run_night_phase(living_names)
        
        # Resolve night actions
        result = NightResult()
        
        # Mafia kill
        if mafia_target_name:
            result.kill_target = self.state.get_player_by_name(mafia_target_name)
            self.logger.info(f"  Mafia final target: {mafia_target_name}")
        
        # Doctor protection
        if doctor_target:
            result.saved_player = self.state.get_player_by_name(doctor_target)
            self.logger.info(f"  Doctor protected: {doctor_target}")
        
        # Determine if kill succeeds
        if result.kill_target:
            if result.saved_player and result.kill_target.name == result.saved_player.name:
                # Doctor saved the target!
                self.logger.info(f"  SAVE! Doctor saved {result.kill_target.name} from mafia!")
                result.killed_player = None
            else:
                # Kill succeeds
                result.killed_player = result.kill_target
                result.killed_player.is_alive = False
                self.logger.info(f"  DEATH! {result.killed_player.name} was killed by mafia")
                
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.PLAYER_DEATH,
                        data={
                            "player": result.killed_player.name,
                            "cause": "killed_by_mafia",
                            "was_mafia": result.killed_player.role == Role.MAFIA
                        }
                    )
                )
                self.event_log.log_event("PLAYER_DEATH", {
                    "player": result.killed_player.name,
                    "cause": "killed_by_mafia",
                    "was_mafia": result.killed_player.role == Role.MAFIA
                })
        
        # Handle detective investigation
        if detective_target:
            target = self.state.get_player_by_name(detective_target)
            if target:
                is_mafia = target.role == Role.MAFIA
                result.investigation_result = (target, is_mafia)
                result_text = "IS MAFIA" if is_mafia else "is NOT mafia"
                self.logger.info(f"  Investigation result: {target.name} {result_text}")
                
                # Tell the detective the result
                for player in self.state.living_players:
                    if player.role == Role.DETECTIVE:
                        self.agents[player.name].add_game_event(
                            f"Your investigation reveals: {target.name} {result_text}."
                        )
        
        # Store result
        self.state.night_results.append(result)
        
        # Emit GM message for night result (sun rise already announced above)
        if result.killed_player:
            alignment = "Mafia" if result.killed_player.role == Role.MAFIA else "not Mafia"
            await self._emit_gm_message(f"{result.killed_player.name} was found dead. They were {alignment}.")
        elif result.kill_target and result.saved_player and result.kill_target.name == result.saved_player.name:
            await self._emit_gm_message("The doctor saved someone during the night. No one died.")
        else:
            await self._emit_gm_message("The night passes peacefully. No one died.")
        
        # Broadcast night result
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.NIGHT_RESULT,
                data={
                    "killed": result.killed_player.name if result.killed_player else None,
                    "saved": result.kill_target.name if result.kill_target and result.kill_target == result.saved_player else None
                }
            )
        )
        
        return result
    
    async def _run_day_phase(self, night_result: NightResult) -> None:
        """Run the day discussion phase."""
        print(f"\n{'='*50}")
        print(f"DAY {self.state.day_number} - Discussion ({self.day_duration_seconds} seconds)")
        print(f"{'='*50}")
        
        self.state.phase = Phase.DAY
        self.state.phase_start_time = datetime.now()
        self.state.phase_end_time = datetime.now() + timedelta(seconds=self.day_duration_seconds)
        self._phase_end_event.clear()
        
        # Update agents with phase info
        self._update_agents_phase("DAY", self.state.day_number)
        
        # Broadcast phase change
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PHASE_CHANGE,
                data={
                    "phase": "DAY", 
                    "day_number": self.state.day_number,
                    "duration_seconds": self.day_duration_seconds
                }
            )
        )
        self.event_log.log_event("PHASE_CHANGE", {"phase": "DAY", "day_number": self.state.day_number})
        
        # Announce night results to all players
        if night_result.killed_player:
            alignment = "Mafia" if night_result.killed_player.role == Role.MAFIA else "not Mafia"
            announcement = f"{night_result.killed_player.name} was killed during the night. They were {alignment}."
        else:
            announcement = "Nobody was killed during the night."
        
        for agent in self.agents.values():
            if agent.player.is_alive:
                agent.add_game_event(announcement)
        
        print(f"\n{announcement}\n")
        
        living_names = [p.name for p in self.state.living_players]
        
        # Start all living players' day phase coroutines
        # Get previous day's votes (if any) to show players
        previous_votes: Optional[dict[str, str]] = None
        if self.state.vote_results:
            previous_votes = self.state.vote_results[-1].votes.copy()
        
        player_tasks = []
        for player in self.state.living_players:
            task = asyncio.create_task(
                self.agents[player.name].run_day_phase(
                    living_names,
                    self._get_messages,
                    self._phase_end_event,
                    previous_votes=previous_votes,
                    phase_start_time=self.state.phase_start_time,
                    phase_duration_seconds=self.day_duration_seconds,
                )
            )
            player_tasks.append(task)
        
        # Wait for the day phase duration
        await asyncio.sleep(self.day_duration_seconds)
        
        # Signal end of day phase
        self._phase_end_event.set()
        self._new_message_event.set()  # Wake up any waiting players
        
        # Wait for all player tasks to finish
        for task in player_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        print(f"\nDay phase ended. {len(self.state.public_messages)} messages were sent.")
    
    async def _run_voting_phase(self) -> VoteResult:
        """Run the voting phase."""
        print(f"\n{'='*50}")
        print(f"VOTING PHASE")
        print(f"{'='*50}")
        
        self.state.phase = Phase.VOTING
        self.state.phase_start_time = datetime.now()
        self.state.current_votes.clear()
        
        # Update agents with phase info
        self._update_agents_phase("VOTING", self.state.day_number)
        
        # Broadcast phase change
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PHASE_CHANGE,
                data={"phase": "VOTING", "day_number": self.state.day_number}
            )
        )
        self.event_log.log_event("PHASE_CHANGE", {"phase": "VOTING", "day_number": self.state.day_number})
        
        # Announce voting phase to all players
        for agent in self.agents.values():
            if agent.player.is_alive:
                agent.add_game_event("It is now time to vote. Choose a player to lynch.")
        
        living_names = [p.name for p in self.state.living_players]
        
        # Emit GM message for voting
        await self._emit_gm_message("The town gathers to vote. Each player will cast their vote.")
        
        # Valid vote targets: living players + "no_lynch"
        valid_targets = set(living_names + ["no_lynch"])
        
        # Collect votes from all living players (self-votes are allowed)
        votes: dict[str, str] = {}
        vote_reasonings: dict[str, str] = {}
        invalid_votes: list[dict] = []
        
        for player in self.state.living_players:
            vote, reasoning = await self.agents[player.name].run_voting_phase(living_names, self.state.public_messages)
            
            # Normalize vote (case-insensitive no_lynch)
            normalized_vote = vote.lower() if vote and vote.lower() == "no_lynch" else vote
            if normalized_vote and normalized_vote.lower() == "no_lynch":
                normalized_vote = "no_lynch"
            
            if normalized_vote and normalized_vote in valid_targets:
                votes[player.name] = normalized_vote
                vote_reasonings[player.name] = reasoning or ""
                self.state.current_votes[player.name] = normalized_vote
                
                # Emit vote with reasoning
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.VOTE_REASONING,
                        data={
                            "voter": player.name, 
                            "target": normalized_vote,
                            "reasoning": reasoning
                        }
                    )
                )
                self.event_log.log_event("VOTE_REASONING", {
                    "voter": player.name, 
                    "target": normalized_vote,
                    "reasoning": reasoning
                })
            else:
                # Track invalid votes for debugging
                invalid_votes.append({
                    "voter": player.name,
                    "attempted_target": vote,
                    "reason": "Target not in valid options"
                })
                self.logger.warning(f"Invalid vote from {player.name}: '{vote}' (not in {valid_targets})")
                self.event_log.log_event("INVALID_VOTE", {
                    "voter": player.name,
                    "attempted_target": vote,
                    "valid_options": list(valid_targets)
                })
        
        # Tally votes - require strict majority to lynch
        vote_counts = Counter(votes.values())
        result = VoteResult(votes=votes)
        num_voters = len(self.state.living_players)
        majority_threshold = (num_voters // 2) + 1  # Strict majority
        
        if vote_counts:
            max_votes = max(vote_counts.values())
            top_voted = [name for name, count in vote_counts.items() if count == max_votes]
            
            # Check if we have a strict majority
            if max_votes >= majority_threshold and len(top_voted) == 1:
                lynched_name = top_voted[0]
                
                # Check if "no_lynch" won
                if lynched_name == "no_lynch":
                    # No lynch - town voted to skip
                    self.logger.info(f"Town voted to skip lynching ({max_votes}/{num_voters} votes)")
                else:
                    # Clear winner with majority - lynch them
                    lynched_player = self.state.get_player_by_name(lynched_name)
                    if lynched_player:
                        result.lynched_player = lynched_player
                        lynched_player.is_alive = False
                        
                        await self.broadcaster.broadcast(
                            GameEvent(
                                event_type=EventType.PLAYER_DEATH,
                                data={
                                    "player": lynched_player.name,
                                    "cause": "lynched",
                                    "was_mafia": lynched_player.role == Role.MAFIA
                                }
                            )
                        )
                        self.event_log.log_event("PLAYER_DEATH", {
                            "player": lynched_player.name,
                            "cause": "lynched",
                            "was_mafia": lynched_player.role == Role.MAFIA
                        })
            else:
                # No majority or tie - no lynch
                result.was_tie = len(top_voted) > 1
                if max_votes < majority_threshold:
                    self.logger.info(f"No majority reached ({max_votes}/{num_voters}, need {majority_threshold}). No lynch.")
        
        # Store result
        self.state.vote_results.append(result)
        
        # Broadcast vote result
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.VOTE_RESULT,
                data={
                    "votes": votes,
                    "lynched": result.lynched_player.name if result.lynched_player else None,
                    "was_tie": result.was_tie,
                    "lynched_was_mafia": result.lynched_player.role == Role.MAFIA if result.lynched_player else None
                }
            )
        )
        
        # Announce result via GM message and to players
        if result.lynched_player:
            alignment = "Mafia" if result.lynched_player.role == Role.MAFIA else "not Mafia"
            announcement = f"{result.lynched_player.name} was lynched. They were {alignment}."
        elif result.was_tie:
            announcement = "The vote was tied. Nobody was lynched."
        elif votes and vote_counts.get("no_lynch", 0) > 0:
            # Check if no_lynch had votes but didn't reach majority
            no_lynch_votes = vote_counts.get("no_lynch", 0)
            announcement = f"No majority reached. {no_lynch_votes} voted no_lynch. Nobody was lynched."
        else:
            announcement = "No majority reached. Nobody was lynched."
        
        await self._emit_gm_message(f"The votes are tallied. {announcement}")
        
        # Log voting results
        self.logger.info(f"Voting results: {votes}")
        self.event_log.log_event("VOTE_RESULT", {
            "votes": votes,
            "lynched": result.lynched_player.name if result.lynched_player else None,
            "was_tie": result.was_tie
        })
        if result.lynched_player:
            self.logger.info(f"  LYNCHED: {result.lynched_player.name} ({result.lynched_player.role.value})")
        elif result.was_tie:
            self.logger.info("  TIE - no lynch")
        else:
            self.logger.info("  NO VALID VOTES - no lynch")
        
        print(f"\n{announcement}")
        print(f"Votes: {votes}")
        
        for agent in self.agents.values():
            if agent.player.is_alive:
                agent.add_game_event(announcement)
        
        return result
    
    async def run_game(self) -> str:
        """Run the full game until someone wins."""
        print(f"\n{'#'*60}")
        print(f"GAME START")
        print(f"{'#'*60}")
        
        # Broadcast game start
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.GAME_START,
                data={
                    "players": [
                        {"name": p.name, "is_alive": p.is_alive}
                        for p in self.players
                    ]
                }
            )
        )
        
        # Tell mafia who their partners are
        mafia_members = [p for p in self.players if p.role == Role.MAFIA]
        if len(mafia_members) > 1:
            mafia_names = [p.name for p in mafia_members]
            for player in mafia_members:
                other_mafia = [n for n in mafia_names if n != player.name]
                self.agents[player.name].add_game_event(
                    f"Your fellow mafia member(s): {', '.join(other_mafia)}"
                )
        
        turn_count = 0
        while True:
            turn_count += 1
            self.logger.info(f"=== TURN {turn_count} (Day {self.state.day_number}) ===")
            
            # Night phase
            night_result = await self._run_night_phase()
            
            # Check win condition after night
            winner = self._check_win_condition()
            if winner:
                break
            
            # Day phase
            await self._run_day_phase(night_result)
            
            # Voting phase
            await self._run_voting_phase()
            
            # Check win condition after voting
            winner = self._check_win_condition()
            if winner:
                break
            
            # Check turn limit
            if self.turn_limit and turn_count >= self.turn_limit:
                self.logger.info(f"Turn limit ({self.turn_limit}) reached, stopping game")
                winner = "TURN_LIMIT"
                break
            
            # Increment day counter
            self.state.day_number += 1
        
        # Game over
        self.state.phase = Phase.GAME_OVER
        self.state.winner = winner  # type: ignore
        
        print(f"\n{'#'*60}")
        print(f"GAME OVER - {winner} WINS!")
        print(f"{'#'*60}")
        
        # Reveal all roles
        print("\nFinal roles:")
        for p in self.players:
            status = "alive" if p.is_alive else "dead"
            print(f"  - {p.name}: {p.role.value} ({status})")
        
        await self.broadcaster.broadcast(
            GameEvent(
                event_type=EventType.GAME_END,
                data={
                    "winner": winner,
                    "players": [
                        {"name": p.name, "role": p.role.value, "is_alive": p.is_alive}
                        for p in self.players
                    ]
                }
            )
        )
        
        # Log game end for replay
        self.event_log.log_event("GAME_END", {
            "winner": winner,
            "players": [
                {"name": p.name, "role": p.role.value, "is_alive": p.is_alive}
                for p in self.players
            ],
            "errors": self.errors
        })
        self.event_log.close()
        
        # Print error summary
        if self.errors:
            print(f"\n⚠️ ERRORS ENCOUNTERED ({len(self.errors)} total):")
            by_type: dict[str, int] = {}
            for err in self.errors:
                err_type = err.get("type", "UNKNOWN")
                by_type[err_type] = by_type.get(err_type, 0) + 1
            for err_type, count in sorted(by_type.items()):
                print(f"   {err_type}: {count}")
        
        return winner
    
    def get_player_thoughts(self, player_name: str) -> list[ChatMessage]:
        """Get a player's full chat history (for web viewer)."""
        if player_name in self.agents:
            return self.agents[player_name].player.chat_history
        return []
    
    def get_game_state_dict(self) -> dict:
        """Get the current game state as a dictionary (for web viewer)."""
        # Get event history from broadcaster
        event_history = self.broadcaster.get_history()
        
        # Build phase history with actual events
        phase_history = []
        current_phase = None
        
        for event in event_history:
            event_dict = event.to_dict()
            event_type = event_dict.get("event_type", "")
            data = event_dict.get("data", {})
            
            if event_type == "PHASE_CHANGE":
                # Start a new phase
                current_phase = {
                    "phase": data.get("phase"),
                    "dayNumber": data.get("day_number"),
                    "players": [
                        {"name": p.name, "role": p.role.value, "is_alive": p.is_alive}
                        for p in self.players
                    ],
                    "events": [],
                    "messages": [],
                    "votes": {},
                    "gmMessages": []
                }
                phase_history.append(current_phase)
            
            elif current_phase is not None:
                # Add event to current phase
                # Note: spread data first, then set type to avoid data['type'] overwriting
                if event_type == "GM_MESSAGE":
                    current_phase["events"].append({**data, "type": "gm_message"})
                elif event_type == "NIGHT_REASONING":
                    current_phase["events"].append({**data, "type": "night_reasoning"})
                elif event_type == "MAFIA_CHAT":
                    current_phase["events"].append({**data, "type": "mafia_chat"})
                elif event_type == "MAFIA_KILL_INTENTION":
                    current_phase["events"].append({**data, "type": "kill_intention"})
                elif event_type == "PUBLIC_MESSAGE":
                    msg = {**data, "type": "message"}
                    current_phase["events"].append(msg)
                    current_phase["messages"].append(msg)
                elif event_type == "VOTE_REASONING":
                    current_phase["events"].append({**data, "type": "vote"})
                    current_phase["votes"][data.get("voter")] = data
                elif event_type == "PLAYER_DEATH":
                    current_phase["events"].append({**data, "type": "death"})
                elif event_type == "NIGHT_RESULT":
                    current_phase["events"].append({**data, "type": "night_result"})
                elif event_type == "VOTE_RESULT":
                    current_phase["events"].append({**data, "type": "vote_result"})
        
        # If no phases from events, create one for current state
        if not phase_history:
            phase_history.append({
                "phase": self.state.phase.value,
                "dayNumber": self.state.day_number,
                "players": [
                    {"name": p.name, "role": p.role.value, "is_alive": p.is_alive}
                    for p in self.players
                ],
                "events": [],
                "messages": [],
                "votes": {},
                "gmMessages": []
            })
        
        # Build player thoughts grouped by phase from PLAYER_THOUGHT events
        player_thoughts: dict[str, dict[str, list]] = {}  # {player_name: {phase_key: [thoughts]}}
        for event in event_history:
            event_dict = event.to_dict()
            if event_dict.get("event_type") == "PLAYER_THOUGHT":
                data = event_dict.get("data", {})
                player_name = data.get("player_name")
                phase = data.get("phase", "UNKNOWN")
                day = data.get("day_number", 1)
                phase_key = f"{day}_{phase}"
                
                if player_name not in player_thoughts:
                    player_thoughts[player_name] = {}
                if phase_key not in player_thoughts[player_name]:
                    player_thoughts[player_name][phase_key] = []
                
                player_thoughts[player_name][phase_key].append({
                    "role": "user",
                    "content": data.get("prompt")
                })
                player_thoughts[player_name][phase_key].append({
                    "role": "assistant",
                    "content": data.get("response"),
                    "tool_calls": data.get("tool_calls")
                })
        
        return {
            "phase": self.state.phase.value,
            "day_number": self.state.day_number,
            "players": [
                {"name": p.name, "role": p.role.value, "is_alive": p.is_alive}
                for p in self.players
            ],
            "living_players": [p.name for p in self.state.living_players],
            "dead_players": [
                {"name": p.name, "role": p.role.value}
                for p in self.state.dead_players
            ],
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
            "phase_history": phase_history,
            "player_thoughts": player_thoughts,
        }
