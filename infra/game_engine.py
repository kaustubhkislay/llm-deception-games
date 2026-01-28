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
    
    Returns a dict compatible with the web viewer's expected format.
    """
    players: dict[str, dict] = {}  # name -> {role, is_alive}
    messages: list[dict] = []
    votes: dict[str, str] = {}
    night_actions: list[dict] = []
    phase = "GAME_OVER"
    day_number = 1
    winner: Optional[str] = None
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            event = json.loads(line)
            event_type = event.get("event_type", "")
            data = event.get("data", {})
            
            if event_type == "GAME_INIT":
                for p in data.get("players", []):
                    players[p["name"]] = {"role": p["role"], "is_alive": True}
            
            elif event_type == "PHASE_CHANGE":
                phase = data.get("phase", phase)
                day_number = data.get("day_number", day_number)
            
            elif event_type == "NIGHT_ACTION":
                night_actions.append(data)
            
            elif event_type == "PLAYER_DEATH":
                player_name = data.get("player")
                if player_name and player_name in players:
                    players[player_name]["is_alive"] = False
            
            elif event_type == "PUBLIC_MESSAGE":
                messages.append({
                    "id": data.get("id"),
                    "sender": data.get("sender"),
                    "content": data.get("content"),
                    "timestamp": data.get("timestamp")
                })
            
            elif event_type == "VOTE_CAST":
                voter = data.get("voter")
                target = data.get("target")
                if voter and target:
                    votes[voter] = target
            
            elif event_type == "VOTE_RESULT":
                votes = data.get("votes", votes)
            
            elif event_type == "GAME_END":
                winner = data.get("winner")
                phase = "GAME_OVER"
    
    # Build final state
    player_list = [
        {"name": name, "role": info["role"], "is_alive": info["is_alive"]}
        for name, info in players.items()
    ]
    
    return {
        "phase": phase,
        "day_number": day_number,
        "players": player_list,
        "living_players": [p["name"] for p in player_list if p["is_alive"]],
        "dead_players": [{"name": p["name"], "role": p["role"]} for p in player_list if not p["is_alive"]],
        "messages": messages,
        "current_votes": votes,
        "winner": winner,
        "night_actions": night_actions,
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
        
        # Create player-role pairs and shuffle them together for true randomness
        pairs = list(zip(player_names, role_distribution))
        random.shuffle(pairs)
        
        self.players: list[Player] = [
            Player(name=name, model=model, role=role)
            for name, role in pairs
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
        
        print(f"Game initialized with {len(self.players)} players:")
        for p in self.players:
            print(f"  - {p.name}: {p.role.value}")
    
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
        
        print(f"  [{msg.timestamp.strftime('%H:%M:%S')}] {sender_name}: {content}")
    
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
    
    async def _run_night_phase(self) -> NightResult:
        """Run the night phase."""
        print(f"\n{'='*50}")
        print(f"NIGHT {self.state.day_number}")
        print(f"{'='*50}")
        
        self.state.phase = Phase.NIGHT
        self.state.phase_start_time = datetime.now()
        self.state.current_night_actions.clear()
        
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
        
        # Collect night actions from each player
        # Use lists for roles that may have multiple players (mafia)
        mafia_targets: list[str] = []
        doctor_target: Optional[str] = None
        detective_target: Optional[str] = None
        
        for player in self.state.living_players:
            if player.role == Role.MAFIA:
                target = await self.agents[player.name].run_night_phase(living_names)
                if target:
                    mafia_targets.append(target)
                self.logger.info(f"  MAFIA {player.name} targets: {target}")
                
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.NIGHT_ACTION,
                        data={"role": "MAFIA", "player": player.name, "target": target}
                    )
                )
                self.event_log.log_event("NIGHT_ACTION", {"role": "MAFIA", "player": player.name, "target": target})
                
            elif player.role == Role.DOCTOR:
                target = await self.agents[player.name].run_night_phase(living_names)
                doctor_target = target
                self.logger.info(f"  DOCTOR {player.name} protects: {target}")
                
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.NIGHT_ACTION,
                        data={"role": "DOCTOR", "player": player.name, "target": target}
                    )
                )
                self.event_log.log_event("NIGHT_ACTION", {"role": "DOCTOR", "player": player.name, "target": target})
                
            elif player.role == Role.DETECTIVE:
                target = await self.agents[player.name].run_night_phase(living_names)
                detective_target = target
                self.logger.info(f"  DETECTIVE {player.name} investigates: {target}")
                
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.NIGHT_ACTION,
                        data={"role": "DETECTIVE", "player": player.name, "target": target}
                    )
                )
                self.event_log.log_event("NIGHT_ACTION", {"role": "DETECTIVE", "player": player.name, "target": target})
            else:
                # Town members just wait
                await self.agents[player.name].run_night_phase(living_names)
        
        # Resolve night actions
        result = NightResult()
        
        # Mafia kill: use majority vote, or first target if no majority
        if mafia_targets:
            target_counts = Counter(mafia_targets)
            mafia_target_name = target_counts.most_common(1)[0][0]
            result.kill_target = self.state.get_player_by_name(mafia_target_name)
            self.logger.info(f"  Mafia final target (from {mafia_targets}): {mafia_target_name}")
        
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
        player_tasks = []
        for player in self.state.living_players:
            task = asyncio.create_task(
                self.agents[player.name].run_day_phase(
                    living_names,
                    self._get_messages,
                    self._phase_end_event,
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
        
        # Collect votes from all living players
        votes: dict[str, str] = {}
        for player in self.state.living_players:
            vote = await self.agents[player.name].run_voting_phase(living_names)
            if vote and vote in living_names and vote != player.name:
                votes[player.name] = vote
                self.state.current_votes[player.name] = vote
                
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.VOTE_CAST,
                        data={"voter": player.name, "target": vote}
                    )
                )
                self.event_log.log_event("VOTE_CAST", {"voter": player.name, "target": vote})
        
        # Tally votes
        vote_counts = Counter(votes.values())
        result = VoteResult(votes=votes)
        
        if vote_counts:
            max_votes = max(vote_counts.values())
            top_voted = [name for name, count in vote_counts.items() if count == max_votes]
            
            if len(top_voted) == 1:
                # Clear winner - lynch them
                lynched_name = top_voted[0]
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
                # Tie - no lynch
                result.was_tie = True
        
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
        
        # Announce result to players
        if result.lynched_player:
            alignment = "Mafia" if result.lynched_player.role == Role.MAFIA else "not Mafia"
            announcement = f"{result.lynched_player.name} was lynched. They were {alignment}."
        elif result.was_tie:
            announcement = "The vote was tied. Nobody was lynched."
        else:
            announcement = "No valid votes were cast. Nobody was lynched."
        
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
            ]
        })
        self.event_log.close()
        
        return winner
    
    def get_player_thoughts(self, player_name: str) -> list[ChatMessage]:
        """Get a player's full chat history (for web viewer)."""
        if player_name in self.agents:
            return self.agents[player_name].player.chat_history
        return []
    
    def get_game_state_dict(self) -> dict:
        """Get the current game state as a dictionary (for web viewer)."""
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
        }
