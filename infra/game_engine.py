"""Game engine with night/day/voting phases and win conditions."""

import asyncio
import random
from datetime import datetime, timedelta
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


class MafiaGame:
    """Main game engine for LLM Mafia."""
    
    def __init__(
        self,
        player_names: list[str] = DEFAULT_PLAYER_NAMES,
        role_distribution: list[Role] = DEFAULT_ROLE_DISTRIBUTION,
        model: str = "gpt-5-nano",
        day_duration_seconds: int = DAY_PHASE_DURATION_SECONDS,
    ):
        assert len(player_names) == len(role_distribution), \
            f"Player count ({len(player_names)}) must match role count ({len(role_distribution)})"
        
        self.model = model
        self.day_duration_seconds = day_duration_seconds
        self.llm_client = get_llm_client(use_cache=True)
        self.broadcaster = get_broadcaster()
        
        # Shuffle roles and assign to players
        shuffled_roles = role_distribution.copy()
        random.shuffle(shuffled_roles)
        
        self.players: list[Player] = [
            Player(name=name, model=model, role=role)
            for name, role in zip(player_names, shuffled_roles)
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
        
        living_names = [p.name for p in self.state.living_players]
        
        # Collect night actions from each player with a special role
        night_actions: dict[Role, Optional[str]] = {}
        
        for player in self.state.living_players:
            if player.role in [Role.MAFIA, Role.DOCTOR, Role.DETECTIVE]:
                target = await self.agents[player.name].run_night_phase(living_names)
                night_actions[player.role] = target
                
                # Broadcast night action to viewers
                await self.broadcaster.broadcast(
                    GameEvent(
                        event_type=EventType.NIGHT_ACTION,
                        data={
                            "role": player.role.value,
                            "player": player.name,
                            "target": target
                        }
                    )
                )
            else:
                # Town members just wait
                await self.agents[player.name].run_night_phase(living_names)
        
        # Resolve night actions
        result = NightResult()
        
        # Get mafia kill target (if multiple mafia, they should coordinate - for now just use the last one)
        mafia_target_name = None
        for player in self.state.living_players:
            if player.role == Role.MAFIA and Role.MAFIA in night_actions:
                mafia_target_name = night_actions.get(Role.MAFIA)
                break
        
        if mafia_target_name:
            result.kill_target = self.state.get_player_by_name(mafia_target_name)
        
        # Check if doctor saved the target
        doctor_target_name = night_actions.get(Role.DOCTOR)
        if doctor_target_name:
            result.saved_player = self.state.get_player_by_name(doctor_target_name)
        
        # Determine if kill succeeds
        if result.kill_target and result.kill_target != result.saved_player:
            result.killed_player = result.kill_target
            result.killed_player.is_alive = False
            
            await self.broadcaster.broadcast(
                GameEvent(
                    event_type=EventType.PLAYER_DEATH,
                    data={
                        "player": result.killed_player.name,
                        "cause": "killed_by_mafia"
                    }
                )
            )
        
        # Handle detective investigation
        detective_target_name = night_actions.get(Role.DETECTIVE)
        if detective_target_name:
            target = self.state.get_player_by_name(detective_target_name)
            if target:
                is_mafia = target.role == Role.MAFIA
                result.investigation_result = (target, is_mafia)
                
                # Tell the detective the result
                for player in self.state.living_players:
                    if player.role == Role.DETECTIVE:
                        result_text = "IS MAFIA" if is_mafia else "is NOT mafia"
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
        
        # Announce night results to all players
        if night_result.killed_player:
            announcement = f"{night_result.killed_player.name} was killed during the night. They were a {night_result.killed_player.role.value}."
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
                                "role": lynched_player.role.value
                            }
                        )
                    )
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
                    "lynched_role": result.lynched_player.role.value if result.lynched_player else None
                }
            )
        )
        
        # Announce result to players
        if result.lynched_player:
            announcement = f"{result.lynched_player.name} was lynched. They were a {result.lynched_player.role.value}."
        elif result.was_tie:
            announcement = "The vote was tied. Nobody was lynched."
        else:
            announcement = "No valid votes were cast. Nobody was lynched."
        
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
        
        while True:
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
                {"name": p.name, "is_alive": p.is_alive}
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
