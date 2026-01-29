#!/usr/bin/env python3
"""Run multiple Mafia games in parallel and analyze win rates."""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.game_engine import MafiaGame
from infra.llm_client import get_llm_client


@dataclass
class GameResult:
    """Result from a single game."""
    game_id: str
    winner: str  # "TOWN", "MAFIA", or "TURN_LIMIT"
    turns: int
    mafia_killed: int
    town_killed: int
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float


@dataclass 
class BatchResults:
    """Aggregated results from multiple games."""
    games: list[GameResult] = field(default_factory=list)
    
    @property
    def total_games(self) -> int:
        return len(self.games)
    
    @property
    def town_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "TOWN")
    
    @property
    def mafia_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "MAFIA")
    
    @property
    def incomplete(self) -> int:
        return sum(1 for g in self.games if g.winner == "TURN_LIMIT")
    
    def summary(self) -> dict:
        if not self.games:
            return {}
        
        completed = [g for g in self.games if g.winner != "TURN_LIMIT"]
        
        return {
            "total_games": self.total_games,
            "town_wins": self.town_wins,
            "mafia_wins": self.mafia_wins,
            "incomplete": self.incomplete,
            "town_win_rate": self.town_wins / len(completed) if completed else 0,
            "mafia_win_rate": self.mafia_wins / len(completed) if completed else 0,
            "avg_turns": sum(g.turns for g in self.games) / len(self.games),
            "avg_duration_seconds": sum(g.duration_seconds for g in self.games) / len(self.games),
            "total_prompt_tokens": sum(g.prompt_tokens for g in self.games),
            "total_completion_tokens": sum(g.completion_tokens for g in self.games),
            "total_tokens": sum(g.prompt_tokens + g.completion_tokens for g in self.games),
        }


class QuietMafiaGame(MafiaGame):
    """MafiaGame variant that suppresses most output for batch runs."""
    
    def __init__(self, *args, quiet: bool = True, **kwargs):
        self._quiet = quiet
        super().__init__(*args, **kwargs)
    
    async def _send_message(self, sender_name: str, content: str) -> None:
        """Override to suppress message printing."""
        from infra.mafia import PublicMessage
        from datetime import datetime
        
        msg = PublicMessage(
            sender_name=sender_name,
            content=content,
            timestamp=datetime.now()
        )
        self.state.public_messages.append(msg)
        self._new_message_event.set()
        self._new_message_event.clear()
        
        # Log for replay (still useful)
        self.event_log.log_event("PUBLIC_MESSAGE", {
            "id": msg.id,
            "sender": sender_name,
            "content": content,
            "timestamp": msg.timestamp.isoformat()
        })
    
    async def run_game(self) -> str:
        """Run game with minimal output."""
        # Skip the banner and just run
        from infra.mafia import Role, Phase
        
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
            
            winner = self._check_win_condition()
            if winner:
                break
            
            # Day phase  
            await self._run_day_phase(night_result)
            
            # Voting phase
            await self._run_voting_phase()
            
            winner = self._check_win_condition()
            if winner:
                break
            
            if self.turn_limit and turn_count >= self.turn_limit:
                self.logger.info(f"Turn limit ({self.turn_limit}) reached")
                winner = "TURN_LIMIT"
                break
            
            self.state.day_number += 1
        
        self.state.phase = Phase.GAME_OVER
        self.state.winner = winner  # type: ignore
        
        # Log game end
        self.event_log.log_event("GAME_END", {
            "winner": winner,
            "players": [
                {"name": p.name, "role": p.role.value, "is_alive": p.is_alive}
                for p in self.players
            ]
        })
        self.event_log.close()
        
        return winner
    
    async def _run_night_phase(self):
        """Quiet night phase."""
        self.state.phase = self.state.phase.__class__.NIGHT
        self.state.phase_start_time = datetime.now()
        self.state.current_night_actions.clear()
        
        self.event_log.log_event("PHASE_CHANGE", {"phase": "NIGHT", "day_number": self.state.day_number})
        
        living_names = [p.name for p in self.state.living_players]
        self.logger.info(f"Night {self.state.day_number} - Living: {living_names}")
        
        # Run mafia coordination
        mafia_target_name = await self._run_mafia_coordination(living_names)
        
        # Collect other night actions
        from infra.mafia import Role, NightResult
        doctor_target = None
        detective_target = None
        
        for player in self.state.living_players:
            if player.role == Role.DOCTOR:
                target, reasoning = await self.agents[player.name].run_night_phase(living_names)
                doctor_target = target
                self.logger.info(f"  DOCTOR protects: {target}")
                self.event_log.log_event("NIGHT_REASONING", {"role": "DOCTOR", "player": player.name, "target": target, "reasoning": reasoning})
                
            elif player.role == Role.DETECTIVE:
                target, reasoning = await self.agents[player.name].run_night_phase(living_names)
                detective_target = target
                self.logger.info(f"  DETECTIVE investigates: {target}")
                self.event_log.log_event("NIGHT_REASONING", {"role": "DETECTIVE", "player": player.name, "target": target, "reasoning": reasoning})
                
            elif player.role == Role.TOWN:
                await self.agents[player.name].run_night_phase(living_names)
        
        # Resolve night actions
        result = NightResult()
        
        if mafia_target_name:
            result.kill_target = self.state.get_player_by_name(mafia_target_name)
            self.logger.info(f"  Mafia targets: {mafia_target_name}")
        
        if doctor_target:
            result.saved_player = self.state.get_player_by_name(doctor_target)
            self.logger.info(f"  Doctor protects: {doctor_target}")
        
        # Determine if kill succeeds
        if result.kill_target:
            if result.saved_player and result.kill_target.name == result.saved_player.name:
                self.logger.info(f"  SAVE! {result.kill_target.name}")
                result.killed_player = None
            else:
                result.killed_player = result.kill_target
                result.killed_player.is_alive = False
                self.logger.info(f"  DEATH! {result.killed_player.name}")
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
                self.logger.info(f"  Investigation: {target.name} {result_text}")
                
                for player in self.state.living_players:
                    if player.role == Role.DETECTIVE:
                        self.agents[player.name].add_game_event(
                            f"Your investigation reveals: {target.name} {result_text}."
                        )
        
        self.state.night_results.append(result)
        return result
    
    async def _run_day_phase(self, night_result):
        """Quiet day phase."""
        from infra.mafia import Phase
        from datetime import timedelta
        
        self.state.phase = Phase.DAY
        self.state.phase_start_time = datetime.now()
        self.state.phase_end_time = datetime.now() + timedelta(seconds=self.day_duration_seconds)
        self._phase_end_event.clear()
        
        self.event_log.log_event("PHASE_CHANGE", {"phase": "DAY", "day_number": self.state.day_number})
        
        # Announce night results
        from infra.mafia import Role
        if night_result.killed_player:
            alignment = "Mafia" if night_result.killed_player.role == Role.MAFIA else "not Mafia"
            announcement = f"{night_result.killed_player.name} was killed. They were {alignment}."
        else:
            announcement = "Nobody was killed during the night."
        
        for agent in self.agents.values():
            if agent.player.is_alive:
                agent.add_game_event(announcement)
        
        living_names = [p.name for p in self.state.living_players]
        
        # Start player tasks
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
        
        await asyncio.sleep(self.day_duration_seconds)
        
        self._phase_end_event.set()
        self._new_message_event.set()
        
        for task in player_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _run_voting_phase(self):
        """Quiet voting phase."""
        from infra.mafia import Phase, VoteResult, Role
        from collections import Counter
        
        self.state.phase = Phase.VOTING
        self.state.phase_start_time = datetime.now()
        self.state.current_votes.clear()
        
        self.event_log.log_event("PHASE_CHANGE", {"phase": "VOTING", "day_number": self.state.day_number})
        
        for agent in self.agents.values():
            if agent.player.is_alive:
                agent.add_game_event("It is now time to vote.")
        
        living_names = [p.name for p in self.state.living_players]
        
        votes = {}
        for player in self.state.living_players:
            vote, reasoning = await self.agents[player.name].run_voting_phase(living_names, self.state.public_messages)
            if vote and vote in living_names:
                votes[player.name] = vote
                self.state.current_votes[player.name] = vote
                self.event_log.log_event("VOTE_CAST", {"voter": player.name, "target": vote})
        
        vote_counts = Counter(votes.values())
        result = VoteResult(votes=votes)
        
        if vote_counts:
            max_votes = max(vote_counts.values())
            top_voted = [name for name, count in vote_counts.items() if count == max_votes]
            
            if len(top_voted) == 1:
                lynched_name = top_voted[0]
                lynched_player = self.state.get_player_by_name(lynched_name)
                if lynched_player:
                    result.lynched_player = lynched_player
                    lynched_player.is_alive = False
                    self.event_log.log_event("PLAYER_DEATH", {
                        "player": lynched_player.name,
                        "cause": "lynched",
                        "was_mafia": lynched_player.role == Role.MAFIA
                    })
                    self.logger.info(f"  LYNCHED: {lynched_player.name} ({lynched_player.role.value})")
            else:
                result.was_tie = True
                self.logger.info("  VOTE TIE - no lynch")
        
        self.state.vote_results.append(result)
        
        # Announce
        if result.lynched_player:
            alignment = "Mafia" if result.lynched_player.role == Role.MAFIA else "not Mafia"
            announcement = f"{result.lynched_player.name} was lynched. They were {alignment}."
        elif result.was_tie:
            announcement = "The vote was tied. Nobody was lynched."
        else:
            announcement = "No valid votes. Nobody was lynched."
        
        self.event_log.log_event("VOTE_RESULT", {
            "votes": votes,
            "lynched": result.lynched_player.name if result.lynched_player else None,
            "was_tie": result.was_tie
        })
        
        for agent in self.agents.values():
            if agent.player.is_alive:
                agent.add_game_event(announcement)
        
        return result
    
    async def _run_mafia_coordination(self, living_names):
        """Quiet mafia coordination."""
        from infra.mafia import Role
        from collections import Counter
        
        living_mafia = [p for p in self.state.living_players if p.role == Role.MAFIA]
        
        if not living_mafia:
            return None
        
        if len(living_mafia) == 1:
            mafia = living_mafia[0]
            target, reasoning = await self.agents[mafia.name].run_night_phase(living_names)
            if target:
                self.logger.info(f"  MAFIA {mafia.name} targets: {target}")
                self.event_log.log_event("NIGHT_REASONING", {"role": "MAFIA", "player": mafia.name, "target": target, "reasoning": reasoning})
            return target
        
        # Multiple mafia
        self.logger.info(f"  Mafia coordination - {len(living_mafia)} members")
        
        mafia_messages = []
        mafia_message_event = asyncio.Event()
        
        async def send_mafia_message(sender, content):
            mafia_messages.append({"sender": sender, "content": content, "timestamp": datetime.now().isoformat()})
            mafia_message_event.set()
            mafia_message_event.clear()
            self.logger.info(f"    MAFIA CHAT: {sender}: {content}")
        
        def get_mafia_messages():
            return mafia_messages.copy()
        
        mafia_names = [p.name for p in living_mafia]
        for mafia in living_mafia:
            self.agents[mafia.name].set_mafia_callbacks(send_mafia_message, mafia_message_event)
        
        discussion_duration = self.day_duration_seconds // 2
        phase_end_event = asyncio.Event()
        
        discussion_tasks = []
        for mafia in living_mafia:
            other_mafia = [n for n in mafia_names if n != mafia.name]
            task = asyncio.create_task(
                self.agents[mafia.name].run_mafia_discussion(
                    living_names, other_mafia, get_mafia_messages, phase_end_event
                )
            )
            discussion_tasks.append(task)
        
        await asyncio.sleep(discussion_duration)
        phase_end_event.set()
        mafia_message_event.set()
        
        for task in discussion_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Collect votes
        mafia_votes = []
        for mafia in living_mafia:
            other_mafia = [n for n in mafia_names if n != mafia.name]
            vote = await self.agents[mafia.name].run_mafia_vote(living_names, other_mafia)
            if vote:
                mafia_votes.append(vote)
                self.logger.info(f"  MAFIA {mafia.name} votes to kill: {vote}")
                self.event_log.log_event("NIGHT_ACTION", {"role": "MAFIA", "player": mafia.name, "target": vote, "action": "kill_vote"})
        
        if mafia_votes:
            vote_counts = Counter(mafia_votes)
            max_votes = max(vote_counts.values())
            top_voted = [t for t, c in vote_counts.items() if c == max_votes]
            
            if len(top_voted) == 1:
                final_target = top_voted[0]
                self.logger.info(f"  Mafia final target: {final_target}")
                return final_target
            else:
                self.logger.info(f"  Mafia vote tied - no kill")
                return None
        
        return None


async def run_single_game(
    game_id: int,
    model: str,
    day_duration: int,
    turn_limit: int,
    semaphore: asyncio.Semaphore,
) -> GameResult:
    """Run a single game and return the result."""
    async with semaphore:
        start_time = datetime.now()
        
        # Create a fresh LLM client for this game (shared cache)
        game = QuietMafiaGame(
            model=model,
            day_duration_seconds=day_duration,
            turn_limit=turn_limit,
            quiet=True,
        )
        
        print(f"  🎮 Game {game_id} started...")
        
        try:
            winner = await game.run_game()
        except Exception as e:
            print(f"  ❌ Game {game_id} failed: {e}")
            return GameResult(
                game_id=game.game_id,
                winner="ERROR",
                turns=0,
                mafia_killed=0,
                town_killed=0,
                prompt_tokens=0,
                completion_tokens=0,
                duration_seconds=0,
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Count deaths by role
        from infra.mafia import Role
        mafia_killed = sum(1 for p in game.players if p.role == Role.MAFIA and not p.is_alive)
        town_killed = sum(1 for p in game.players if p.role != Role.MAFIA and not p.is_alive)
        
        # Get token usage from the LLM client
        usage = game.llm_client.usage_stats
        
        result = GameResult(
            game_id=game.game_id,
            winner=winner,
            turns=game.state.day_number,
            mafia_killed=mafia_killed,
            town_killed=town_killed,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            duration_seconds=duration,
        )
        
        print(f"  ✅ Game {game_id} complete: {winner} wins ({result.turns} turns, {duration:.1f}s)")
        
        return result


async def run_batch(
    num_games: int,
    parallel: int,
    model: str,
    day_duration: int,
    turn_limit: int,
) -> BatchResults:
    """Run multiple games with controlled parallelism."""
    
    print(f"\n{'='*60}")
    print(f"BATCH RUN: {num_games} games (max {parallel} parallel)")
    print(f"Model: {model} | Day: {day_duration}s | Turn limit: {turn_limit}")
    print(f"{'='*60}\n")
    
    semaphore = asyncio.Semaphore(parallel)
    
    tasks = [
        run_single_game(i + 1, model, day_duration, turn_limit, semaphore)
        for i in range(num_games)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    batch_results = BatchResults()
    for result in results:
        if isinstance(result, GameResult):
            batch_results.games.append(result)
        else:
            print(f"  ⚠️ Game returned exception: {result}")
    
    return batch_results


def print_results(results: BatchResults):
    """Print formatted results summary."""
    summary = results.summary()
    
    if not summary:
        print("\n❌ No games completed!")
        return
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    completed = summary["total_games"] - summary["incomplete"]
    
    print(f"\n📊 Games: {summary['total_games']} total, {completed} completed")
    print(f"\n🏆 WIN RATES (of {completed} completed games):")
    print(f"   Town:  {summary['town_wins']:3d} wins ({summary['town_win_rate']*100:5.1f}%)")
    print(f"   Mafia: {summary['mafia_wins']:3d} wins ({summary['mafia_win_rate']*100:5.1f}%)")
    if summary["incomplete"]:
        print(f"   Incomplete: {summary['incomplete']}")
    
    print(f"\n⏱️  Average game: {summary['avg_turns']:.1f} turns, {summary['avg_duration_seconds']:.1f}s")
    
    # Cost calculation
    INPUT_PRICE = 0.05   # per 1M tokens
    OUTPUT_PRICE = 0.40  # per 1M tokens
    
    input_cost = (summary["total_prompt_tokens"] / 1_000_000) * INPUT_PRICE
    output_cost = (summary["total_completion_tokens"] / 1_000_000) * OUTPUT_PRICE
    total_cost = input_cost + output_cost
    
    print(f"\n💰 TOTAL COST (gpt-5-nano pricing):")
    print(f"   Tokens: {summary['total_tokens']:,} ({summary['total_prompt_tokens']:,} in / {summary['total_completion_tokens']:,} out)")
    print(f"   Cost:   ${total_cost:.4f} (${total_cost / summary['total_games']:.4f} per game)")
    
    # Save results to file
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"04_batch_{timestamp}.json"
    
    output = {
        "timestamp": timestamp,
        "summary": summary,
        "games": [
            {
                "game_id": g.game_id,
                "winner": g.winner,
                "turns": g.turns,
                "mafia_killed": g.mafia_killed,
                "town_killed": g.town_killed,
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "duration_seconds": g.duration_seconds,
            }
            for g in results.games
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run batch Mafia games")
    parser.add_argument("--games", "-n", type=int, default=10, help="Number of games to run")
    parser.add_argument("--parallel", "-p", type=int, default=5, help="Max parallel games")
    parser.add_argument("--model", "-m", type=str, default="gpt-5-nano", help="Model to use")
    parser.add_argument("--day-duration", "-d", type=int, default=30, help="Day phase duration (seconds)")
    parser.add_argument("--turn-limit", "-t", type=int, default=10, help="Max turns per game")
    
    args = parser.parse_args()
    
    results = asyncio.run(run_batch(
        num_games=args.games,
        parallel=args.parallel,
        model=args.model,
        day_duration=args.day_duration,
        turn_limit=args.turn_limit,
    ))
    
    print_results(results)


if __name__ == "__main__":
    main()
