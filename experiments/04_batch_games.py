#!/usr/bin/env python3
"""Run multiple ONUW games in parallel and analyze win rates."""

import asyncio
import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.game_engine import ONUWGame
from infra.onuw import Role, DEFAULT_ROLE_POOL

# Set up file logging for errors
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

error_logger = logging.getLogger("batch_errors")
error_logger.setLevel(logging.ERROR)

# Create file handler for errors
error_log_file = LOG_DIR / f"batch_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(error_log_file)
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))
error_logger.addHandler(file_handler)


@dataclass
class GameResult:
    """Result from a single game."""
    game_id: str
    winner: str  # "VILLAGE", "WEREWOLF", "TANNER", or "ERROR"
    killed_players: list[str]
    werewolves_in_game: int  # How many werewolves were among players (not center)
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


@dataclass 
class BatchResults:
    """Aggregated results from multiple games."""
    games: list[GameResult] = field(default_factory=list)
    
    @property
    def total_games(self) -> int:
        return len(self.games)
    
    @property
    def village_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "VILLAGE")
    
    @property
    def werewolf_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "WEREWOLF")
    
    @property
    def tanner_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "TANNER")
    
    @property
    def errors(self) -> int:
        return sum(1 for g in self.games if g.winner == "ERROR")
    
    def summary(self) -> dict:
        if not self.games:
            return {}
        
        completed = [g for g in self.games if g.winner in ("VILLAGE", "WEREWOLF", "TANNER")]
        
        return {
            "total_games": self.total_games,
            "village_wins": self.village_wins,
            "werewolf_wins": self.werewolf_wins,
            "tanner_wins": self.tanner_wins,
            "errors": self.errors,
            "village_win_rate": self.village_wins / len(completed) if completed else 0,
            "werewolf_win_rate": self.werewolf_wins / len(completed) if completed else 0,
            "tanner_win_rate": self.tanner_wins / len(completed) if completed else 0,
            "avg_duration_seconds": sum(g.duration_seconds for g in self.games) / len(self.games) if self.games else 0,
            "total_prompt_tokens": sum(g.prompt_tokens for g in self.games),
            "total_completion_tokens": sum(g.completion_tokens for g in self.games),
            "total_tokens": sum(g.prompt_tokens + g.completion_tokens for g in self.games),
        }


async def run_single_game(
    game_id: int,
    model: str,
    day_duration: int,
    num_players: int,
    semaphore: asyncio.Semaphore,
) -> GameResult:
    """Run a single ONUW game."""
    async with semaphore:
        start_time = datetime.now()
        
        # Create player names
        player_names = ["Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George"][:num_players]
        
        game = ONUWGame(
            player_names=player_names,
            role_pool=DEFAULT_ROLE_POOL,
            model=model,
            day_duration_seconds=day_duration,
        )
        
        print(f"  🎮 Game {game_id} started...")
        
        try:
            winner = await game.run_game()
        except Exception as e:
            error_msg = str(e)
            error_tb = traceback.format_exc()
            
            error_logger.error(
                f"Game {game_id} (internal ID: {game.game_id}) CRASHED\n"
                f"Phase: {game.state.phase.value if game.state.phase else 'unknown'}\n"
                f"Error: {error_msg}\n"
                f"Traceback:\n{error_tb}\n"
                f"{'='*60}"
            )
            
            print(f"  ❌ Game {game_id} failed: {error_msg}")
            print(f"     See {error_log_file} for details")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return GameResult(
                game_id=game.game_id,
                winner="ERROR",
                killed_players=[],
                werewolves_in_game=sum(1 for p in game.players if p.current_role == Role.WEREWOLF),
                prompt_tokens=game.llm_client.usage_stats.get("prompt_tokens", 0),
                completion_tokens=game.llm_client.usage_stats.get("completion_tokens", 0),
                duration_seconds=duration,
                error_message=error_msg,
                error_traceback=error_tb,
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get killed players
        killed = game.state.vote_result.killed_players if game.state.vote_result else []
        
        # Count werewolves in game
        werewolves = sum(1 for p in game.players if p.current_role == Role.WEREWOLF)
        
        # Get token usage
        usage = game.llm_client.usage_stats
        
        result = GameResult(
            game_id=game.game_id,
            winner=winner,
            killed_players=killed,
            werewolves_in_game=werewolves,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            duration_seconds=duration,
        )
        
        print(f"  ✅ Game {game_id}: {winner} wins ({duration:.1f}s)")
        
        return result


async def run_batch(
    num_games: int,
    parallel: int,
    model: str,
    day_duration: int,
    num_players: int,
) -> BatchResults:
    """Run multiple games with controlled parallelism."""
    
    print(f"\n{'='*60}")
    print(f"BATCH RUN: {num_games} ONUW games (max {parallel} parallel)")
    print(f"Model: {model} | Players: {num_players} | Day: {day_duration}s")
    print(f"Error log: {error_log_file}")
    print(f"{'='*60}\n")
    
    semaphore = asyncio.Semaphore(parallel)
    
    tasks = [
        run_single_game(i + 1, model, day_duration, num_players, semaphore)
        for i in range(num_games)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    batch_results = BatchResults()
    for i, result in enumerate(results):
        if isinstance(result, GameResult):
            batch_results.games.append(result)
        else:
            error_msg = str(result)
            error_tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
            
            error_logger.error(
                f"Game {i+1} returned raw exception (not caught in run_single_game)\n"
                f"Error: {error_msg}\n"
                f"Traceback:\n{error_tb}\n"
                f"{'='*60}"
            )
            
            print(f"  ⚠️ Game {i+1} returned exception: {error_msg}")
            
            batch_results.games.append(GameResult(
                game_id=f"unknown_{i+1}",
                winner="ERROR",
                killed_players=[],
                werewolves_in_game=0,
                prompt_tokens=0,
                completion_tokens=0,
                duration_seconds=0,
                error_message=error_msg,
                error_traceback=error_tb,
            ))
    
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
    
    completed = summary["total_games"] - summary["errors"]
    
    print(f"\n📊 Games: {summary['total_games']} total, {completed} completed")
    if summary["errors"]:
        print(f"   ⚠️  {summary['errors']} games had errors (see {error_log_file})")
    
    print(f"\n🏆 WIN RATES (of {completed} completed games):")
    print(f"   Village:  {summary['village_wins']:3d} wins ({summary['village_win_rate']*100:5.1f}%)")
    print(f"   Werewolf: {summary['werewolf_wins']:3d} wins ({summary['werewolf_win_rate']*100:5.1f}%)")
    if summary["tanner_wins"]:
        print(f"   Tanner:   {summary['tanner_wins']:3d} wins ({summary['tanner_win_rate']*100:5.1f}%)")
    
    print(f"\n⏱️  Average game: {summary['avg_duration_seconds']:.1f}s")
    
    # Cost calculation (adjust pricing for your model)
    INPUT_PRICE = 0.15   # per 1M tokens
    OUTPUT_PRICE = 0.60  # per 1M tokens
    
    input_cost = (summary["total_prompt_tokens"] / 1_000_000) * INPUT_PRICE
    output_cost = (summary["total_completion_tokens"] / 1_000_000) * OUTPUT_PRICE
    total_cost = input_cost + output_cost
    
    print("\n💰 TOTAL COST:")
    print(f"   Tokens: {summary['total_tokens']:,} ({summary['total_prompt_tokens']:,} in / {summary['total_completion_tokens']:,} out)")
    cost_per_game = total_cost / summary['total_games'] if summary['total_games'] > 0 else 0
    print(f"   Cost:   ${total_cost:.4f} (${cost_per_game:.4f} per game)")
    
    # Save results to file
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"04_batch_{timestamp}.json"
    
    output = {
        "timestamp": timestamp,
        "summary": summary,
        "error_log": str(error_log_file),
        "games": [
            {
                "game_id": g.game_id,
                "winner": g.winner,
                "killed_players": g.killed_players,
                "werewolves_in_game": g.werewolves_in_game,
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "duration_seconds": g.duration_seconds,
                "error_message": g.error_message,
                "error_traceback": g.error_traceback,
            }
            for g in results.games
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run batch ONUW games")
    parser.add_argument("--games", "-n", type=int, default=10, help="Number of games to run")
    parser.add_argument("--parallel", "-p", type=int, default=5, help="Max parallel games")
    parser.add_argument("--model", "-m", type=str, default="gpt-5-mini", help="Model to use")
    parser.add_argument("--day-duration", "-d", type=int, default=60, help="Day phase duration (seconds)")
    parser.add_argument("--players", type=int, default=5, help="Number of players")
    
    args = parser.parse_args()
    
    results = asyncio.run(run_batch(
        num_games=args.games,
        parallel=args.parallel,
        model=args.model,
        day_duration=args.day_duration,
        num_players=args.players,
    ))
    
    print_results(results)


if __name__ == "__main__":
    main()
