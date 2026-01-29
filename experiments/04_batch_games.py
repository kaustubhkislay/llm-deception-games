#!/usr/bin/env python3
"""Run multiple Mafia games in parallel and analyze win rates."""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.game_engine import MafiaGame
from infra.mafia import Role


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
    
    @property
    def errors(self) -> int:
        return sum(1 for g in self.games if g.winner == "ERROR")
    
    def summary(self) -> dict:
        if not self.games:
            return {}
        
        completed = [g for g in self.games if g.winner in ("TOWN", "MAFIA")]
        
        return {
            "total_games": self.total_games,
            "town_wins": self.town_wins,
            "mafia_wins": self.mafia_wins,
            "incomplete": self.incomplete,
            "errors": self.errors,
            "town_win_rate": self.town_wins / len(completed) if completed else 0,
            "mafia_win_rate": self.mafia_wins / len(completed) if completed else 0,
            "avg_turns": sum(g.turns for g in self.games) / len(self.games) if self.games else 0,
            "avg_duration_seconds": sum(g.duration_seconds for g in self.games) / len(self.games) if self.games else 0,
            "total_prompt_tokens": sum(g.prompt_tokens for g in self.games),
            "total_completion_tokens": sum(g.completion_tokens for g in self.games),
            "total_tokens": sum(g.prompt_tokens + g.completion_tokens for g in self.games),
        }


async def run_single_game(
    game_id: int,
    model: str,
    day_duration: int,
    turn_limit: int,
    semaphore: asyncio.Semaphore,
) -> GameResult:
    """Run a single game using the real MafiaGame class."""
    async with semaphore:
        start_time = datetime.now()
        
        # Use the real MafiaGame class - no duplication!
        game = MafiaGame(
            model=model,
            day_duration_seconds=day_duration,
            turn_limit=turn_limit,
        )
        
        print(f"  🎮 Game {game_id} started...")
        
        try:
            winner = await game.run_game()
        except Exception as e:
            print(f"  ❌ Game {game_id} failed: {e}")
            import traceback
            traceback.print_exc()
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
    
    completed = summary["total_games"] - summary["incomplete"] - summary["errors"]
    
    print(f"\n📊 Games: {summary['total_games']} total, {completed} completed")
    if summary["errors"]:
        print(f"   ⚠️  {summary['errors']} games had errors")
    
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
    
    print(f"\n💰 TOTAL COST (gpt-5-mini pricing):")
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
    parser.add_argument("--model", "-m", type=str, default="gpt-5-mini", help="Model to use")
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
