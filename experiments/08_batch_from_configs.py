#!/usr/bin/env python3
"""Run multiple ONUW games from config files in a directory."""

import asyncio
import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import RateLimitError

from infra.game_engine import ONUWGame, LOG_DIR
from infra.onuw import Role, GameConfig

# Set up file logging for errors
BATCH_LOG_DIR = Path(__file__).parent.parent / "logs"
BATCH_LOG_DIR.mkdir(exist_ok=True)

error_logger = logging.getLogger("batch_errors")
error_logger.setLevel(logging.ERROR)


@dataclass
class PlayerResult:
    """Result for a single player in a game."""
    name: str
    starting_role: str
    ending_role: str
    swap_count: int
    won: bool  # Did this player's team win?


@dataclass
class GameResult:
    """Detailed result from a single game."""
    game_id: str
    seed: int
    config_file: str
    winner: str  # "VILLAGE", "WEREWOLF", "TANNER", or "ERROR"
    killed_players: list[str]
    players: list[PlayerResult]
    center_cards: list[str]
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float
    error_type: Optional[str] = None  # "RATE_LIMIT", "OTHER", or None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


@dataclass 
class BatchResults:
    """Aggregated results from multiple games."""
    config_dir: str
    start_time: str
    games: list[GameResult] = field(default_factory=list)
    rate_limit_errors: int = 0
    other_errors: int = 0
    
    @property
    def total_games(self) -> int:
        return len(self.games)
    
    @property
    def completed_games(self) -> int:
        return sum(1 for g in self.games if g.winner in ("VILLAGE", "WEREWOLF", "TANNER"))
    
    @property
    def village_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "VILLAGE")
    
    @property
    def werewolf_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "WEREWOLF")
    
    @property
    def tanner_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == "TANNER")
    
    def summary(self) -> dict:
        if not self.games:
            return {}
        
        completed = self.completed_games
        
        return {
            "total_games": self.total_games,
            "completed_games": completed,
            "village_wins": self.village_wins,
            "werewolf_wins": self.werewolf_wins,
            "tanner_wins": self.tanner_wins,
            "rate_limit_errors": self.rate_limit_errors,
            "other_errors": self.other_errors,
            "village_win_rate": self.village_wins / completed if completed else 0,
            "werewolf_win_rate": self.werewolf_wins / completed if completed else 0,
            "tanner_win_rate": self.tanner_wins / completed if completed else 0,
            "avg_duration_seconds": sum(g.duration_seconds for g in self.games) / len(self.games) if self.games else 0,
            "total_prompt_tokens": sum(g.prompt_tokens for g in self.games),
            "total_completion_tokens": sum(g.completion_tokens for g in self.games),
        }


def get_team(role: Role) -> str:
    """Get the team for a given role."""
    if role in (Role.WEREWOLF, Role.MINION):
        return "WEREWOLF"
    elif role == Role.TANNER:
        return "TANNER"
    else:
        return "VILLAGE"


def count_player_swaps(game_log_file: Path) -> dict[str, int]:
    """
    Count how many times each player was involved in a swap.
    
    Args:
        game_log_file: Path to the game's .jsonl log file
    
    Returns:
        Dict of player_name -> swap_count
    """
    swap_counts: dict[str, int] = {}
    
    if not game_log_file.exists():
        return swap_counts
    
    with open(game_log_file) as f:
        for line in f:
            try:
                event = json.loads(line)
                if event.get("event_type") == "CARD_SWAP":
                    data = event.get("data", {})
                    loc1 = data.get("location1", "")
                    loc2 = data.get("location2", "")
                    
                    # Only count player swaps, not center card swaps
                    if loc1 and not loc1.startswith("center_"):
                        swap_counts[loc1] = swap_counts.get(loc1, 0) + 1
                    if loc2 and not loc2.startswith("center_"):
                        swap_counts[loc2] = swap_counts.get(loc2, 0) + 1
            except json.JSONDecodeError:
                continue
    
    return swap_counts


async def run_single_game(
    config_file: Path,
    semaphore: asyncio.Semaphore,
    batch_name: str | None = None,
) -> GameResult:
    """Run a single ONUW game from a config file."""
    async with semaphore:
        start_time = datetime.now()
        
        # Load config
        with open(config_file) as f:
            config_data = json.load(f)
        
        config = GameConfig.from_dict(config_data)
        seed = config.seed
        
        # Generate game name
        game_name = f"{batch_name} seed={seed}" if batch_name else f"Game seed={seed}"
        
        game = ONUWGame(config=config, name=game_name)
        game_id = game.game_id
        
        print(f"  Game seed={seed} started (ID: {game_id})...")
        
        try:
            winner = await game.run_game()
            error_type = None
            error_msg = None
            error_tb = None
            
        except RateLimitError as e:
            error_type = "RATE_LIMIT"
            error_msg = str(e)
            error_tb = traceback.format_exc()
            winner = "ERROR"
            
            error_logger.error(
                f"Game seed={seed} (ID: {game_id}) RATE LIMITED\n"
                f"Error: {error_msg}\n"
                f"{'='*60}"
            )
            print(f"  RATE LIMITED: Game seed={seed}")
            
        except Exception as e:
            error_type = "OTHER"
            error_msg = str(e)
            error_tb = traceback.format_exc()
            winner = "ERROR"
            
            error_logger.error(
                f"Game seed={seed} (ID: {game_id}) CRASHED\n"
                f"Phase: {game.state.phase.value if game.state.phase else 'unknown'}\n"
                f"Error: {error_msg}\n"
                f"Traceback:\n{error_tb}\n"
                f"{'='*60}"
            )
            print(f"  ERROR: Game seed={seed}: {error_msg}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get killed players
        killed = game.state.vote_result.killed_players if game.state.vote_result else []
        
        # Count swaps per player from the game log
        game_log_file = LOG_DIR / f"game_{game_id}.jsonl"
        swap_counts = count_player_swaps(game_log_file)
        
        # Build player results
        player_results = []
        for p in game.players:
            ending_team = get_team(p.current_role)
            
            # Determine if this player won
            if winner == "ERROR":
                won = False
            elif winner == "TANNER" and p.current_role == Role.TANNER:
                # Tanner wins alone
                won = True
            elif winner == "TANNER":
                won = False
            else:
                won = (ending_team == winner)
            
            player_results.append(PlayerResult(
                name=p.name,
                starting_role=p.original_role.value,
                ending_role=p.current_role.value,
                swap_count=swap_counts.get(p.name, 0),
                won=won,
            ))
        
        # Get token usage
        usage = game.llm_client.usage_stats
        
        result = GameResult(
            game_id=game_id,
            seed=seed,
            config_file=str(config_file),
            winner=winner,
            killed_players=killed,
            players=player_results,
            center_cards=[r.value for r in game.state.center_cards],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            duration_seconds=duration,
            error_type=error_type,
            error_message=error_msg,
            error_traceback=error_tb,
        )
        
        if winner != "ERROR":
            print(f"  Completed: seed={seed} -> {winner} wins ({duration:.1f}s)")
        
        return result


async def run_batch(
    config_dir: Path,
    parallel: int = 5,
    batch_name: str | None = None,
    max_games: int | None = None,
) -> BatchResults:
    """Run games from all configs in a directory."""
    
    # Find all config files
    config_files = sorted(config_dir.glob("config_*.json"))
    
    if not config_files:
        print(f"No config files found in {config_dir}")
        return BatchResults(config_dir=str(config_dir), start_time=datetime.now().isoformat())
    
    if max_games:
        config_files = config_files[:max_games]
    
    # Set up error logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_log_file = BATCH_LOG_DIR / f"batch_errors_{timestamp}.log"
    file_handler = logging.FileHandler(error_log_file)
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    error_logger.addHandler(file_handler)
    
    print(f"\n{'='*60}")
    print(f"BATCH RUN: {len(config_files)} games from {config_dir}")
    print(f"Max parallel: {parallel}")
    if batch_name:
        print(f"Batch name: {batch_name}")
    print(f"Error log: {error_log_file}")
    print(f"{'='*60}\n")
    
    batch_results = BatchResults(
        config_dir=str(config_dir),
        start_time=datetime.now().isoformat(),
    )
    
    semaphore = asyncio.Semaphore(parallel)
    
    tasks = [
        run_single_game(config_file, semaphore, batch_name)
        for config_file in config_files
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, GameResult):
            batch_results.games.append(result)
            if result.error_type == "RATE_LIMIT":
                batch_results.rate_limit_errors += 1
            elif result.error_type == "OTHER":
                batch_results.other_errors += 1
        else:
            # Raw exception (not caught in run_single_game)
            error_msg = str(result)
            error_tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
            
            error_logger.error(
                f"Config {i} returned raw exception\n"
                f"Error: {error_msg}\n"
                f"Traceback:\n{error_tb}\n"
                f"{'='*60}"
            )
            
            print(f"  Raw exception for config {i}: {error_msg}")
            batch_results.other_errors += 1
            
            # Create a minimal error result
            batch_results.games.append(GameResult(
                game_id=f"unknown_{i}",
                seed=-1,
                config_file=str(config_files[i]) if i < len(config_files) else "unknown",
                winner="ERROR",
                killed_players=[],
                players=[],
                center_cards=[],
                prompt_tokens=0,
                completion_tokens=0,
                duration_seconds=0,
                error_type="OTHER",
                error_message=error_msg,
                error_traceback=error_tb,
            ))
    
    return batch_results


def print_results(results: BatchResults, error_log_file: Path | None = None):
    """Print formatted results summary."""
    summary = results.summary()
    
    if not summary:
        print("\nNo games completed!")
        return
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    completed = summary["completed_games"]
    
    print(f"\nGames: {summary['total_games']} total, {completed} completed")
    if summary["rate_limit_errors"]:
        print(f"   RATE LIMIT ERRORS: {summary['rate_limit_errors']}")
    if summary["other_errors"]:
        print(f"   OTHER ERRORS: {summary['other_errors']}")
    
    if completed > 0:
        print(f"\nWIN RATES (of {completed} completed games):")
        print(f"   Village:  {summary['village_wins']:3d} wins ({summary['village_win_rate']*100:5.1f}%)")
        print(f"   Werewolf: {summary['werewolf_wins']:3d} wins ({summary['werewolf_win_rate']*100:5.1f}%)")
        if summary["tanner_wins"]:
            print(f"   Tanner:   {summary['tanner_wins']:3d} wins ({summary['tanner_win_rate']*100:5.1f}%)")
    
    print(f"\nAverage game: {summary['avg_duration_seconds']:.1f}s")
    
    # Token usage
    total_tokens = summary["total_prompt_tokens"] + summary["total_completion_tokens"]
    print(f"\nTokens: {total_tokens:,} total ({summary['total_prompt_tokens']:,} in / {summary['total_completion_tokens']:,} out)")


def save_results(results: BatchResults) -> Path:
    """Save detailed results to JSON file."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"08_batch_{timestamp}.json"
    
    # Convert to serializable format
    output = {
        "config_dir": results.config_dir,
        "start_time": results.start_time,
        "summary": results.summary(),
        "rate_limit_errors": results.rate_limit_errors,
        "other_errors": results.other_errors,
        "games": [
            {
                "game_id": g.game_id,
                "seed": g.seed,
                "config_file": g.config_file,
                "winner": g.winner,
                "killed_players": g.killed_players,
                "players": [asdict(p) for p in g.players],
                "center_cards": g.center_cards,
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "duration_seconds": g.duration_seconds,
                "error_type": g.error_type,
                "error_message": g.error_message,
            }
            for g in results.games
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Run ONUW games from config files")
    parser.add_argument(
        "config_dir",
        type=str,
        help="Directory containing config files (e.g., 'settings/vanilla onuw gpt5mini')"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=5,
        help="Max parallel games"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Batch name for game labeling"
    )
    parser.add_argument(
        "--max-games", "-m",
        type=int,
        default=None,
        help="Maximum number of games to run (for testing)"
    )
    
    args = parser.parse_args()
    
    # Resolve config directory
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = Path(__file__).parent.parent / config_dir
    
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)
    
    results = asyncio.run(run_batch(
        config_dir=config_dir,
        parallel=args.parallel,
        batch_name=args.name,
        max_games=args.max_games,
    ))
    
    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()
