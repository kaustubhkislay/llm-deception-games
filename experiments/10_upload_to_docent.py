#!/usr/bin/env python3
"""Upload batch game results to Docent for viewing and analysis.

This module converts ONUW game logs to Docent AgentRun format, allowing
you to view full game transcripts in Docent's web interface.

Usage:
    # From command line:
    python experiments/10_upload_to_docent.py results/08_batch_xxx.json --collection "my_collection"
    
    # Or programmatically:
    from experiments.upload_to_docent import upload_batch_to_docent
    upload_batch_to_docent("results/08_batch_xxx.json", "my_collection")
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.game_engine import LOG_DIR, load_game_from_log


def make_message(role: str, content: str, metadata: Optional[dict] = None, **kwargs) -> Any:
    """
    Create a Docent chat message using parse_chat_message.
    
    Args:
        role: Message role ("system", "user", "assistant", "tool")
        content: Message content
        metadata: Optional metadata dict to attach to the message
        **kwargs: Additional message fields (tool_call_id for tool messages, etc.)
    
    Returns:
        A Docent chat message object.
    """
    msg_dict = {"role": role, "content": content}
    if metadata:
        msg_dict["metadata"] = metadata
    msg_dict.update(kwargs)
    return parse_chat_message(msg_dict)


def load_player_thoughts_by_round(log_path: Path) -> dict[str, dict[int, list[dict]]]:
    """
    Load player thoughts from a game log, organized by player and round.
    
    Args:
        log_path: Path to the game's .jsonl log file
    
    Returns:
        Dict of player_name -> round_number -> list of thought dicts
        Round 0 is used for NIGHT phase, -1 for VOTING phase
    """
    thoughts_by_player: dict[str, dict[int, list[dict]]] = {}
    current_round = 0  # 0 = NIGHT, 1+ = DAY rounds, -1 = VOTING
    
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            event = json.loads(line)
            event_type = event.get("event_type", "")
            data = event.get("data", {})
            
            if event_type == "ROUND_START":
                current_round = data.get("round", current_round)
            
            elif event_type == "PHASE_CHANGE":
                phase = data.get("phase", "")
                if phase == "NIGHT":
                    current_round = 0
                elif phase == "VOTING":
                    current_round = -1
            
            elif event_type == "PLAYER_THOUGHT":
                player_name = data.get("player_name", "unknown")
                
                if player_name not in thoughts_by_player:
                    thoughts_by_player[player_name] = {}
                
                if current_round not in thoughts_by_player[player_name]:
                    thoughts_by_player[player_name][current_round] = []
                
                thoughts_by_player[player_name][current_round].append({
                    "prompt": data.get("prompt", ""),
                    "response": data.get("response", ""),
                    "tool_calls": data.get("tool_calls"),
                    "phase": data.get("phase", ""),
                })
    
    return thoughts_by_player


def extract_game_rules_from_log(log_path: Path) -> Optional[str]:
    """
    Extract the game rules system prompt from a game log.
    
    The system prompt is stored in the first PLAYER_THOUGHT event that has
    a system_prompt field. It contains the full game rules.
    
    Args:
        log_path: Path to the game's .jsonl log file
    
    Returns:
        The game rules system prompt, or None if not found.
    """
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            event = json.loads(line)
            event_type = event.get("event_type", "")
            data = event.get("data", {})
            
            if event_type == "PLAYER_THOUGHT":
                system_prompt = data.get("system_prompt")
                if system_prompt:
                    # Extract just the rules portion (before the role-specific section)
                    # The rules are everything up to "=== YOUR STARTING ROLE"
                    if "=== YOUR STARTING ROLE" in system_prompt:
                        rules_end = system_prompt.index("=== YOUR STARTING ROLE")
                        return system_prompt[:rules_end].strip()
                    return system_prompt
    
    return None


def format_player_thoughts(thoughts: list[dict]) -> str:
    """Format a list of player thought dicts into a readable string."""
    parts = []
    for i, thought in enumerate(thoughts):
        response = thought.get("response", "")
        tool_calls = thought.get("tool_calls")
        
        if response:
            parts.append(f"Thinking: {response}")
        
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                parts.append(f"Action: {name}({args})")
    
    return "\n\n".join(parts) if parts else "(no thoughts recorded)"


def get_docent_client() -> Docent:
    """Get a Docent client instance."""
    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        raise ValueError(
            "DOCENT_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    return Docent(api_key=api_key)


def game_log_to_transcript(
    game_data: dict,
    player_name: Optional[str] = None,
    log_path: Optional[Path] = None,
) -> Transcript:
    """
    Convert a game log to a Docent Transcript.
    
    Args:
        game_data: Loaded game data from load_game_from_log()
        player_name: If provided, create transcript from this player's POV.
                    If None, create a "God view" transcript showing all events.
        log_path: Path to the raw .jsonl log file (needed for God view with thoughts)
    
    Returns:
        A Docent Transcript containing the game conversation.
    """
    messages: list[Any] = []
    
    if player_name:
        # Player-specific transcript using their chat history
        player_thoughts = game_data.get("player_thoughts", {})
        
        if player_name not in player_thoughts:
            # No thoughts recorded for this player
            messages.append(make_message(
                "user",
                f"[No recorded thoughts for player {player_name}]"
            ))
        else:
            # Iterate through phases and build the conversation
            thoughts = player_thoughts[player_name]
            
            for phase_key in sorted(thoughts.keys()):
                phase_messages = thoughts[phase_key]
                
                for msg in phase_messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls")
                    
                    if role == "system":
                        messages.append(make_message("system", content))
                    elif role == "user":
                        messages.append(make_message("user", content))
                    elif role == "assistant":
                        # Include tool calls info if present
                        if tool_calls:
                            tool_info = "\n\n[Tool calls: " + ", ".join(
                                f"{tc['function']['name']}({tc['function']['arguments']})"
                                for tc in tool_calls
                            ) + "]"
                            content = (content or "") + tool_info
                        messages.append(make_message("assistant", content or ""))
                    elif role == "tool":
                        messages.append(make_message(
                            "tool",
                            content,
                            tool_call_id=msg.get("tool_call_id", "unknown")
                        ))
    else:
        # God view transcript showing game events
        # Load player thoughts organized by round if log_path is available
        thoughts_by_round: dict[str, dict[int, list[dict]]] = {}
        if log_path and log_path.exists():
            thoughts_by_round = load_player_thoughts_by_round(log_path)
        
        # Get all player names
        all_players = [p["name"] for p in game_data.get("players", [])]
        
        messages.append(make_message(
            "system",
            "=== ONE NIGHT ULTIMATE WEREWOLF - Game Transcript ===\n\n"
            "This is a God-view transcript showing all game events.\n"
            "Player private thoughts are attached as metadata on their messages."
        ))
        
        # Add game setup info
        players_info = "\n".join(
            f"  - {p['name']}: {p['original_role']} -> {p['current_role']}"
            for p in game_data.get("players", [])
        )
        center_cards = ", ".join(game_data.get("center_cards", []))
        
        setup_msg = f"""GAME SETUP:
Players:
{players_info}

Center Cards: [{center_cards}]
"""
        messages.append(make_message("user", setup_msg))
        
        # Walk through phase history
        for phase in game_data.get("phase_history", []):
            phase_name = phase.get("phase", "UNKNOWN")
            
            # Phase header
            messages.append(make_message(
                "user",
                f"\n=== {phase_name} PHASE ==="
            ))
            
            # GM messages
            for gm_msg in phase.get("gmMessages", []):
                messages.append(make_message(
                    "assistant",
                    f"[GM] {gm_msg.get('content', '')}"
                ))
            
            # For DAY phase, process messages by round and include silent players
            if phase_name == "DAY":
                # Group messages by round
                messages_by_round: dict[int, list[dict]] = {}
                other_events: list[dict] = []
                
                for event in phase.get("events", []):
                    event_type = event.get("type", "")
                    if event_type == "message":
                        round_num = event.get("round", 0)
                        if round_num not in messages_by_round:
                            messages_by_round[round_num] = []
                        messages_by_round[round_num].append(event)
                    elif event_type not in ("gm_message",):
                        other_events.append(event)
                
                # Process each round
                for round_num in sorted(messages_by_round.keys()):
                    round_msgs = messages_by_round[round_num]
                    senders_this_round = {m.get("sender") for m in round_msgs}
                    
                    # Add round header
                    messages.append(make_message(
                        "assistant",
                        f"--- Round {round_num} ---"
                    ))
                    
                    # Add messages from players who spoke
                    for event in round_msgs:
                        sender = event.get("sender", "?")
                        content = event.get("content", "")
                        
                        # Get player's thoughts for this round
                        player_round_thoughts = thoughts_by_round.get(sender, {}).get(round_num, [])
                        thoughts_text = format_player_thoughts(player_round_thoughts)
                        
                        messages.append(make_message(
                            "user",
                            f"[Round {round_num}] {sender}: {content}",
                            metadata={"player_private_thoughts": thoughts_text}
                        ))
                    
                    # Add placeholder for players who didn't speak
                    for player in all_players:
                        if player not in senders_this_round:
                            player_round_thoughts = thoughts_by_round.get(player, {}).get(round_num, [])
                            thoughts_text = format_player_thoughts(player_round_thoughts)
                            
                            messages.append(make_message(
                                "user",
                                f"[Round {round_num}] {player}: (did not send a message)",
                                metadata={"player_private_thoughts": thoughts_text}
                            ))
                
                # Add any other events (shouldn't be many in DAY phase)
                for event in other_events:
                    event_type = event.get("type", "")
                    if event_type == "card_swap":
                        loc1 = event.get("location1", "?")
                        loc2 = event.get("location2", "?")
                        messages.append(make_message(
                            "assistant",
                            f"[Card Swap] {loc1} <-> {loc2}"
                        ))
            
            elif phase_name == "VOTING":
                # Process votes with player thoughts
                voters_seen = set()
                
                for event in phase.get("events", []):
                    event_type = event.get("type", "")
                    
                    if event_type == "vote":
                        voter = event.get("voter", "?")
                        target = event.get("target", "?")
                        reasoning = event.get("reasoning", "")
                        voters_seen.add(voter)
                        
                        # Get player's voting thoughts (round -1)
                        player_vote_thoughts = thoughts_by_round.get(voter, {}).get(-1, [])
                        thoughts_text = format_player_thoughts(player_vote_thoughts)
                        
                        vote_str = f"[Vote] {voter} -> {target}"
                        if reasoning:
                            vote_str += f"\n  Reasoning: {reasoning[:500]}"
                        
                        messages.append(make_message(
                            "user",
                            vote_str,
                            metadata={"player_private_thoughts": thoughts_text}
                        ))
                    
                    elif event_type == "vote_result":
                        votes = event.get("votes", {})
                        killed = event.get("killed", [])
                        messages.append(make_message(
                            "assistant",
                            f"[Vote Result] Killed: {killed}\nVotes: {votes}"
                        ))
                
                # Add placeholder for players who didn't vote (shouldn't happen normally)
                for player in all_players:
                    if player not in voters_seen:
                        player_vote_thoughts = thoughts_by_round.get(player, {}).get(-1, [])
                        thoughts_text = format_player_thoughts(player_vote_thoughts)
                        
                        messages.append(make_message(
                            "user",
                            f"[Vote] {player}: (no vote recorded)",
                            metadata={"player_private_thoughts": thoughts_text}
                        ))
            
            else:
                # Other phases (NIGHT, GAME_OVER) - process events normally
                for event in phase.get("events", []):
                    event_type = event.get("type", "")
                    
                    if event_type == "night_action":
                        player = event.get("player", "?")
                        role = event.get("role", "?")
                        action = event.get("action", "?")
                        targets = event.get("targets", [])
                        result = event.get("result")
                        
                        # Get night phase thoughts (round 0)
                        player_night_thoughts = thoughts_by_round.get(player, {}).get(0, [])
                        thoughts_text = format_player_thoughts(player_night_thoughts)
                        
                        action_str = f"[Night Action] {player} ({role}): {action}"
                        if targets:
                            action_str += f" -> {targets}"
                        if result:
                            action_str += f" = {result}"
                        
                        messages.append(make_message(
                            "assistant",
                            action_str,
                            metadata={"player_private_thoughts": thoughts_text}
                        ))
                        
                    elif event_type == "card_swap":
                        loc1 = event.get("location1", "?")
                        loc2 = event.get("location2", "?")
                        messages.append(make_message(
                            "assistant",
                            f"[Card Swap] {loc1} <-> {loc2}"
                        ))
        
        # Add game result
        winner = game_data.get("winner", "UNKNOWN")
        killed = game_data.get("killed", [])
        
        result_msg = f"""
=== GAME RESULT ===
Winner: {winner}
Killed players: {killed}

Final roles:
{players_info}
"""
        messages.append(make_message("user", result_msg))
    
    return Transcript(messages=messages)


def game_result_to_agent_run(
    game_result: dict,
    batch_metadata: Optional[dict] = None,
    player_perspective: Optional[str] = None,
) -> Optional[AgentRun]:
    """
    Convert a game result from batch JSON to a Docent AgentRun.
    
    Args:
        game_result: Single game result dict from batch JSON
        batch_metadata: Optional metadata from the batch (config_dir, etc.)
        player_perspective: If provided, create from this player's POV.
                           If None, create a "God view" transcript.
    
    Returns:
        AgentRun or None if game log not found.
    """
    game_id = game_result.get("game_id", "unknown")
    
    # Find and load the game log
    log_file = LOG_DIR / f"game_{game_id}.jsonl"
    
    if not log_file.exists():
        print(f"Warning: Game log not found for {game_id}")
        return None
    
    try:
        game_data = load_game_from_log(str(log_file))
    except Exception as e:
        print(f"Warning: Failed to load game log {game_id}: {e}")
        return None
    
    # Build metadata
    metadata = {
        "game_id": game_id,
        "seed": game_result.get("seed"),
        "winner": game_result.get("winner"),
        "killed_players": game_result.get("killed_players", []),
        "error_type": game_result.get("error_type"),
        "duration_seconds": game_result.get("duration_seconds"),
        "prompt_tokens": game_result.get("prompt_tokens", 0),
        "completion_tokens": game_result.get("completion_tokens", 0),
    }
    
    # Add player info
    players = game_result.get("players", [])
    if players:
        metadata["players"] = {
            p["name"]: {
                "starting_role": p["starting_role"],
                "ending_role": p["ending_role"],
                "won": p["won"],
                "swap_count": p.get("swap_count", 0),
            }
            for p in players
        }
    
    # Add batch metadata if provided
    if batch_metadata:
        metadata["config_dir"] = batch_metadata.get("config_dir")
        metadata["batch_start_time"] = batch_metadata.get("start_time")
    
    # Extract and add game rules system prompt
    game_rules = extract_game_rules_from_log(log_file)
    if game_rules:
        metadata["game_rules_system_prompt"] = game_rules
    
    # Add perspective info
    if player_perspective:
        metadata["perspective"] = player_perspective
    else:
        metadata["perspective"] = "God View"
    
    # Create transcript (pass log_file for God view to get player thoughts)
    transcript = game_log_to_transcript(game_data, player_perspective, log_file)
    
    return AgentRun(
        transcripts=[transcript],
        metadata=metadata,
    )


def upload_batch_to_docent(
    results_path: str | Path,
    collection_name: str,
    player_perspectives: Optional[list[str]] = None,
    skip_errors: bool = True,
    version_existing: bool = True,
) -> str:
    """
    Upload batch results to Docent.
    
    Args:
        results_path: Path to the batch results JSON file
        collection_name: Name of the Docent collection to create/update
        player_perspectives: List of player names to create separate transcripts for.
                           If None, only creates "God view" transcripts.
                           Use ["all"] to create transcripts for all players.
        skip_errors: If True, skip games that had errors
        version_existing: If True, increment version number for existing collections
    
    Returns:
        Collection ID
    """
    # Load batch results
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path) as f:
        batch_data = json.load(f)
    
    games = batch_data.get("games", [])
    if not games:
        raise ValueError("No games found in batch results")
    
    print(f"Loaded {len(games)} games from {results_path}")
    
    # Extract batch metadata
    batch_metadata = {
        "config_dir": batch_data.get("config_dir"),
        "start_time": batch_data.get("start_time"),
        "summary": batch_data.get("summary", {}),
    }
    
    # Get Docent client
    client = get_docent_client()
    
    # Create or get collection
    collections = client.list_collections()
    matching = [c for c in collections if c["name"] == collection_name]
    
    if not matching:
        collection_id = client.create_collection(
            name=collection_name,
            description=f"ONUW batch results from {batch_metadata.get('config_dir', 'unknown')}",
        )
        print(f"Created new collection: {collection_name}")
    else:
        collection_id = matching[0]["id"]
        print(f"Using existing collection: {collection_name}")
    
    # Get existing version info if versioning
    highest_version = 0
    if version_existing:
        existing_run_ids = client.list_agent_run_ids(collection_id)
        if existing_run_ids:
            print(f"Found {len(existing_run_ids)} existing runs, checking versions...")
            existing_runs = [
                client.get_agent_run(collection_id, rid)
                for rid in tqdm(existing_run_ids, desc="Loading existing runs")
            ]
            highest_version = max(
                (int(r.metadata.get("version", 0)) for r in existing_runs if r and r.metadata),
                default=0,
            )
            print(f"Highest existing version: {highest_version}")
    
    new_version = highest_version + 1
    
    # Determine player perspectives to use
    perspectives_to_create: list[Optional[str]] = [None]  # Always create God view
    
    if player_perspectives:
        if player_perspectives == ["all"]:
            # Get all player names from first game
            if games and games[0].get("players"):
                perspectives_to_create.extend(
                    p["name"] for p in games[0]["players"]
                )
        else:
            perspectives_to_create.extend(player_perspectives)
    
    # Convert games to agent runs
    agent_runs: list[AgentRun] = []
    skipped = 0
    
    for game in tqdm(games, desc="Converting games"):
        # Skip errors if requested
        if skip_errors and game.get("error_type"):
            skipped += 1
            continue
        
        for perspective in perspectives_to_create:
            agent_run = game_result_to_agent_run(
                game, 
                batch_metadata,
                player_perspective=perspective,
            )
            
            if agent_run:
                agent_run.metadata["version"] = new_version
                agent_runs.append(agent_run)
    
    if skipped:
        print(f"Skipped {skipped} games with errors")
    
    if not agent_runs:
        print("No agent runs to upload!")
        return collection_id
    
    print(f"Uploading {len(agent_runs)} agent runs to Docent...")
    client.add_agent_runs(collection_id, agent_runs)
    
    print(f"Successfully uploaded {len(agent_runs)} runs to collection '{collection_name}'")
    print(f"Collection ID: {collection_id}")
    
    return collection_id


def upload_single_game_to_docent(
    game_id: str,
    collection_name: str,
    player_perspectives: Optional[list[str]] = None,
) -> str:
    """
    Upload a single game to Docent by game ID.
    
    Args:
        game_id: Game ID (timestamp format like "20260203_173221")
        collection_name: Name of the Docent collection
        player_perspectives: List of player names to create transcripts for
    
    Returns:
        Collection ID
    """
    log_file = LOG_DIR / f"game_{game_id}.jsonl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Game log not found: {log_file}")
    
    game_data = load_game_from_log(str(log_file))
    
    client = get_docent_client()
    
    # Create or get collection
    collections = client.list_collections()
    matching = [c for c in collections if c["name"] == collection_name]
    
    if not matching:
        collection_id = client.create_collection(
            name=collection_name,
            description=f"ONUW game {game_id}",
        )
    else:
        collection_id = matching[0]["id"]
    
    # Determine perspectives
    perspectives: list[Optional[str]] = [None]  # God view
    
    if player_perspectives:
        if player_perspectives == ["all"]:
            perspectives.extend(
                p["name"] for p in game_data.get("players", [])
            )
        else:
            perspectives.extend(player_perspectives)
    
    # Create agent runs
    agent_runs: list[AgentRun] = []
    
    for perspective in perspectives:
        transcript = game_log_to_transcript(game_data, perspective, log_file)
        
        metadata = {
            "game_id": game_id,
            "winner": game_data.get("winner"),
            "killed": game_data.get("killed", []),
            "perspective": perspective or "God View",
        }
        
        # Add player info
        if game_data.get("players"):
            metadata["players"] = {
                p["name"]: {
                    "original_role": p["original_role"],
                    "current_role": p["current_role"],
                }
                for p in game_data["players"]
            }
        
        # Add game rules system prompt
        game_rules = extract_game_rules_from_log(log_file)
        if game_rules:
            metadata["game_rules_system_prompt"] = game_rules
        
        agent_runs.append(AgentRun(
            transcripts=[transcript],
            metadata=metadata,
        ))
    
    print(f"Uploading {len(agent_runs)} agent runs...")
    client.add_agent_runs(collection_id, agent_runs)
    
    print(f"Successfully uploaded to collection '{collection_name}'")
    return collection_id


def main():
    parser = argparse.ArgumentParser(
        description="Upload ONUW batch results to Docent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload batch results with God view only:
  python 10_upload_to_docent.py results/08_batch_xxx.json --collection "My Games"
  
  # Upload with player perspectives:
  python 10_upload_to_docent.py results/08_batch_xxx.json --collection "My Games" --players Alice Bob
  
  # Upload with all player perspectives:
  python 10_upload_to_docent.py results/08_batch_xxx.json --collection "My Games" --players all
  
  # Upload a single game by ID:
  python 10_upload_to_docent.py --game-id 20260203_173221 --collection "Single Game"
"""
    )
    
    parser.add_argument(
        "results_path",
        type=str,
        nargs="?",
        help="Path to batch results JSON file"
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        required=True,
        help="Name of the Docent collection"
    )
    parser.add_argument(
        "--players", "-p",
        type=str,
        nargs="*",
        default=None,
        help="Player names to create perspectives for (use 'all' for all players)"
    )
    parser.add_argument(
        "--game-id", "-g",
        type=str,
        default=None,
        help="Upload a single game by ID instead of batch results"
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include games that had errors"
    )
    parser.add_argument(
        "--no-version",
        action="store_true",
        help="Don't increment version for existing collections"
    )
    
    args = parser.parse_args()
    
    if args.game_id:
        # Upload single game
        upload_single_game_to_docent(
            game_id=args.game_id,
            collection_name=args.collection,
            player_perspectives=args.players,
        )
    elif args.results_path:
        # Upload batch
        upload_batch_to_docent(
            results_path=args.results_path,
            collection_name=args.collection,
            player_perspectives=args.players,
            skip_errors=not args.include_errors,
            version_existing=not args.no_version,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
