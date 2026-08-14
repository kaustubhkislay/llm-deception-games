#!/usr/bin/env python3
"""Export batch results + game logs + judgments to Parquet tables for sharing.

Produces four tables in results/parquet/:
  games.parquet    — one row per game (condition, seed, outcome, judge verdict)
  players.parquet  — one row per player-instance (model, roles, judge scores)
  messages.parquet — one row per public message
  thoughts.parquet — one row per private LLM call (reasoning summary + response)

Usage:
    python experiments/18_export_parquet.py --spec results/suite_spec.json
Conditions are read from the suite spec; judged files are joined when present.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "game_logs"

PLAYER_DIMS = ["information_handling", "deception_quality", "deduction_quality",
               "vote_play", "reasoning_message_gap"]


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, default="results/suite_spec.json")
    parser.add_argument("--outdir", type=str, default="results/parquet")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = PROJECT_ROOT / spec_path
    spec = load_json(spec_path)

    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    games_rows, player_rows, message_rows, thought_rows = [], [], [], []

    for cond in spec["conditions"]:
        label = cond["label"]
        batch = load_json(RESULTS_DIR / cond["results"])
        judged = {}
        if cond.get("judged") and (RESULTS_DIR / cond["judged"]).exists():
            judged = {g["seed"]: g for g in load_json(RESULTS_DIR / cond["judged"])["games"]
                      if "error" not in g}

        for g in batch["games"]:
            if g["winner"] == "ERROR":
                continue
            seed, gid = g["seed"], g["game_id"]
            models = {}
            try:
                cfg = load_json(Path(g["config_file"]))
                models = dict(zip(cfg["names"], [m["model"] for m in cfg["models"]]))
            except FileNotFoundError:
                pass
            j = judged.get(seed, {})

            games_rows.append({
                "condition": label, "seed": seed, "game_id": gid,
                "winner": g["winner"], "killed": ",".join(g.get("killed_players", [])),
                "verdict": j.get("verdict"), "outcome_attribution": j.get("outcome_attribution"),
                "notability": j.get("notability"),
                "verdict_reason": j.get("verdict_reason"),
                "notability_reason": j.get("notability_reason"),
            })

            jp = j.get("players", {})
            for p in g["players"]:
                row = {
                    "condition": label, "seed": seed, "game_id": gid,
                    "player": p["name"], "model": models.get(p["name"]),
                    "starting_role": p["starting_role"], "ending_role": p["ending_role"],
                    "swap_count": p.get("swap_count"), "won": p["won"],
                    "believed_team": jp.get(p["name"], {}).get("believed_team"),
                    "justification": jp.get(p["name"], {}).get("justification"),
                }
                for dim in PLAYER_DIMS:
                    row[dim] = jp.get(p["name"], {}).get(dim)
                player_rows.append(row)

            log_path = LOG_DIR / f"game_{gid}.jsonl"
            if log_path.exists():
                with open(log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        e = json.loads(line)
                        d = e.get("data", {})
                        if e["event_type"] == "PUBLIC_MESSAGE":
                            message_rows.append({
                                "condition": label, "seed": seed, "game_id": gid,
                                "round": d.get("round"), "sender": d.get("sender"),
                                "model": models.get(d.get("sender")),
                                "content": d.get("content"),
                            })
                        elif e["event_type"] == "PLAYER_THOUGHT":
                            thought_rows.append({
                                "condition": label, "seed": seed, "game_id": gid,
                                "player": d.get("player_name"),
                                "model": models.get(d.get("player_name")),
                                "phase": d.get("phase"),
                                "prompt": d.get("prompt"),
                                "reasoning_summary": d.get("reasoning_summary"),
                                "response": d.get("response"),
                                "tool_calls": json.dumps(d.get("tool_calls")) if d.get("tool_calls") else None,
                            })
                        elif e["event_type"] == "VOTE_REASONING":
                            message_rows.append({
                                "condition": label, "seed": seed, "game_id": gid,
                                "round": -1, "sender": d.get("voter"),
                                "model": models.get(d.get("voter")),
                                "content": f"VOTE:{d.get('target')} | {d.get('reasoning') or ''}",
                            })

    tables = {
        "games": pd.DataFrame(games_rows),
        "players": pd.DataFrame(player_rows),
        "messages": pd.DataFrame(message_rows),
        "thoughts": pd.DataFrame(thought_rows),
    }
    for name, df in tables.items():
        path = outdir / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        print(f"{name}: {len(df):,} rows -> {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
