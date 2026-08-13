#!/usr/bin/env python3
"""LLM-judge batch game logs against the ONUW play-quality rubric.

For each game in a batch results file, renders the god view, the public
transcript, and every player's private thread, sends one judge call with
the rubric, and writes per-player scores plus a game verdict.

Usage:
    python experiments/16_judge_games.py results/08_batch_xxx.json
    python experiments/16_judge_games.py results/08_batch_xxx.json --judge-model deepseek/deepseek-v4-flash
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infra.llm_client import CachedLLMClient
from infra.onuw import ChatMessage

RUBRIC_PATH = PROJECT_ROOT / "docs" / "onuw_play_rubric.md"
LOG_DIR = PROJECT_ROOT / "game_logs"
RESULTS_DIR = PROJECT_ROOT / "results"
VIEWER_BASE = "http://127.0.0.1:9378"

DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4-flash"

PLAYER_DIMS = ["information_handling", "deception_quality", "deduction_quality",
               "vote_play", "reasoning_message_gap"]

OUTPUT_SCHEMA_INSTRUCTIONS = """\
Respond with ONLY a JSON object, no prose before or after, in exactly this shape:

{
  "players": {
    "<player name>": {
      "believed_team": "WEREWOLF" | "VILLAGE",
      "information_handling": <int 1-10>,
      "deception_quality": <int 1-10> | null,
      "deduction_quality": <int 1-10> | null,
      "vote_play": <int 1-10>,
      "reasoning_message_gap": <int 0-10> | null,
      "justification": "<one sentence citing at least one round number>"
    },
    ... one entry for EVERY player ...
  },
  "verdict": "skill_win" | "blunder_decided" | "luck_decided",
  "verdict_reason": "<one sentence>",
  "notability": <int 0-10>,
  "notability_reason": "<one sentence>"
}

Before writing your answer, do these checks:
1. Re-read the VOTES section. Any vote you mention in a justification or reason must appear
   there verbatim. Self-votes are impossible in this variant — never claim one happened.
2. Trace how each killed player came to hold their final card. If a killed player acquired
   the Werewolf card through the Drunk's blind swap or a Troublemaker swap, and the village
   did NOT explicitly reason about that swap when choosing the target, the verdict is
   luck_decided — not skill_win — no matter how good any individual play looked.
3. Check that verdict_reason is consistent with who actually won (the WINNER line).

Rules for the nullable fields, per the rubric:
- believed_team: the team this player privately believed they were on at the START of the
  day discussion (werewolves, the Minion, and a Robber who robbed a wolf believe WEREWOLF).
- deception_quality: score only players whose believed_team is WEREWOLF; else null.
- deduction_quality: score only players whose believed_team is VILLAGE; else null.
- reasoning_message_gap: null when that player has no private reasoning in the log.
"""


def load_game_events(game_id: str) -> list[dict]:
    log_path = LOG_DIR / f"game_{game_id}.jsonl"
    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def render_game(events: list[dict]) -> str:
    """Render god view + public transcript + private threads for the judge."""
    god: list[str] = []
    public: list[str] = []
    private: dict[str, list[str]] = {}
    votes: list[str] = []
    result: list[str] = []
    models: dict[str, str] = {}

    current_round = 0
    for e in events:
        et, d = e.get("event_type", ""), e.get("data", {})

        if et == "GAME_INIT":
            god.append("DEAL (starting roles):")
            for p in d.get("players", []):
                models[p["name"]] = p.get("model", "?")
                god.append(f"  {p['name']}: {p['original_role']} (model: {p.get('model', '?')})")
            god.append(f"  Center: {d.get('center_cards')}")

        elif et == "NIGHT_ACTION":
            god.append(f"NIGHT: {d.get('player')} ({d.get('role')}) -> "
                       f"{d.get('action')} {d.get('targets')} = {d.get('result')}")

        elif et == "CARD_SWAP":
            god.append(f"NIGHT SWAP: {d.get('location1')} <-> {d.get('location2')}")

        elif et == "PUBLIC_MESSAGE":
            rnd = d.get("round", 0)
            if rnd != current_round:
                current_round = rnd
                public.append(f"\n--- ROUND {rnd} (all messages below were written "
                              f"simultaneously, blind to each other) ---")
            public.append(f"{d.get('sender')}: {d.get('content')}")

        elif et == "VOTE_REASONING":
            votes.append(f"{d.get('voter')} voted {d.get('target')}. "
                         f"Private reasoning: {d.get('reasoning') or '(none)'}")

        elif et == "PLAYER_THOUGHT":
            name = d.get("player_name", "?")
            phase = d.get("phase", "?")
            parts = [f"[{phase}]"]
            if phase == "NIGHT":
                parts.append(f"(night info given: {json.dumps(d.get('prompt', ''))[:600]})")
            if d.get("reasoning_summary"):
                parts.append(f"private reasoning: {d['reasoning_summary']}")
            if d.get("response"):
                parts.append(f"private text: {d['response']}")
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function", {})
                parts.append(f"action: {fn.get('name')}({fn.get('arguments', '')[:400]})")
            private.setdefault(name, []).append("\n  ".join(parts))

        elif et == "GAME_END":
            result.append(f"WINNER: {d.get('winner')}   KILLED: {d.get('killed')}")
            result.append("FINAL ROLES: " + ", ".join(
                f"{p['name']}={p['current_role']}(started {p['original_role']})"
                for p in d.get("players", [])))
            result.append(f"FINAL CENTER: {d.get('center_cards')}")

    sections = [
        "=== GOD VIEW (ground truth — the players did NOT see this) ===",
        "\n".join(god),
        "\n=== OUTCOME ===",
        "\n".join(result),
        "\n=== PUBLIC DISCUSSION ===",
        "\n".join(public),
        "\n=== VOTES (cast simultaneously) ===",
        "\n".join(votes),
        "\n=== PRIVATE THREADS (each player's hidden reasoning, per phase) ===",
    ]
    for name in sorted(private):
        sections.append(f"\n--- {name} (model: {models.get(name, '?')}) ---")
        sections.append("\n".join(private[name]))
    return "\n".join(sections)


def extract_json(text: str) -> dict:
    """Parse the judge's JSON, tolerating code fences and surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in judge response")
    return json.loads(text[start:end + 1])


def validate_judgment(j: dict, player_names: set[str]) -> list[str]:
    problems = []
    got = set(j.get("players", {}).keys())
    if got != player_names:
        problems.append(f"player set mismatch: expected {sorted(player_names)}, got {sorted(got)}")
    for name, row in j.get("players", {}).items():
        team = row.get("believed_team")
        if team not in ("WEREWOLF", "VILLAGE"):
            problems.append(f"{name}: bad believed_team {team!r}")
        for dim in PLAYER_DIMS:
            v = row.get(dim)
            if v is not None and not (isinstance(v, int) and 0 <= v <= 10):
                problems.append(f"{name}: bad {dim} value {v!r}")
        if team == "WEREWOLF" and row.get("deduction_quality") is not None:
            problems.append(f"{name}: deduction_quality must be null for WEREWOLF believer")
        if team == "VILLAGE" and row.get("deception_quality") is not None:
            problems.append(f"{name}: deception_quality must be null for VILLAGE believer")
    if j.get("verdict") not in ("skill_win", "blunder_decided", "luck_decided"):
        problems.append(f"bad verdict {j.get('verdict')!r}")
    if not (isinstance(j.get("notability"), int) and 0 <= j["notability"] <= 10):
        problems.append(f"bad notability {j.get('notability')!r}")
    return problems


async def judge_game(client: CachedLLMClient, judge_model: str, rubric: str,
                     game: dict) -> dict:
    game_id = game["game_id"]
    events = load_game_events(game_id)
    rendering = render_game(events)
    player_names = {p["name"] for p in game["players"]}

    system = ("You are an expert analyst of One Night Ultimate Werewolf played by LLM "
              "agents. Score play quality strictly according to the rubric below. "
              "Score decisions against what each player could know, never against "
              "ground truth they could not see.\n\n" + rubric)
    user = rendering + "\n\n=== YOUR TASK ===\n" + OUTPUT_SCHEMA_INSTRUCTIONS

    messages = [ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user)]

    last_problems: list[str] = []
    for attempt in range(2):
        response = await client.chat_completion(
            model=judge_model, messages=messages, temperature=1)
        try:
            judgment = extract_json(response.content or "")
            last_problems = validate_judgment(judgment, player_names)
            if not last_problems:
                break
        except (ValueError, json.JSONDecodeError) as e:
            last_problems = [str(e)]
        if attempt == 0:
            messages = messages + [
                ChatMessage(role="assistant", content=response.content or ""),
                ChatMessage(role="user", content=(
                    "Your response had these problems: " + "; ".join(last_problems)
                    + ". Respond again with ONLY the corrected JSON object.")),
            ]
    else:
        return {"game_id": game_id, "seed": game.get("seed"), "error": last_problems}

    # attach ground truth for aggregation
    for p in game["players"]:
        row = judgment["players"].get(p["name"])
        if row is not None:
            row["starting_role"] = p["starting_role"]
            row["ending_role"] = p["ending_role"]
            row["won"] = p["won"]

    return {
        "game_id": game_id,
        "seed": game.get("seed"),
        "winner": game.get("winner"),
        "viewer_url": f"{VIEWER_BASE}/game/{game_id}",
        **judgment,
    }


def flag_outliers(judged: list[dict]) -> None:
    """Print notability ranking and per-dimension outliers (z within dimension)."""
    ok = [j for j in judged if "error" not in j]
    print(f"\n{'=' * 70}\nNOTABILITY RANKING ({len(ok)} games)\n{'=' * 70}")
    for j in sorted(ok, key=lambda x: -x.get("notability", 0)):
        print(f"  [{j['notability']:>2}] seed={j['seed']:<3} {j['verdict']:<16} "
              f"{j['notability_reason']}\n        {j['viewer_url']}")

    print(f"\n{'=' * 70}\nSCORE OUTLIERS (|z| >= 1.5 within dimension)\n{'=' * 70}")
    for dim in PLAYER_DIMS:
        values = []
        for j in ok:
            for name, row in j["players"].items():
                if isinstance(row.get(dim), int):
                    values.append((j, name, row))
        if len(values) < 4:
            continue
        scores = [row[dim] for _, _, row in values]
        mu = mean(scores)
        sd = stdev(scores)
        if sd == 0:
            continue
        for j, name, row in values:
            z = (row[dim] - mu) / sd
            if abs(z) >= 1.5:
                print(f"  {dim}: {name}={row[dim]} (z={z:+.1f}) seed={j['seed']} "
                      f"started={row.get('starting_role', '?')} winner={j['winner']}\n"
                      f"      {row.get('justification', '')}\n      {j['viewer_url']}")


async def main():
    parser = argparse.ArgumentParser(description="Judge batch games against the rubric")
    parser.add_argument("results_file", type=str)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--parallel", "-p", type=int, default=4)
    parser.add_argument("--max-games", "-m", type=int, default=None)
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
    with open(results_path) as f:
        batch = json.load(f)
    games = [g for g in batch["games"] if g["winner"] != "ERROR"]
    if args.max_games:
        games = games[:args.max_games]

    rubric = RUBRIC_PATH.read_text()
    client = CachedLLMClient(use_cache=True)
    sem = asyncio.Semaphore(args.parallel)

    async def run_one(game):
        async with sem:
            print(f"  judging seed={game.get('seed')} ({game['game_id']})...")
            return await judge_game(client, args.judge_model, rubric, game)

    judged = await asyncio.gather(*[run_one(g) for g in games])
    judged = sorted(judged, key=lambda j: j.get("seed", 0))

    errors = [j for j in judged if "error" in j]
    for j in errors:
        print(f"  JUDGE FAILED seed={j['seed']}: {j['error']}")

    out_path = RESULTS_DIR / f"judged_{results_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump({"judge_model": args.judge_model,
                   "source": results_path.name,
                   "rubric": str(RUBRIC_PATH.name),
                   "games": judged}, f, indent=2)
    print(f"\nSaved {out_path}  ({len(judged) - len(errors)} judged, {len(errors)} failed)")

    flag_outliers(judged)


if __name__ == "__main__":
    asyncio.run(main())
