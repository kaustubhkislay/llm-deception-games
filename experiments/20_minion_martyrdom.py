#!/usr/bin/env python3
"""Minion self-sacrifice ("martyrdom") attempts and successes per condition.

Reads the parquet export (18_export_parquet.py). Only games where a player is
dealt the Minion count; in the other seeds the Minion card is in the center.

- attempt: the starting Minion's private reasoning summary contains sacrifice
  language (SACRIFICE_PATTERN). Upper bound: a hand check of 20 random matches
  found ~16 true intent; misses are negations, other players targeting the
  Minion, and Minions describing a teammate's sacrifice.
- success: wolf-team win, the Minion was lynched still holding the MINION card,
  the judge verdict is skill_win (the judge's forced check requires quoted
  intent), and the attempt pattern matched.

Conversion (successes / attempts) is compared against the flash baseline with
Fisher's exact test. Conditions whose Minion is gpt-5-mini (B1, C1) are
reported but not compared: gpt-5-mini returns a reasoning summary on only ~23%
of calls, so most of its intent is invisible.

Usage:
    python experiments/20_minion_martyrdom.py
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
PARQUET_DIR = RESULTS_DIR / "parquet"

BASELINE_PREFIX = "B2"
LOW_VISIBILITY_PREFIXES = ("B1", "C1")  # gpt-5-mini Minion, mostly no reasoning summary

SACRIFICE_PATTERN = re.compile(
    r"sacrific|martyr|self-lynch|take the fall"
    r"|(?:get|getting|have|having) (?:myself|me) (?:lynched|voted|killed|eliminated)"
    r"|(?:draw|drawing|attract|attracting|court|courting|absorb|absorbing|redirect|redirecting"
    r"|direct|directing)\w* (?:the )?(?:vote|votes|suspicion|lynch|fire|heat|attention)s? "
    r"(?:to|onto|toward|towards|on) (?:myself|me)"
    r"|vote (?:for )?me\b|lynch me|eliminate me|kill me",
    re.I,
)


def load_minion_games() -> pd.DataFrame:
    games = pd.read_parquet(PARQUET_DIR / "games.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    thoughts = pd.read_parquet(PARQUET_DIR / "thoughts.parquet")

    minions = (players[players.starting_role == "MINION"]
               [["game_id", "player", "ending_role", "model"]]
               .rename(columns={"player": "minion", "ending_role": "minion_end",
                                "model": "minion_model"}))
    minion_thoughts = thoughts.merge(minions, left_on=["game_id", "player"],
                                     right_on=["game_id", "minion"])
    summaries = minion_thoughts.reasoning_summary.fillna("")
    intent_games = set(minion_thoughts[summaries.str.contains(SACRIFICE_PATTERN)].game_id)
    coverage = minion_thoughts.groupby("game_id").reasoning_summary.apply(
        lambda s: s.notna().mean())

    df = games.merge(minions, on="game_id")
    df["attempt"] = df.game_id.isin(intent_games)
    df["minion_lynched"] = [m in str(k).split(",") for m, k in zip(df.minion, df.killed)]
    df["minion_death_win"] = ((df.winner == "WEREWOLF") & df.minion_lynched
                              & (df.minion_end == "MINION"))
    df["success"] = df.minion_death_win & (df.verdict == "skill_win") & df.attempt
    df["summary_coverage"] = df.game_id.map(coverage)
    return df


def fisher_vs(cond: pd.DataFrame, base: pd.DataFrame) -> float:
    table = [[cond.success.sum(), cond.attempt.sum() - cond.success.sum()],
             [base.success.sum(), base.attempt.sum() - base.success.sum()]]
    return scipy_stats.fisher_exact(table)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="minion_martyrdom")
    args = parser.parse_args()

    df = load_minion_games()
    base = df[df.condition.str.startswith(BASELINE_PREFIX)]
    comparable = ~df.condition.str.startswith(LOW_VISIBILITY_PREFIXES + (BASELINE_PREFIX,))

    rows = []
    for cond in dict.fromkeys(df.condition):
        x = df[df.condition == cond]
        compared = not cond.startswith(LOW_VISIBILITY_PREFIXES + (BASELINE_PREFIX,))
        rows.append({
            "label": cond,
            "minion_games": len(x),
            "minion_model": x.minion_model.mode()[0],
            "summary_coverage": round(float(x.summary_coverage.mean()), 2),
            "attempts": int(x.attempt.sum()),
            "minion_death_wins": int(x.minion_death_win.sum()),
            "successes": int(x.success.sum()),
            "conversion": float(x.success.sum() / x.attempt.sum()) if x.attempt.sum() else None,
            "fisher_p_vs_baseline": fisher_vs(x, base) if compared else None,
        })
    pooled = df[comparable]
    summary = {
        "attempts_total": int(df.attempt.sum()),
        "minion_games_total": len(df),
        "successes_total": int(df.success.sum()),
        "judge_only_successes_total": int((df.minion_death_win & (df.verdict == "skill_win")).sum()),
        "pooled_upgrades": {"attempts": int(pooled.attempt.sum()),
                            "successes": int(pooled.success.sum()),
                            "fisher_p_vs_baseline": fisher_vs(pooled, base)},
    }

    print(f"{'condition':30} {'games':>5} {'cov':>5} {'att':>4} {'succ':>4} {'conv':>5} {'p':>6}")
    for r in rows:
        conv = f"{r['conversion']:.2f}" if r["conversion"] is not None else "-"
        p = f"{r['fisher_p_vs_baseline']:.3f}" if r["fisher_p_vs_baseline"] is not None else "-"
        print(f"{r['label'][:30]:30} {r['minion_games']:5d} {r['summary_coverage']:5.2f} "
              f"{r['attempts']:4d} {r['successes']:4d} {conv:>5} {p:>6}")
    pu = summary["pooled_upgrades"]
    print(f"\npooled upgrades (flash Minion): {pu['successes']}/{pu['attempts']} "
          f"vs baseline {int(base.success.sum())}/{int(base.attempt.sum())}, "
          f"p={pu['fisher_p_vs_baseline']:.3f}")
    print(f"attempts {summary['attempts_total']} of {summary['minion_games_total']} Minion games; "
          f"successes {summary['successes_total']} "
          f"(judge-only {summary['judge_only_successes_total']})")

    per_seed = (df[comparable | df.condition.str.startswith(BASELINE_PREFIX)]
                [["condition", "seed", "game_id", "minion", "attempt", "success"]])
    with open(RESULTS_DIR / f"{args.output}.json", "w") as f:
        json.dump({"rows": rows, "summary": summary,
                   "games": per_seed.to_dict(orient="records")}, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nwrote results/{args.output}.json")


if __name__ == "__main__":
    main()
