"""Compare agents on the SAME seeded games and save the heuristic baseline.

Run from the repo root:
    python -m scripts.evaluate                       # 100 games, seeds 10000..10099, cap 10000
    python -m scripts.evaluate --games 20            # quicker, noisier
    python -m scripts.evaluate --no-save             # don't touch results/
    python -m scripts.evaluate --tuned               # also evaluate results/tuned_weights.json

Prints one table (random vs heuristic) and writes results/baseline_heuristic.json
with the seeds, cap, weights and git commit. For a traceable result, commit
your code first: the file records the commit and whether the tree was "dirty".

EVAL_SEEDS are the official evaluation games. Tuning (step 2d) must use
DIFFERENT seeds, or the tuned weights could just be memorizing these games.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.heuristic_agent import LEE_WEIGHTS, HeuristicAgent
from agents.random_agent import RandomAgent
from core.evaluate import evaluate, git_commit
from games.tetris.env import TetrisEnv
from games.tetris.features import FEATURE_NAMES

EVAL_FIRST_SEED = 10_000
EVAL_GAMES = 100
EVAL_MAX_PIECES = 10_000  # ~3x the longest game Lee's weights played in 30 test games
METRICS = ("lines", "score", "pieces")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def print_table(results: dict[str, dict]) -> None:
    """One table: for each metric, one row per agent, so agents sit next to each other."""
    header = f"{'metric':<7} {'agent':<16} {'mean':>10} {'median':>10} {'std':>10} {'min':>9} {'max':>9}"
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        for i, (name, res) in enumerate(results.items()):
            s = res["summary"][metric]
            label = metric if i == 0 else ""
            print(f"{label:<7} {name:<16} {s['mean']:>10.1f} {s['median']:>10.1f} {s['std']:>10.1f}"
                  f" {s['min']:>9.0f} {s['max']:>9.0f}")
        print()
    print(f"{'agent':<16} {'capped':>8} {'games/s':>9} {'pieces/s':>9} {'seconds':>8}")
    for name, res in results.items():
        s = res["summary"]
        print(f"{name:<16} {s['capped']:>4}/{s['games']:<3} {s['games_per_second']:>9.2f}"
              f" {s['steps_per_second']:>9.0f} {s['seconds']:>8.1f}")


def warn_if_capped(results: dict[str, dict]) -> None:
    for name, res in results.items():
        s = res["summary"]
        if s["capped"] > s["games"] / 2:
            print(f"\nWARNING: {name} hit the {s['max_steps']}-piece cap in {s['capped']}/{s['games']}"
                  " games. The cap is too low to tell strong agents apart; raise --max-pieces.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=EVAL_GAMES)
    parser.add_argument("--seed", type=int, default=EVAL_FIRST_SEED, help="seed of the first game")
    parser.add_argument("--max-pieces", type=int, default=EVAL_MAX_PIECES)
    parser.add_argument("--no-save", action="store_true", help="don't write results/baseline_heuristic.json")
    parser.add_argument("--tuned", nargs="?", const=str(RESULTS_DIR / "tuned_weights.json"), default=None,
                        help="also evaluate tuned weights from this JSON (default path if given without a value)")
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.games))
    env = TetrisEnv()  # no env-level cap: core's StepLimit applies --max-pieces

    # Heuristic candidates: name -> (weights, where they came from).
    heuristics = {"heuristic (Lee)": (LEE_WEIGHTS, "Lee 2013")}
    if args.tuned:
        tuned = json.loads(Path(args.tuned).read_text())
        assert tuned["feature_names"] == list(FEATURE_NAMES), "tuned weights use different features"
        heuristics["heuristic (tuned)"] = (tuned["weights"], f"CEM ({Path(args.tuned).name})")

    agents = {
        # The random agent gets a per-game rng derived from the game seed
        # (the [seed, 1] pair keeps it distinct from the game's own rng stream).
        "random": lambda seed: RandomAgent(np.random.default_rng([seed, 1])),
    }
    for name, (weights, _) in heuristics.items():
        # Heuristics are deterministic, so one instance can play every game.
        # (agent=... binds THIS agent now; a plain closure would see the last one.)
        agent = HeuristicAgent(weights)
        agents[name] = lambda seed, agent=agent: agent

    print(f"evaluating on {len(seeds)} games, seeds {seeds[0]}..{seeds[-1]}, cap {args.max_pieces} pieces\n")
    results = {name: evaluate(env, make, seeds, args.max_pieces, METRICS) for name, make in agents.items()}
    print_table(results)
    warn_if_capped(results)

    # The baseline is the best heuristic by mean score (the env's reward, so
    # it's what a learning agent will be maximizing too).
    best = max(heuristics, key=lambda n: results[n]["summary"]["score"]["mean"])
    print(f"\nbest heuristic by mean score: {best}")

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / "baseline_heuristic.json"
        weights, source = heuristics[best]
        payload = {
            "description": "Rule-based baseline on the evaluation seeds. This is the score to beat.",
            "agent": best,
            "weights_source": source,
            "feature_names": list(FEATURE_NAMES),
            "weights": [float(w) for w in weights],
            "seeds": seeds,
            "max_pieces": args.max_pieces,
            "git": git_commit(),
            "summary": results[best]["summary"],
            "other_summaries": {n: r["summary"] for n, r in results.items() if n != best},
            "games": results[best]["games"],
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"saved {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
