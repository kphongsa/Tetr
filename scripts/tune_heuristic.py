"""Tune the heuristic agent's 4 weights with the cross-entropy method (core/cem.py).

Run from the repo root:
    python -m scripts.tune_heuristic                     # defaults below
    python -m scripts.tune_heuristic --rounds 3 --population 10 --games 2   # quick smoke test

Each round, every candidate plays the SAME games (fair comparison), and each
round uses NEW games. All tuning games come from seeds 0..9999, never from the
evaluation seeds (10000+) used by scripts/evaluate.py. Otherwise the weights
could overfit: get good at the particular piece sequences they practised on
rather than at Tetris in general, and the evaluation would flatter them.

Fitness = mean score over the round's games, each capped at --max-pieces.
The cap keeps a run to minutes, but it changes the question slightly: it
rewards scoring well in the first N pieces, not surviving for thousands.
The long evaluation afterwards checks whether that transfers.

Writes results/tuned_weights.json (weights + full per-round history).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from agents.heuristic_agent import LEE_WEIGHTS, HeuristicAgent
from core.cem import CEMConfig, cem
from core.evaluate import evaluate, git_commit
from games.tetris.env import TetrisEnv
from games.tetris.features import FEATURE_NAMES
from scripts.evaluate import EVAL_FIRST_SEED, RESULTS_DIR

TUNE_FIRST_SEED = 0


def round_seeds(round_idx: int, games: int) -> list[int]:
    """Fresh seeds for each round; every candidate in that round plays these same games."""
    first = TUNE_FIRST_SEED + round_idx * games
    seeds = list(range(first, first + games))
    # Guard against ever tuning on evaluation games (see module docstring).
    assert seeds[-1] < EVAL_FIRST_SEED, "tuning seeds ran into the evaluation seeds"
    return seeds


def make_fitness(games: int, max_pieces: int):
    env = TetrisEnv()

    def fitness(candidates: np.ndarray, round_idx: int) -> np.ndarray:
        seeds = round_seeds(round_idx, games)
        scores = []
        for w in candidates:
            agent = HeuristicAgent(w)
            res = evaluate(env, lambda s: agent, seeds, max_pieces, info_keys=("score",))
            scores.append(res["summary"]["score"]["mean"])
        return np.array(scores)

    return fitness


def fmt(v) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--games", type=int, default=3, help="games per candidate per round")
    parser.add_argument("--max-pieces", type=int, default=300, help="piece cap per tuning game")
    parser.add_argument("--init", choices=["zero", "lee"], default="zero",
                        help="start the search from scratch (zero) or from Lee's weights")
    parser.add_argument("--seed", type=int, default=0, help="seed for CEM's own sampling")
    parser.add_argument("--out", default=str(RESULTS_DIR / "tuned_weights.json"))
    args = parser.parse_args()

    config = CEMConfig(rounds=args.rounds, population=args.population, elite_frac=args.elite_frac)
    init_mean = None if args.init == "zero" else np.array(LEE_WEIGHTS)
    print(f"CEM: {config}\n{args.games} games/candidate, cap {args.max_pieces} pieces, "
          f"features {FEATURE_NAMES}\n")
    print(f"{'round':>5} {'best':>9} {'mean':>9} {'elite':>9} {'secs':>6}  mean weights (unit length)")

    t_start = time.perf_counter()
    t_round = [t_start]

    def on_round(rec: dict) -> None:
        now = time.perf_counter()
        rec["seconds"] = now - t_round[0]
        t_round[0] = now
        print(f"{rec['round']:>5} {rec['best_score']:>9.0f} {rec['mean_score']:>9.0f}"
              f" {rec['elite_mean_score']:>9.0f} {rec['seconds']:>6.1f}  {fmt(rec['mean'])}", flush=True)

    rng = np.random.default_rng(args.seed)
    result = cem(make_fitness(args.games, args.max_pieces), len(FEATURE_NAMES), rng, config,
                 init_mean=init_mean, on_round=on_round)
    total = time.perf_counter() - t_start

    weights = result["mean"].tolist()
    print(f"\ndone in {total / 60:.1f} min. tuned weights (final mean): {fmt(weights)}")
    print(f"for comparison, Lee's weights (unit length):   {fmt(np.array(LEE_WEIGHTS) / np.linalg.norm(LEE_WEIGHTS))}")

    payload = {
        "description": "Heuristic weights tuned by CEM on seeds disjoint from the evaluation seeds.",
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "config": {**config.__dict__, "games_per_candidate": args.games, "max_pieces": args.max_pieces,
                   "init": args.init, "cem_seed": args.seed, "fitness": "mean score"},
        "tuning_seeds": f"round r uses seeds {TUNE_FIRST_SEED} + r*{args.games} .. +{args.games - 1}",
        "minutes": total / 60,
        "git": git_commit(),
        "history": result["history"],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
