"""Play N seeded games with the random agent, no rendering, and report speed + scores.

Run from the repo root:
    python -m scripts.benchmark                  # 1000 games, seeds 0..999
    python -m scripts.benchmark --games 5000 --seed 100

Game i uses seed (--seed + i), so the same command always plays the same games.
Only env.reset/step and agent.act are timed; nothing is printed inside the loop.
"""

import argparse
import time

import numpy as np

from agents.random_agent import RandomAgent
from games.tetris.env import TetrisEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="seed of the first game")
    parser.add_argument("--max-pieces", type=int, default=None,
                        help="truncate games after this many pieces (default: play to game over)")
    args = parser.parse_args()

    env = TetrisEnv(max_pieces=args.max_pieces)
    scores, lines, pieces = [], [], []

    start = time.perf_counter()
    for i in range(args.games):
        seed = args.seed + i
        agent = RandomAgent(np.random.default_rng([seed, 1]))
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(agent.act(obs, info))
        scores.append(info["score"])
        lines.append(info["lines"])
        pieces.append(info["pieces"])
    elapsed = time.perf_counter() - start

    total_pieces = sum(pieces)
    print(f"games          {args.games}  (seeds {args.seed}..{args.seed + args.games - 1})")
    print(f"total pieces   {total_pieces}")
    print(f"time           {elapsed:.2f} s")
    print(f"pieces/second  {total_pieces / elapsed:,.0f}")
    print(f"avg pieces     {np.mean(pieces):.1f}")
    print(f"avg score      {np.mean(scores):.1f}   (max {max(scores)})")
    print(f"avg lines      {np.mean(lines):.3f}   (max {max(lines)})")


if __name__ == "__main__":
    main()
