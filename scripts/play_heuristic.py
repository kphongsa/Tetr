"""Watch the heuristic agent play one seeded game in the terminal.

Run from the repo root:
    python -m scripts.play_heuristic                          # seed 0, board every 50 pieces
    python -m scripts.play_heuristic --seed 3 --every 10 --max-pieces 200
    python -m scripts.play_heuristic --every 1 --max-pieces 30 --explain

A good heuristic can play for a very long time, so games are capped at
--max-pieces (the env reports this as "truncated", not a loss).
"""

import argparse
import time

from agents.heuristic_agent import HeuristicAgent
from games.tetris.env import TetrisEnv
from games.tetris.placements import decode_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="game seed")
    parser.add_argument("--every", type=int, default=50, help="print the board every N pieces")
    parser.add_argument("--max-pieces", type=int, default=500, help="stop the game after this many pieces")
    parser.add_argument("--explain", action="store_true",
                        help="also print the top 3 placements and their scores each time")
    args = parser.parse_args()

    env = TetrisEnv(max_pieces=args.max_pieces)
    agent = HeuristicAgent()

    obs, info = env.reset(seed=args.seed)
    print(f"seed {info['seed']}, weights {tuple(agent.weights)}\n")

    start = time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated):
        if args.explain and (info["pieces"] + 1) % args.every == 0:
            top = sorted(agent.score_placements(obs, info), key=lambda x: -x[1])[:3]
            desc = ", ".join(f"rot {decode_action(a)[0]} col {decode_action(a)[1]}: {s:.2f}"
                             for a, s in top)
            print(f"top choices for the next piece: {desc}")
        obs, reward, terminated, truncated, info = env.step(agent.act(obs, info))
        if info["pieces"] % args.every == 0:
            print(f"after piece #{info['pieces']} (reward {reward})")
            print(env.render(), "\n")
    elapsed = time.perf_counter() - start

    how = "GAME OVER" if terminated else f"stopped at the {args.max_pieces}-piece cap"
    print("final board:")
    print(env.render())
    print(f"\n{how}: pieces {info['pieces']}  score {info['score']}  lines {info['lines']}")
    print(f"{info['pieces'] / elapsed:,.0f} pieces/second")


if __name__ == "__main__":
    main()
