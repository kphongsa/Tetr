"""Watch the random agent play one seeded game in the terminal.

Run from the repo root:
    python -m scripts.play_random                 # seed 0, show first 5 boards
    python -m scripts.play_random --seed 7 --show 20
"""

import argparse

import numpy as np

from agents.random_agent import RandomAgent
from games.tetris.env import TetrisEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="game seed")
    parser.add_argument("--show", type=int, default=5, help="print the board after this many pieces")
    args = parser.parse_args()

    env = TetrisEnv()
    # [seed, 1] gives the agent a stream independent of the game's own
    # default_rng(seed) stream, while still being fully determined by --seed.
    agent = RandomAgent(np.random.default_rng([args.seed, 1]))

    obs, info = env.reset(seed=args.seed)
    print(f"seed {info['seed']}\n")
    print(env.render(), "\n")

    actions = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs, info)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if len(actions) <= args.show:
            print(f"piece #{len(actions)}: action {action}, "
                  f"{len(info['legal_actions'])} legal actions for the next piece, reward {reward}")
            print(env.render(), "\n")

    print("final board:")
    print(env.render())
    print(f"\npieces {info['pieces']}  score {info['score']}  lines {info['lines']}")
    print(f"replay = seed {info['seed']} + these {len(actions)} actions:\n{actions}")


if __name__ == "__main__":
    main()
