"""Train the afterstate value network on Tetris by self-play.

Run from the repo root:
    python -m scripts.train --config configs/smoke.json            # ~1 minute sanity run
    python -m scripts.train --config configs/smoke.json --fresh    # same, deleting an old runs/smoke/
    python -m scripts.train --config configs/smoke.json --name try2

Writes runs/<name>/{config.json, log.csv, tb/}. Watch live with
    tensorboard --logdir runs
then open http://localhost:6006 in a browser.

This script is the only place where Tetris and the generic parts meet:
it builds the Tetris env and the Tetris candidates function and hands
them to the game-agnostic agent, learner and training loop.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agents.afterstate_value_agent import AfterstateTDLearner, AfterstateValueAgent, TDConfig, make_value_net
from core.train import LoopConfig, format_config, load_config, make_run_dir, train
from games.tetris.env import TetrisEnv
from games.tetris.features import FEATURE_NAMES, candidates
from scripts.evaluate import EVAL_FIRST_SEED, EVAL_GAMES

ROOT = Path(__file__).resolve().parents[1]
INFO_KEYS = ("lines", "score", "pieces")


def build(config: dict) -> tuple[TetrisEnv, AfterstateTDLearner, LoopConfig]:
    """Turn a config dict into ready-to-train objects. Unknown keys raise TypeError."""
    loop = LoopConfig(**config["loop"])
    td_kwargs = dict(config["learner"])
    td_kwargs["hidden"] = tuple(td_kwargs.get("hidden", TDConfig.hidden))  # JSON lists -> tuple
    td = TDConfig(**td_kwargs)

    # Training must never play the evaluation games (that would be training on the test).
    if loop.train_seed_start <= EVAL_FIRST_SEED + EVAL_GAMES - 1:
        raise ValueError(f"train_seed_start must be above the evaluation seeds "
                         f"({EVAL_FIRST_SEED}..{EVAL_FIRST_SEED + EVAL_GAMES - 1})")

    torch.set_num_threads(config.get("torch_threads", 4))
    env = TetrisEnv()  # the loop's StepLimit applies max_episode_steps
    net = make_value_net(len(FEATURE_NAMES), td.hidden, seed=td.net_seed)
    agent = AfterstateValueAgent(net, candidates, np.random.default_rng(td.rng_seed),
                                 gamma=td.gamma, reward_scale=td.reward_scale)
    learner = AfterstateTDLearner(agent, td, n_features=len(FEATURE_NAMES), max_candidates=env.num_actions)
    return env, learner, loop


def full_config(config: dict) -> dict:
    """The config with every default filled in, so the saved copy is complete."""
    env, learner, loop = build(config)
    out = copy.deepcopy(config)
    out["loop"] = asdict(loop)
    out["learner"] = asdict(learner.cfg)
    out.setdefault("torch_threads", 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", help="run name (default: the config's 'name')")
    parser.add_argument("--fresh", action="store_true", help="delete runs/<name>/ first if it exists")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.name:
        config["name"] = args.name
    config = full_config(config)
    print(f"run {config['name']!r}, config:\n{format_config(config)}\n")

    run_dir = make_run_dir(ROOT / "runs", config["name"], config, fresh=args.fresh)
    env, learner, loop = build(config)
    print(f"replay buffer: {learner.buffer.nbytes() / 1e6:.0f} MB, "
          f"network: {sum(p.numel() for p in learner.net.parameters())} parameters\n")
    train(env, learner, loop, run_dir, INFO_KEYS, tensorboard=not args.no_tensorboard)
    print(f"\nlogs in {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
