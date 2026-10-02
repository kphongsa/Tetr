""""What was the AI thinking": add the network's top-k options to an exported game.

Run from the repo root:
    python -m scripts.thoughts runs/lr_decay/replays/step000086944_seed10100.json -o thoughts.json
    python -m scripts.thoughts <replay> --checkpoint runs/lr_decay/checkpoints/best.pt -k 5 -o out.json

(scripts/build_site.py does this automatically for the web viewer.)

How it works: load the checkpoint saved at the SAME training step as the
replay, re-play the replay, and before every piece ask the agent to rank
the legal placements. Each frame then gets

    "options": [{"a": action id, "v": Q value, "cells": [[row, col], ...]}, ...]

= the k best placements for the piece about to be placed, best first.
Q = 0.01 x points scored by the placement + 0.99 x the network's value of
the board it leaves (exactly what the greedy agent maximizes). Only the
differences between options mean something.

Safety check: the agent's #1 option must be the move that was actually
played. If not (wrong checkpoint, changed code), we stop instead of
showing invented "thoughts".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.checkpoint import load_checkpoint
from core.replay import load_replay
from games.tetris.env import TetrisEnv
from scripts.export_frames import tetris_frames

DEFAULT_K = 4


def checkpoint_for_step(run_dir: Path, step: int) -> Path | None:
    """The checkpoint saved at exactly this training step, if it still exists."""
    exact = run_dir / "checkpoints" / f"step{step:09d}.pt"
    if exact.exists():
        return exact
    # Runs without per-evaluation snapshots still have best.pt / latest.pt;
    # they match when the replay came from that same evaluation.
    for name in ("best.pt", "latest.pt"):
        path = run_dir / "checkpoints" / name
        if path.exists() and load_checkpoint(path)["progress"]["step"] == step:
            return path
    return None


def record_options(replay: dict, agent, k: int = DEFAULT_K) -> list[list[dict]]:
    """For each move of the replay: the agent's top-k options (see module docstring)."""
    env = TetrisEnv()
    obs, info = env.reset(seed=replay["seed"])
    out = []
    for i, action in enumerate(replay["actions"]):
        ranked = agent.ranked_choices(obs, info, k)
        if ranked[0][0] != action:
            raise ValueError(f"piece {i}: agent's top choice is action {ranked[0][0]}, "
                             f"but the replay played {action}; wrong checkpoint for this replay?")
        cells = {p.action: p.cells for p in info["placements"]}
        out.append([{"a": a, "v": round(q, 3), "cells": [list(rc) for rc in cells[a]]} for a, q in ranked])
        obs, _, _, _, info = env.step(action)
    return out


def add_thoughts(doc: dict, replay: dict, agent, checkpoint: Path, k: int = DEFAULT_K) -> dict:
    """Attach options to a frames document in place (frame i = board before piece i+1)."""
    for frame, options in zip(doc["frames"], record_options(replay, agent, k)):
        frame["options"] = options
    doc["thoughts"] = {
        "checkpoint": checkpoint.name,
        "k": k,
        "value": "0.01 x points scored + 0.99 x network's value of the resulting board",
    }
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="default: <run>/checkpoints/ at the replay's training step")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    from scripts.train import load_eval_agent  # imports torch

    replay = load_replay(args.replay)
    checkpoint = args.checkpoint or checkpoint_for_step(args.replay.parent.parent, replay["metadata"]["step"])
    if checkpoint is None:
        raise SystemExit("no checkpoint from this replay's training step; pass --checkpoint")
    agent, _ = load_eval_agent(checkpoint)
    doc = add_thoughts(tetris_frames(replay), replay, agent, checkpoint, args.k)
    args.out.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"{len(replay['actions']):,} moves with top-{args.k} options from {checkpoint} -> {args.out}")


if __name__ == "__main__":
    main()
