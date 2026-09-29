"""Describe HOW an agent plays, from saved replay files: stack height, holes, clear sizes.

Run from the repo root:
    python -m scripts.replay_style runs/full                 # every evaluation replay, seed 10100
    python -m scripts.replay_style runs/full --seed 10101
    python -m scripts.replay_style runs/full --max-pieces 2000

Each replay is re-simulated (seed + actions, CLAUDE.md rule 4), and after
every piece we measure the board. Because every evaluation saves the SAME
seeds, the rows are the same piece sequence played at different stages of
training, so differences are play style, not luck.

Columns
    step        training step when the replay was recorded
    pieces      pieces the game lasted
    avg/max h   average / worst height of the tallest column (20 = top of the visible
                board, 21-22 = the hidden spawn rows, i.e. about to lose)
    holes       average number of holes (empty cells with a block above them)
    bump        average bumpiness (sum of height differences between neighbour columns)
    1/2/3/4     share of line-clear events that cleared 1, 2, 3 or 4 lines at once
    pts/line    points per line: 100 = only singles, 200 = only tetrises

--max-pieces: only look at the first N pieces of each game, so a game that
lasted 10,000 pieces and one that lasted 200 are compared over the same stretch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.replay import load_replay
from games.tetris.env import TetrisEnv
from games.tetris.features import board_features, column_heights

ROOT = Path(__file__).resolve().parents[1]


def style(replay: dict, max_pieces: int | None = None) -> dict:
    """Play the replay back and summarize the board after every piece."""
    env = TetrisEnv()
    obs, info = env.reset(seed=replay["seed"])
    heights, holes, bumps = [], [], []
    clears = np.zeros(5, dtype=int)  # clears[k] = how many placements cleared k lines
    actions = replay["actions"][:max_pieces] if max_pieces else replay["actions"]
    for action in actions:
        obs, _, terminated, truncated, info = env.step(action)
        grid = obs["board"]
        h = column_heights(grid)
        feats = board_features(grid, 0)
        heights.append(int(h.max()))
        holes.append(feats[2])
        bumps.append(feats[3])
        clears[info["lines_cleared"]] += 1
        if terminated or truncated:
            break
    events = clears[1:].sum()
    lines = int((np.arange(5) * clears).sum())
    return {
        "step": replay["metadata"].get("step"),
        "pieces": len(heights),
        "avg_height": float(np.mean(heights)),
        "max_height": int(np.max(heights)),
        "holes": float(np.mean(holes)),
        "bumpiness": float(np.mean(bumps)),
        "clear_shares": (clears[1:] / events).tolist() if events else [0.0] * 4,
        "points_per_line": float(np.dot(clears[1:], [100, 300, 500, 800]) / lines) if lines else 0.0,
    }


def format_table(rows: list[dict]) -> str:
    out = [f"{'step':>9} {'pieces':>7} {'avg h':>6} {'max h':>6} {'holes':>6} {'bump':>6}"
           f" {'1':>5} {'2':>5} {'3':>5} {'4':>5} {'pts/line':>8}"]
    for r in rows:
        shares = " ".join(f"{100 * s:4.0f}%" for s in r["clear_shares"])
        out.append(f"{r['step']:>9,} {r['pieces']:>7,} {r['avg_height']:6.1f} {r['max_height']:6d} "
                   f"{r['holes']:6.2f} {r['bumpiness']:6.1f} {shares} {r['points_per_line']:8.0f}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=10_100)
    parser.add_argument("--max-pieces", type=int, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    paths = sorted((run_dir / "replays").glob(f"*_seed{args.seed}.json"))
    if not paths:
        raise SystemExit(f"no replays for seed {args.seed} in {run_dir / 'replays'}")
    print(f"seed {args.seed}, {len(paths)} replays" + (f", first {args.max_pieces} pieces" if args.max_pieces else ""))
    print(format_table([style(load_replay(p), args.max_pieces) for p in paths]))


if __name__ == "__main__":
    main()
