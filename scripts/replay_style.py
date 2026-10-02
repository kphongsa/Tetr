"""Describe HOW an agent plays, from saved replay files: stack height, holes, clear sizes.

Run from the repo root:
    python -m scripts.replay_style runs/full                 # every evaluation replay, seed 10100
    python -m scripts.replay_style runs/full --seed 10101
    python -m scripts.replay_style runs/full --max-pieces 2000

Each replay is re-simulated (seed + actions, CLAUDE.md rule 4) and measured
after every piece with games/tetris/style.py (metric definitions there).
Because every evaluation saves the SAME seeds, the rows are the same piece
sequence played at different stages of training, so differences are play
style, not luck. For plots over training (and more games per step), see
scripts/style_report.py.

Columns
    step        training step when the replay was recorded
    pieces      pieces the game lasted
    avg/max h   average / worst height of the tallest column (20 = top of the visible
                board, 21-22 = the hidden spawn rows, i.e. about to lose)
    holes       average number of holes (empty cells with a block above them)
    bump        average bumpiness (sum of height differences between neighbour columns)
    1/2/3/4     share of cleared LINES that came from singles / doubles / triples / tetrises
    pcs/clear   pieces placed per line-clear event

--max-pieces: only look at the first N pieces of each game, so a game that
lasted 10,000 pieces and one that lasted 200 are compared over the same stretch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.replay import load_replay
from games.tetris.env import TetrisEnv
from games.tetris.style import StyleTracker

ROOT = Path(__file__).resolve().parents[1]


def style(replay: dict, max_pieces: int | None = None) -> dict:
    """Play the replay back and return its style metrics, plus its training step."""
    env = TetrisEnv()
    env.reset(seed=replay["seed"])
    tracker = StyleTracker()
    actions = replay["actions"][:max_pieces] if max_pieces else replay["actions"]
    for action in actions:
        obs, _, terminated, truncated, info = env.step(action)
        tracker.add(obs["board"], info["lines_cleared"])
        if terminated or truncated:
            break
    return {"step": replay["metadata"].get("step"), **tracker.metrics()}


def format_table(rows: list[dict]) -> str:
    out = [f"{'step':>9} {'pieces':>7} {'avg h':>6} {'max h':>6} {'holes':>6} {'bump':>6}"
           f" {'1':>5} {'2':>5} {'3':>5} {'4':>5} {'pcs/clear':>9}"]
    for r in rows:
        shares = " ".join(f"{100 * r[k]:4.0f}%" for k in
                          ("share_singles", "share_doubles", "share_triples", "share_tetrises"))
        ppc = f"{r['pieces_per_clear']:9.1f}" if r["pieces_per_clear"] is not None else f"{'-':>9}"
        out.append(f"{r['step']:>9,} {r['pieces']:>7,} {r['avg_stack_height']:6.1f} {r['max_stack_height']:6d} "
                   f"{r['avg_holes']:6.2f} {r['avg_bumpiness']:6.1f} {shares} {ppc}")
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
