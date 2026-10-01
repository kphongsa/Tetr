"""Check that saved replays re-simulate to exactly the recorded score, lines and pieces.

Run from the repo root:
    python -m scripts.verify_replays                 # every replay under runs/
    python -m scripts.verify_replays runs/lr_decay   # one run

Why: a replay stores only seed + actions (CLAUDE.md rule 4). If the engine's
rules or randomizer ever change, old replays would quietly play out a
different game. This catches that. The recorded git commit (metadata "git")
tells you which code to diff against if something fails.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from core.replay import load_replay, verify_replay
from games.tetris.env import TetrisEnv

ROOT = Path(__file__).resolve().parents[1]


def replay_paths(where: Path) -> list[Path]:
    # "**" also matches zero folders, so this works for runs/ and runs/<name>.
    return sorted(where.glob("**/replays/*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("where", type=Path, nargs="?", default=ROOT / "runs")
    args = parser.parse_args()
    paths = replay_paths(args.where)
    if not paths:
        raise SystemExit(f"no replays found under {args.where}")
    t0 = time.perf_counter()
    failed = 0
    for path in paths:
        problems = verify_replay(TetrisEnv(), load_replay(path))
        if problems:
            failed += 1
            print(f"FAIL {path}: {'; '.join(problems)}")
    print(f"{len(paths) - failed}/{len(paths)} replays verify ({time.perf_counter() - t0:.0f} s)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
