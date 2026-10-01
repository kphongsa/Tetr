"""Export a replay (seed + actions) to a frames JSON file for viewers.

Run from the repo root:
    python -m scripts.export_frames runs/lr_decay/replays/step000086944_seed10100.json out.json

The game is re-simulated by the Python engine and verified against the
numbers recorded in the replay; each frame is the board after one piece
(format documented in games/tetris/frames.py). Other scripts (watch,
make_gif, build_site) call tetris_frames() directly instead of writing a file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.replay import export_frames, load_replay
from games.tetris.env import TetrisEnv
from games.tetris.frames import TetrisFrameRecorder


def tetris_frames(replay: dict | Path) -> dict:
    """Replay dict or path -> frames document. Raises if the replay doesn't verify."""
    if not isinstance(replay, dict):
        replay = load_replay(Path(replay))
    return export_frames(TetrisEnv(), replay, TetrisFrameRecorder())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    doc = tetris_frames(args.replay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # separators without spaces: ~10% smaller, and nobody reads this by hand.
    args.out.write_text(json.dumps(doc, separators=(",", ":")))
    size = args.out.stat().st_size
    print(f"{len(doc['frames']) - 1:,} pieces -> {args.out} ({size / 1e6:.2f} MB, {size / len(doc['frames']):.0f} B/frame)")


if __name__ == "__main__":
    main()
