"""Watch a recorded game in the terminal.

Run from the repo root:
    python -m scripts.watch runs/lr_decay/replays/step000086944_seed10100.json
    python -m scripts.watch <replay> --speed 50            # 50 pieces per second
    python -m scripts.watch <replay> --start 2000          # skip to piece 2,000
    python -m scripts.watch <replay> --clears-only         # only the line clears
    python -m scripts.watch <replay> --clears-only --min-clear 2   # doubles and up
    python -m scripts.watch <replay> --no-flash            # don't pause on line clears

Ctrl+C stops. The replay is re-simulated by the engine (and verified against
its recorded score) before anything is shown, then each frame is drawn with
the text renderer. When rows clear, the moment before the clear is shown
first: the new piece as @ and the full rows marked "<- clear".
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from games.tetris.render_text import render_frame
from scripts.export_frames import tetris_frames

# ANSI escape codes: move the cursor to the top-left, then clear the screen
# below it. Redrawing in place looks like animation; printing new lines
# would scroll forever.
HOME_AND_CLEAR = "\x1b[H\x1b[J"


def frames_to_show(frames: list[dict], start: int, end: int | None, clears_only: bool,
                   min_clear: int) -> list[int]:
    """Indices of the frames to draw, in order."""
    last = len(frames) - 1 if end is None else min(end, len(frames) - 1)
    idx = list(range(max(0, start), last + 1))
    if clears_only:
        # Keep line clears (big enough) plus the final frame, so you still
        # see how the game ended.
        idx = [i for i in idx if len(frames[i]["cleared_rows"]) >= min_clear or i == last]
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--speed", type=float, default=8.0, help="frames per second (default 8)")
    parser.add_argument("--start", type=int, default=0, help="first piece number to show")
    parser.add_argument("--end", type=int, default=None, help="last piece number to show")
    parser.add_argument("--clears-only", action="store_true", help="only show pieces that cleared lines")
    parser.add_argument("--min-clear", type=int, default=1, help="with --clears-only: smallest clear to show (1-4)")
    parser.add_argument("--no-flash", action="store_true", help="don't show the board before each clear")
    parser.add_argument("--no-hidden", action="store_true", help="hide the 2 spawn rows above the board")
    args = parser.parse_args()

    doc = tetris_frames(args.replay)
    frames = doc["frames"]
    meta = doc["metadata"]
    title = (f"{args.replay.name}  run {meta.get('run', '?')}  training step {meta.get('step', '?'):,}  "
             f"seed {doc['seed']}  ({len(frames) - 1:,} pieces)")
    show = frames_to_show(frames, args.start, args.end, args.clears_only, args.min_clear)
    delay = 1.0 / args.speed

    if os.name == "nt":
        os.system("")  # a no-op command that switches Windows consoles into ANSI-escape mode
    try:
        for i in show:
            frame, prev = frames[i], frames[i - 1] if i > 0 else None
            if frame["cleared_rows"] and prev and not args.no_flash:
                draw(title, render_frame(frame, prev, flash=True, show_hidden=not args.no_hidden))
                time.sleep(delay)
            draw(title, render_frame(frame, show_hidden=not args.no_hidden))
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nstopped")


def draw(title: str, body: str) -> None:
    sys.stdout.write(HOME_AND_CLEAR + title + "\n" + body + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
