"""Make a GIF (or MP4) of a recorded game, or of two games side by side.

Run from the repo root:
    # pieces 0-300 of one game, 10 pieces per second
    python -m scripts.make_gif runs/lr_decay/replays/step000086944_seed10100.json -o best.gif

    # early vs late training on the same seed, side by side
    python -m scripts.make_gif runs/lr_decay/replays/step000002581_seed10100.json \
        --compare runs/lr_decay/replays/step000086944_seed10100.json -o early_vs_late.gif

    # a whole 3,000-piece game as a video, 30 pieces per second
    python -m scripts.make_gif <replay> --end 3091 --fps 30 -o game.mp4

Options
    --start/--end   piece range (default: pieces 0..300, cut at the longest game)
    --every N       draw only every Nth piece (fast-forward long games; turns off clear flashes)
    --fps           pieces drawn per second (GIFs: at most 50, browsers slow anything faster)
    --cell          size of one square in pixels (16 -> one board is ~176x388)
    --label         panel title, once per game (default "step 86,944")
    --no-flash      don't show the white "rows about to clear" frame
    -o              output file; .gif or .mp4 (MP4 needs ffmpeg on your PATH)

Side by side: both games show the SAME piece numbers at the same moment. A
game that already ended stays frozen on its last board with "GAME OVER".
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from games.tetris.draw import BACKGROUND, DIM_TEXT, FLASH, GAME_OVER_BG, GRID_LINE, HIDDEN_BG, TEXT, draw_panel, side_by_side
from games.tetris.frames import PIECE_COLORS
from scripts.export_frames import tetris_frames

HOLD_LAST_SECONDS = 2.0  # pause on the final picture before the GIF loops


def timeline(lengths: list[int], start: int, end: int, every: int, flash_at: list[set[int]]) -> list[list[tuple[int, bool]]]:
    """Which frame each game shows at each tick of the animation.

    lengths[g]  number of frames of game g (pieces + 1)
    flash_at[g] piece numbers where game g cleared lines
    Returns ticks; tick = [(frame index, is_flash) for each game].

    A flash tick (only when every == 1) is inserted before piece n if any
    game clears lines at n: the clearing game shows its pre-clear "flash"
    picture, the others simply keep showing piece n-1.
    """
    ticks = []
    for n in range(start, end + 1, every):
        if every == 1 and n > start and any(n in f and n < length for f, length in zip(flash_at, lengths)):
            ticks.append([(min(n, length - 1), True) if n in f and n < length else (min(n - 1, length - 1), False)
                          for f, length in zip(flash_at, lengths)])
        ticks.append([(min(n, length - 1), False) for length in lengths])
    return ticks


def default_label(doc: dict) -> str:
    step = doc["metadata"].get("step")
    return f"step {step:,}" if isinstance(step, int) else f"seed {doc['seed']}"


def render_ticks(docs: list[dict], ticks, labels: list[str], cell: int):
    """Yield one PIL image per tick."""
    for tick in ticks:
        panels = []
        for doc, label, (i, flash) in zip(docs, labels, tick):
            frames = doc["frames"]
            panels.append(draw_panel(frames[i], doc["header"], cell, label,
                                     prev=frames[i - 1] if flash else None, flash=flash))
        yield panels[0] if len(panels) == 1 else side_by_side(panels)


def gif_palette() -> Image.Image:
    """One fixed palette for every GIF frame.

    GIFs can only hold 256 colours per frame. Letting Pillow pick a palette
    per frame makes colours flicker slightly between frames; a fixed one
    (our colours + a gray ramp for the anti-aliased text edges) doesn't.
    """
    colors = [BACKGROUND, HIDDEN_BG, GRID_LINE, TEXT, DIM_TEXT, FLASH, GAME_OVER_BG, *PIECE_COLORS]
    rgb = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in colors]
    rgb += [(v, v, v) for v in range(0, 256, 8)]  # 32 grays
    flat = [x for color in rgb for x in color]
    pal = Image.new("P", (1, 1))
    pal.putpalette(flat + [0] * (768 - len(flat)))
    return pal


def save_gif(images, path: Path, fps: float) -> int:
    pal = gif_palette()
    # dither NONE: map every pixel to the nearest palette colour instead of
    # speckling, which keeps blocks flat (and the file small).
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    ms = round(1000 / fps)
    durations = [ms] * (len(frames) - 1) + [int(HOLD_LAST_SECONDS * 1000)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return len(frames)


def save_mp4(images, path: Path, fps: float) -> int:
    """Pipe raw RGB pixels into ffmpeg, which encodes them as H.264 video."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH; use a .gif output instead")
    proc = None
    count = 0
    last = None
    for im in images:
        if proc is None:
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{im.width}x{im.height}", "-r", str(fps), "-i", "-",
                 # yuv420p = the colour format every player supports; crf 18 = high quality.
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                stdin=subprocess.PIPE)
        assert proc.stdin is not None
        proc.stdin.write(im.tobytes())
        last = im
        count += 1
    if proc is None or proc.stdin is None or last is None:
        raise SystemExit("nothing to draw")
    for _ in range(int(HOLD_LAST_SECONDS * fps)):  # hold the final picture
        proc.stdin.write(last.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--compare", type=Path, default=None, help="second replay, drawn to the right")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--cell", type=int, default=16)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--no-flash", action="store_true")
    args = parser.parse_args()

    docs = [tetris_frames(p) for p in [args.replay, args.compare] if p is not None]
    lengths = [len(d["frames"]) for d in docs]
    end = min(args.end if args.end is not None else args.start + 300, max(lengths) - 1)
    labels = args.label or [default_label(d) for d in docs]
    if len(labels) != len(docs):
        raise SystemExit("give --label once per game")
    flash_at = [set() if args.no_flash else {f["n"] for f in d["frames"] if f["cleared_rows"]} for d in docs]
    ticks = timeline(lengths, args.start, end, args.every, flash_at)
    if args.out.suffix == ".gif" and args.fps > 50:
        print("note: GIF viewers play anything faster than 50 fps slowly; use --every or .mp4")

    images = render_ticks(docs, ticks, labels, args.cell)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".mp4":
        n = save_mp4(images, args.out, args.fps)
    elif args.out.suffix == ".gif":
        n = save_gif(images, args.out, args.fps)
    else:
        raise SystemExit("output must end in .gif or .mp4")
    print(f"pieces {args.start}..{end} ({n} pictures) -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
