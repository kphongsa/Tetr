"""Draw exported Tetris frames as images (Pillow), for GIFs and videos.

Pure drawing: every number and every cell comes from a frame produced by
games/tetris/frames.py. Nothing here knows the rules.

One "panel" looks like this (cell = size of one square in pixels):

    +---------------------+
    | title (e.g. step)   |   title strip
    |  hidden rows (dim)  |   2 spawn rows, darker, above a thin line
    |  20 visible rows    |
    | score / lines / n   |   stats strip
    +---------------------+

Pillow ("PIL") is the standard Python image library: Image is a picture
in memory, ImageDraw draws rectangles and text onto it.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from games.tetris.frames import board_before_clear, board_from_string

BACKGROUND = "#0e0f13"
HIDDEN_BG = "#15161c"  # the 2 spawn rows: slightly different so you can tell they're off-board
GRID_LINE = "#262833"
TEXT = "#e8e8ee"
DIM_TEXT = "#9a9cab"
FLASH = "#ffffff"  # rows about to clear
GAME_OVER_BG = "#b3261e"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # load_default(size) gives Pillow's built-in scalable font (Pillow >= 10.1),
    # so we don't depend on which fonts this computer has installed.
    return ImageFont.load_default(size=size)


def panel_size(header: dict, cell: int) -> tuple[int, int]:
    """(width, height) in pixels of one panel. Always even, which MP4 encoders need."""
    width = header["width"] * cell + 2 * _pad(cell)
    height = _title_h(cell) + header["height"] * cell + _stats_h(cell)
    return width + width % 2, height + height % 2


def draw_panel(frame: dict, header: dict, cell: int = 16, title: str = "",
               prev: dict | None = None, flash: bool = False) -> Image.Image:
    """Draw one frame. flash=True draws the moment before a line clear
    (previous board + the new piece, cleared rows in white); needs `prev`."""
    w, h = panel_size(header, cell)
    img = Image.new("RGB", (w, h), BACKGROUND)
    d = ImageDraw.Draw(img)
    pad, top = _pad(cell), _title_h(cell)
    colors = header["piece_colors"]
    hidden = header["hidden_rows"]

    if flash and prev is not None:
        grid = board_before_clear(prev["board"], frame["piece"], frame["cells"])
        white_rows = set(frame["cleared_rows"])
    else:
        grid = board_from_string(frame["board"])
        white_rows = set()

    # Board cells. Empty cells are drawn too (background colour), with a 1px
    # gap so the grid is visible; filled cells get the piece colour.
    for r in range(header["height"]):
        for c in range(header["width"]):
            x0, y0 = pad + c * cell, top + r * cell
            v = int(grid[r, c])
            if r in white_rows:
                fill = FLASH
            elif v:
                fill = colors[v]
            else:
                fill = HIDDEN_BG if r < hidden else colors[0]
            d.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=fill)
    # Line between the hidden spawn rows and the visible board.
    y = top + hidden * cell - 1
    d.line([pad, y, pad + header["width"] * cell - 2, y], fill=GRID_LINE, width=max(1, cell // 8))

    # Title strip and stats strip.
    small, big = _font(max(9, int(cell * 0.7))), _font(max(10, int(cell * 0.85)))
    d.text((pad, cell * 0.3), title, fill=TEXT, font=big)
    y = top + header["height"] * cell + cell * 0.25
    d.text((pad, y), f"score {frame['score']:,}", fill=TEXT, font=small)
    d.text((pad, y + cell * 0.95), f"lines {frame['lines']:,}   piece {frame['n']:,}", fill=DIM_TEXT, font=small)

    if frame["game_over"]:
        _banner(d, "GAME OVER", w, top + header["height"] * cell // 2, cell)
    return img


def side_by_side(panels: list[Image.Image], gap: int = 8) -> Image.Image:
    """Put panels next to each other on one image."""
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    w += w % 2  # keep it even for MP4
    h = max(p.height for p in panels)
    out = Image.new("RGB", (w, h), BACKGROUND)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


# ----------------------------------------------------------------------
def _pad(cell: int) -> int:
    return cell // 2


def _title_h(cell: int) -> int:
    return int(cell * 1.5)


def _stats_h(cell: int) -> int:
    return int(cell * 2.3)


def _banner(d: ImageDraw.ImageDraw, text: str, width: int, y_mid: int, cell: int) -> None:
    font = _font(max(10, cell))
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    d.rectangle([0, y_mid - th, width, y_mid + th], fill=GAME_OVER_BG)
    d.text(((width - tw) / 2, y_mid - th / 2 - top), text, fill=TEXT, font=font)
