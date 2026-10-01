"""Plain-text rendering of a board, for debugging and the random-player demo.

Rendering is kept out of the engine on purpose: training runs millions of
moves and never draws anything, so drawing code must cost nothing unless
someone asks for it.

Legend:
    .      empty
    I O T  locked blocks, labelled by the piece that left them
    @      the active (still falling) piece
Hidden spawn rows are shown above a dashed line when show_hidden=True.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from games.tetris.board import HEIGHT, HIDDEN_ROWS, WIDTH, Board
from games.tetris.pieces import PIECE_NAMES

ACTIVE_CHAR = "@"


def render(
    board: Board,
    active_cells: Iterable[tuple[int, int]] = (),
    active_piece: int | None = None,
    score: int | None = None,
    lines: int | None = None,
    show_hidden: bool = False,
    row_notes: dict[int, str] | None = None,
) -> str:
    """Return the board as a multi-line string.

    active_cells are absolute (row, col) board coordinates of the falling
    piece. active_piece, score and lines are optional status-line fields, so
    this works for a bare board (1a) and a full game state (1b onwards).
    row_notes: text printed to the right of given rows (e.g. "<- clear").
    """
    # Build a list of character rows first, then overlay the active piece.
    rows = [[PIECE_NAMES[v] for v in grid_row] for grid_row in board.grid]
    for r, c in active_cells:
        rows[r][c] = ACTIVE_CHAR

    out: list[str] = []
    status = []
    if active_piece is not None:
        status.append(f"piece: {PIECE_NAMES[active_piece]}")
    if score is not None:
        status.append(f"score: {score}")
    if lines is not None:
        status.append(f"lines: {lines}")
    if status:
        out.append("  ".join(status))

    for r, chars in enumerate(rows):
        if r < HIDDEN_ROWS and not show_hidden:
            continue
        note = (row_notes or {}).get(r)
        out.append("|" + "".join(chars) + "|" + (f" {note}" if note else ""))
        if r == HIDDEN_ROWS - 1 and show_hidden:
            out.append("+" + "-" * WIDTH + "+")  # top of the visible area
    out.append("+" + "-" * WIDTH + "+")  # floor
    return "\n".join(out)


def render_game(engine, show_hidden: bool = False) -> str:
    """Render a TetrisEngine's full state (board, active piece, score, lines).

    Takes the engine by duck typing rather than importing TetrisEngine, so
    this module stays a leaf with no dependency on game logic.
    """
    active = [] if engine.game_over else engine.active_cells()
    text = render(
        engine.board,
        active_cells=active,
        active_piece=engine.piece,
        score=engine.score,
        lines=engine.lines,
        show_hidden=show_hidden,
    )
    if engine.game_over:
        text += "\nGAME OVER"
    return text


CLEAR_NAMES = {1: "single", 2: "double", 3: "triple", 4: "TETRIS"}


def render_frame(frame: dict, prev: dict | None = None, flash: bool = False,
                 show_hidden: bool = True) -> str:
    """Render one exported frame (see games/tetris/frames.py) as text.

    flash=True draws the moment BEFORE the line clear instead: the previous
    board with the new piece shown as @ and the full rows marked, so you can
    see what got cleared. Needs `prev` (the previous frame). Pure drawing:
    every number comes from the frame, nothing is simulated here.
    """
    board_text = prev["board"] if flash and prev else frame["board"]
    # Board strings are '0'..'7' digits; subtracting ord('0') gives piece ids.
    grid = np.frombuffer(board_text.encode(), dtype=np.uint8).reshape(HEIGHT, WIDTH) - ord("0")
    cells = [tuple(rc) for rc in frame["cells"]] if flash else []
    notes = {r: "<- clear" for r in frame["cleared_rows"]} if flash else None
    text = render(Board(grid.copy()), active_cells=cells, show_hidden=show_hidden, row_notes=notes)
    status = (f"piece {frame['n']:,}  score {frame['score']:,}  lines {frame['lines']:,}  "
              f"next {PIECE_NAMES[frame['next']]}\n"
              f"height {frame['height']}  holes {frame['holes']}  bumpiness {frame['bumpiness']}")
    k = len(frame["cleared_rows"])
    if k:
        status += f"  >> {CLEAR_NAMES[k]}!"
    if frame["game_over"]:
        text += "\nGAME OVER"
    return status + "\n" + text
