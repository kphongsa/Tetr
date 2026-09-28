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

from games.tetris.board import HIDDEN_ROWS, WIDTH, Board
from games.tetris.pieces import PIECE_NAMES

ACTIVE_CHAR = "@"


def render(
    board: Board,
    active_cells: Iterable[tuple[int, int]] = (),
    active_piece: int | None = None,
    score: int | None = None,
    lines: int | None = None,
    show_hidden: bool = False,
) -> str:
    """Return the board as a multi-line string.

    active_cells are absolute (row, col) board coordinates of the falling
    piece. active_piece, score and lines are optional status-line fields, so
    this works for a bare board (1a) and a full game state (1b onwards).
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
        out.append("|" + "".join(chars) + "|")
        if r == HIDDEN_ROWS - 1 and show_hidden:
            out.append("+" + "-" * WIDTH + "+")  # top of the visible area
    out.append("+" + "-" * WIDTH + "+")  # floor
    return "\n".join(out)
