"""Board features and afterstates: the raw material for rule-based and learned agents.

Afterstate
----------
The board right AFTER a placement locks and full lines clear, but BEFORE the
next piece appears. Each legal placement leads to exactly one afterstate, so
choosing a move = choosing the afterstate you like best. The next piece is
random, so the afterstate is the last thing the player fully controls.

How we get afterstates without touching the real game ("copy", not "peek")
-------------------------------------------------------------------------
We copy the board grid (22 x 10 uint8 = 220 bytes, cheap), write the
placement's final cells into the copy, and clear full lines on the copy. The
real engine is never modified. The alternative, "peek" (place on the real
board, measure, then undo), would need to undo line clears too, which is
fiddly and one bug away from corrupting the actual game.

Line clearing reuses Board.clear_full_lines, so there is exactly one
line-clear rule in the codebase and afterstates can't drift from real play.

Features (all computed on the afterstate)
-----------------------------------------
    aggregate_height  sum of the 10 column heights. Tall stack = close to death.
    lines_cleared     lines this placement cleared. Frees room, scores points.
    holes             empty cells with a block somewhere above them in the same
                      column. They can't be filled until everything above is
                      cleared, so they block line clears.
    bumpiness         sum of |height difference| between neighbouring columns.
                      A jagged surface fits pieces badly and breeds holes.

"Height" of a column = number of rows from the floor up to and including its
top-most block (0 for an empty column), measured over all 22 rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from games.tetris.board import HEIGHT, Board
from games.tetris.placements import Placement

# Fixed order of the feature vector. Weight vectors (step 2b) use this order.
FEATURE_NAMES: tuple[str, ...] = ("aggregate_height", "lines_cleared", "holes", "bumpiness")


@dataclass(frozen=True)
class Afterstate:
    placement: Placement
    grid: np.ndarray  # 0/1 uint8 (22, 10), lines already cleared
    lines_cleared: int


def afterstate(board: np.ndarray, cells) -> tuple[np.ndarray, int]:
    """Board after locking `cells` and clearing lines. Returns (new 0/1 grid, lines cleared).

    `board` can be the env observation (0/1) or the engine's grid (piece ids);
    either way the input is never modified. The result is 0/1 because which
    piece left a block doesn't matter for judging a position.
    """
    grid = (board != 0).astype(np.uint8)  # astype makes a fresh copy
    for r, c in cells:
        grid[r, c] = 1
    cleared = Board(grid).clear_full_lines()  # Board wraps (doesn't copy) our copy
    return grid, cleared


def afterstates(board: np.ndarray, placements: list[Placement]) -> list[Afterstate]:
    """The afterstate of every placement, in the same order as `placements`."""
    result = []
    for p in placements:
        grid, cleared = afterstate(board, p.cells)
        result.append(Afterstate(p, grid, cleared))
    return result


def column_heights(grid: np.ndarray) -> np.ndarray:
    """Height of each column (int array of length 10)."""
    filled = grid != 0
    # argmax on a bool column returns the index of the first True, i.e. the
    # top-most block (row 0 is the top). For an all-empty column it returns 0,
    # which would look like a full-height column, hence the np.where.
    top = filled.argmax(axis=0)
    return np.where(filled.any(axis=0), HEIGHT - top, 0)


def board_features(grid: np.ndarray, lines_cleared: int) -> np.ndarray:
    """Feature vector in FEATURE_NAMES order, as float64 (ready for a dot product)."""
    heights = column_heights(grid)
    aggregate_height = int(heights.sum())
    # Holes trick: a column of height h spans h cells from its top block down to
    # the floor. Every filled cell in that column lies inside that span, so the
    # empty cells in the span (= the holes) number h - (filled cells). Summing
    # over columns: holes = total height - total filled cells. No loops needed.
    holes = aggregate_height - int(np.count_nonzero(grid))
    bumpiness = int(np.abs(np.diff(heights)).sum())
    return np.array([aggregate_height, lines_cleared, holes, bumpiness], dtype=np.float64)
