"""The playfield: a NumPy grid of locked blocks plus collision checks.

The board only knows about *locked* blocks. The falling ("active") piece is
not written into the grid until it locks. That way moving the active piece
never has to erase and redraw cells, and a collision check is just "are the
4 target cells inside the grid and empty?".
"""

from __future__ import annotations

import numpy as np

from games.tetris.pieces import Shape

WIDTH = 10
VISIBLE_HEIGHT = 20
# Two extra rows above the visible area. New pieces spawn up here, so a stack
# that reaches the top of the visible area doesn't instantly end the game.
HIDDEN_ROWS = 2
HEIGHT = VISIBLE_HEIGHT + HIDDEN_ROWS  # 22 rows total; rows 0-1 are hidden

EMPTY = 0


class Board:
    """22 x 10 grid. grid[row, col] is 0 (empty) or a piece id 1..7."""

    def __init__(self, grid: np.ndarray | None = None) -> None:
        if grid is None:
            # uint8 keeps the array tiny (220 bytes) and cheap to copy, which
            # matters later when agents copy boards to evaluate placements.
            grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        if grid.shape != (HEIGHT, WIDTH):
            raise ValueError(f"grid must be {HEIGHT}x{WIDTH}, got {grid.shape}")
        self.grid = grid

    def copy(self) -> "Board":
        return Board(self.grid.copy())

    def fits(self, shape: Shape, row: int, col: int) -> bool:
        """True if `shape` placed with its box corner at (row, col) is legal.

        Legal means every cell is inside the walls, above the floor, not above
        the top of the hidden area, and on an empty square. This one function
        handles walls, floor, and other blocks, so every move (shift, rotate,
        drop, spawn) uses the same rule.
        """
        grid = self.grid
        for dr, dc in shape:
            r = row + dr
            c = col + dc
            if r < 0 or r >= HEIGHT or c < 0 or c >= WIDTH:
                return False
            if grid[r, c] != EMPTY:
                return False
        return True

    def place(self, shape: Shape, row: int, col: int, piece: int) -> None:
        """Write a piece's cells into the grid (i.e. lock it). Caller checks fits()."""
        for dr, dc in shape:
            self.grid[row + dr, col + dc] = piece

    @property
    def visible(self) -> np.ndarray:
        """View of the 20 visible rows (no copy)."""
        return self.grid[HIDDEN_ROWS:]
