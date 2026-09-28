"""The pure Tetris rules engine: low-level moves on a single game.

This is the bottom layer. It knows nothing about agents, rewards, or
"placements"; it only exposes the moves a player's buttons would make:

    spawn()        put a new piece in the hidden rows at the top
    shift(dx)      move left (-1) / right (+1)
    rotate(d)      turn clockwise (+1) / counter-clockwise (-1)
    soft_drop()    move down one row
    hard_drop()    fall as far as possible, then lock
    lock()         freeze the piece into the board, clear lines, spawn next

Every move either succeeds completely or does nothing and returns False, so
the active piece is never in an illegal position.

The placement layer (1c) and later a button-press layer (step 5) are both
thin wrappers that call these moves. There is no gravity/timer: pieces only
move when told to, which is all a turn-based agent needs.
"""

from __future__ import annotations

import numpy as np

from games.tetris.board import WIDTH, Board
from games.tetris.pieces import Shape, SimpleRotation, UniformRandomizer, bounding_box_size

# Points for clearing 0, 1, 2, 3, 4 lines with one piece (classic, no levels).
LINE_CLEAR_SCORES = (0, 100, 300, 500, 800)


class GameOverError(RuntimeError):
    """Raised when a move is attempted after the game has ended.

    Failing loudly here catches agent/env bugs early instead of letting a
    finished game quietly keep "playing".
    """


class TetrisEngine:
    """One game of Tetris.

    State (all public, read-only by convention):
        board         Board with the locked blocks
        piece         id of the active piece (1..7)
        rotation      its rotation index (0..3)
        row, col      board position of its bounding box's top-left corner
        score, lines  running totals
        pieces_placed number of pieces locked so far
        game_over     True once a new piece couldn't spawn
    """

    def __init__(
        self,
        rng: np.random.Generator,
        rotation_system: SimpleRotation | None = None,
        randomizer_cls=UniformRandomizer,
        board: Board | None = None,
        spawn_first: bool = True,
    ) -> None:
        # The rotation system and randomizer are injected so they can be
        # swapped (SRS, 7-bag) without editing this class. The randomizer is
        # built *from* the rng we were given, so that rng stays the one and
        # only source of randomness in the game.
        self.rotation_system = rotation_system or SimpleRotation()
        self.randomizer = randomizer_cls(rng)
        self.board = board if board is not None else Board()

        self.piece = 0
        self.rotation = 0
        self.row = 0
        self.col = 0
        self.score = 0
        self.lines = 0
        self.pieces_placed = 0
        self.game_over = False

        # spawn_first=False lets tests set up a board, then spawn a chosen piece.
        if spawn_first:
            self.spawn()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.rotation_system.cells(self.piece, self.rotation)

    def active_cells(self) -> list[tuple[int, int]]:
        """Absolute (row, col) board cells of the active piece."""
        return [(self.row + dr, self.col + dc) for dr, dc in self.shape()]

    def _check_alive(self) -> None:
        if self.game_over:
            raise GameOverError("game is over; call reset / make a new engine")

    def _try_move(self, d_row: int, d_col: int, rotation: int) -> bool:
        """Move to a new (row, col, rotation) only if it fits. Core of every move."""
        shape = self.rotation_system.cells(self.piece, rotation)
        new_row, new_col = self.row + d_row, self.col + d_col
        if self.board.fits(shape, new_row, new_col):
            self.row, self.col, self.rotation = new_row, new_col, rotation
            return True
        return False

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------
    def spawn_position(self, piece: int) -> tuple[int, int]:
        """Where a new piece appears: box top-left at row 0, horizontally centred.

        Centring: (10 - box size) // 2 -> column 3 for 3- and 4-wide boxes,
        column 4 for the 2-wide O.

        Row 0 for the *box* (not the top cell) matters: every rotation state
        then fits inside rows 0..3, so any piece can rotate right after
        spawning on an empty board. An earlier version pushed the flat I's
        cells up to row 0, which put its box at row -1 and made the vertical
        I impossible to reach. In spawn orientation all pieces still sit
        entirely in the 2 hidden rows (the I lies on row 1).
        """
        return 0, (WIDTH - bounding_box_size(piece)) // 2

    def spawn(self, piece: int | None = None) -> bool:
        """Bring in the next piece (or a specific one, for tests / a future hold).

        Returns False and sets game_over if the spawn position is blocked.
        This is the ONLY way the game can end.
        """
        self._check_alive()
        if piece is None:
            piece = self.randomizer.next_piece()
        self.piece = piece
        self.rotation = 0
        self.row, self.col = self.spawn_position(piece)
        if not self.board.fits(self.shape(), self.row, self.col):
            self.game_over = True
            return False
        return True

    def shift(self, dx: int) -> bool:
        """Move one column left (dx=-1) or right (dx=+1)."""
        self._check_alive()
        return self._try_move(0, dx, self.rotation)

    def rotate(self, direction: int = 1) -> bool:
        """Rotate clockwise (+1) or counter-clockwise (-1).

        Tries each kick offset from the rotation system in order. With
        SimpleRotation that's only (0, 0): rotate in place or not at all.
        """
        self._check_alive()
        n = self.rotation_system.num_rotations
        new_rot = (self.rotation + direction) % n
        for d_row, d_col in self.rotation_system.kicks(self.piece, self.rotation, new_rot):
            if self._try_move(d_row, d_col, new_rot):
                return True
        return False

    def soft_drop(self) -> bool:
        """Move down one row. False means the piece is resting on something."""
        self._check_alive()
        return self._try_move(1, 0, self.rotation)

    def hard_drop(self) -> int:
        """Drop straight down until blocked, then lock. Returns lines cleared."""
        self._check_alive()
        while self._try_move(1, 0, self.rotation):
            pass
        return self.lock()

    def lock(self) -> int:
        """Freeze the active piece where it is, clear lines, score, spawn next.

        Returns the number of lines cleared. Locking is allowed wherever the
        piece currently is (even mid-air) - deciding *when* to lock is the
        caller's job. Placements always hard-drop first.
        """
        self._check_alive()
        self.board.place(self.shape(), self.row, self.col, self.piece)
        cleared = self.board.clear_full_lines()
        self.lines += cleared
        self.score += LINE_CLEAR_SCORES[cleared]
        self.pieces_placed += 1
        self.spawn()  # may set game_over
        return cleared
