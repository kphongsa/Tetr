"""Tetris frames: what a viewer needs to draw the game after each piece.

The Python engine is the single source of truth for the rules. Viewers
(terminal, GIF, web page) never simulate anything: they only draw these
frames, so a viewer can't "disagree" with the engine about where a piece
lands or which rows clear.

One frame = the state right after one piece was placed (frame 0 = the
empty board before the first piece). Keys are short because a long game
has thousands of frames:

    n            pieces placed so far (frame index)
    board        the 22x10 grid as a 220-character string, row by row from
                 the top, one digit per cell: 0 = empty, 1..7 = piece id
                 (I O T S Z J L). Hidden rows 0-1 come first.
    piece        id of the piece just placed (0 in frame 0)
    action       the placement action id that was played (null in frame 0)
    cells        [[row, col], ...] where that piece landed, BEFORE any line
                 clear (so a viewer can flash it on the previous board)
    cleared_rows rows (same "before clear" coordinates) that were full and
                 got cleared; [] most of the time
    score, lines running totals
    next         the piece now waiting to be placed (0 after game over)
    height       tallest column (0..22; 21-22 = in the hidden spawn rows)
    holes        empty cells with a block somewhere above them
    bumpiness    sum of height differences between neighbouring columns
    game_over    true only on the last frame of a lost game

Size: ~380 bytes per frame, so a 3,000-piece game is ~1.1 MB of JSON and
the 10,000-piece cap ~3.8 MB. Board strings repeat a lot, so gzip (which
web servers, including GitHub Pages, apply automatically) shrinks this ~10x.

This module only imports from games/tetris (architecture rule 1). It fits
the FrameRecorder interface in core/replay.py by duck typing: same method
names, no import needed.
"""

from __future__ import annotations

import numpy as np

from games.tetris.board import HEIGHT, HIDDEN_ROWS, VISIBLE_HEIGHT, WIDTH
from games.tetris.features import board_features, column_heights
from games.tetris.pieces import PIECE_NAMES

# Standard "guideline" colours, index = piece id (0 = empty background).
# Shared by every viewer (they read them from the frames header), so the
# GIF and the web page look the same.
PIECE_COLORS: tuple[str, ...] = (
    "#1b1d24",  # empty
    "#31c7ef",  # I cyan
    "#f7d308",  # O yellow
    "#ad4d9c",  # T purple
    "#42b642",  # S green
    "#ef2029",  # Z red
    "#5a65ad",  # J blue
    "#ef7921",  # L orange
)


def board_to_string(grid: np.ndarray) -> str:
    return "".join(str(int(v)) for v in grid.ravel())


def board_from_string(text: str) -> np.ndarray:
    """Inverse of board_to_string: a (22, 10) uint8 grid of piece ids."""
    return np.frombuffer(text.encode(), dtype=np.uint8).reshape(HEIGHT, WIDTH) - ord("0")


def board_stats(grid: np.ndarray) -> dict:
    """Height, holes, bumpiness of a board (piece ids or 0/1, either works)."""
    feats = board_features(grid, 0)
    return {"height": int(column_heights(grid).max()), "holes": int(feats[2]), "bumpiness": int(feats[3])}


def board_before_clear(prev_board: str, piece: int, cells) -> np.ndarray:
    """The previous frame's board with the new piece drawn in, before rows clear.

    Pure drawing (no rules): it just paints 4 cells. Used by viewers to show
    the filled rows for a moment before they disappear.
    """
    grid = board_from_string(prev_board).copy()
    for r, c in cells:
        grid[r, c] = piece
    return grid


class TetrisFrameRecorder:
    """Builds one frame per step from a TetrisEnv (see module docstring)."""

    def header(self) -> dict:
        return {
            "width": WIDTH,
            "height": HEIGHT,
            "hidden_rows": HIDDEN_ROWS,
            "visible_height": VISIBLE_HEIGHT,
            "piece_names": list(PIECE_NAMES),
            "piece_colors": list(PIECE_COLORS),
        }

    def start(self, env, obs, info) -> dict:
        self._remember(env, info)
        return self._frame(env, info, piece=0, action=None, cells=[], cleared_rows=[])

    def step(self, env, action, obs, info) -> dict:
        # The piece that was just placed is the one we remembered last time,
        # and its landing cells come from the placement list the env offered
        # BEFORE this step (the env has already moved on to the next piece).
        cells = [list(rc) for rc in self._placements[int(action)].cells]
        grid = self._grid.copy()
        for r, c in cells:
            grid[r, c] = self._piece
        cleared_rows = [int(r) for r in np.flatnonzero((grid != 0).all(axis=1))]
        # Sanity check: our "which rows were full" must agree with the engine.
        assert len(cleared_rows) == info["lines_cleared"], (cleared_rows, info["lines_cleared"])
        frame = self._frame(env, info, piece=self._piece, action=int(action), cells=cells,
                            cleared_rows=cleared_rows)
        self._remember(env, info)
        return frame

    # ------------------------------------------------------------------
    def _remember(self, env, info) -> None:
        engine = env.engine
        self._piece = int(engine.piece)
        self._grid = engine.board.grid.copy()
        self._placements = {p.action: p for p in info["placements"]}

    def _frame(self, env, info, piece, action, cells, cleared_rows) -> dict:
        engine = env.engine
        grid = engine.board.grid
        return {
            "n": int(info["pieces"]),
            "board": board_to_string(grid),
            "piece": piece,
            "action": action,
            "cells": cells,
            "cleared_rows": cleared_rows,
            "score": int(info["score"]),
            "lines": int(info["lines"]),
            "next": 0 if engine.game_over else int(engine.piece),
            **board_stats(grid),
            "game_over": bool(engine.game_over),
        }
