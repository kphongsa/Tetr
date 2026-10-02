"""Play-style metrics: HOW a game was played, not just how well.

One clean, reusable function (CLAUDE.md step 4d). In step 5 these numbers are
meant to become MAP-Elites "behavior descriptors": coordinates that place
each agent on a map of play styles (e.g. "low flat stack, mostly tetrises"
vs "tall messy stack, mostly singles").

Use it live, piece by piece (no boards are stored, so it's cheap during
training or search):

    tracker = StyleTracker()
    ... after every placement:  tracker.add(board, lines_cleared)
    tracker.metrics()  ->  dict (below)

or on an exported game:  style_of_frames(doc["frames"]).

Metrics (plain-language names; every average is "per piece placed"):
    pieces             how many pieces the game lasted (or how many were measured)
    lines              lines cleared
    avg_stack_height   height of the tallest column after each piece
                       (20 = top of the visible board)
    max_stack_height   the worst moment
    avg_holes          empty squares with a block somewhere above them
    avg_bumpiness      how uneven the surface is: sum of height differences
                       between neighbouring columns
    share_singles      fraction of all cleared LINES that came from clearing
    share_doubles      1 / 2 / 3 / 4 rows with one piece. Counted in lines,
    share_triples      not clear events: one tetris = 4 lines, so a player
    share_tetrises     scoring big via tetrises shows a large share_tetrises.
                       All four sum to 1 (all 0 if no lines were cleared).
    pieces_per_clear   pieces placed per line-clear event (how often it
                       clears anything). None if it never cleared.

Only imports games/tetris code (architecture rule 1).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from games.tetris.features import board_features, column_heights
from games.tetris.frames import board_from_string

METRIC_NAMES: tuple[str, ...] = (
    "pieces", "lines", "avg_stack_height", "max_stack_height", "avg_holes", "avg_bumpiness",
    "share_singles", "share_doubles", "share_triples", "share_tetrises", "pieces_per_clear",
)


class StyleTracker:
    """Accumulates style statistics one placement at a time."""

    def __init__(self) -> None:
        self.pieces = 0
        self.height_sum = 0
        self.max_height = 0
        self.holes_sum = 0
        self.bump_sum = 0
        self.clears = [0, 0, 0, 0, 0]  # clears[k] = placements that cleared k rows

    def add(self, board: np.ndarray, lines_cleared: int) -> None:
        """Record one placement: the board AFTER it (and its line clear), 0/1 or piece ids."""
        feats = board_features(board, 0)  # [aggregate height, lines, holes, bumpiness]
        height = int(column_heights(board).max())
        self.pieces += 1
        self.height_sum += height
        self.max_height = max(self.max_height, height)
        self.holes_sum += int(feats[2])
        self.bump_sum += int(feats[3])
        self.clears[lines_cleared] += 1

    def metrics(self) -> dict:
        n = max(self.pieces, 1)  # avoid dividing by zero for an empty game
        lines_by_size = [k * self.clears[k] for k in range(1, 5)]  # lines from singles, doubles, ...
        lines = sum(lines_by_size)
        events = sum(self.clears[1:])
        shares = [x / lines if lines else 0.0 for x in lines_by_size]
        return {
            "pieces": self.pieces,
            "lines": lines,
            "avg_stack_height": self.height_sum / n,
            "max_stack_height": self.max_height,
            "avg_holes": self.holes_sum / n,
            "avg_bumpiness": self.bump_sum / n,
            "share_singles": shares[0],
            "share_doubles": shares[1],
            "share_triples": shares[2],
            "share_tetrises": shares[3],
            "pieces_per_clear": self.pieces / events if events else None,
        }


def style_metrics(boards: Iterable[np.ndarray], lines_cleared: Iterable[int]) -> dict:
    """Style of a game given the board after each piece and the lines each piece cleared."""
    tracker = StyleTracker()
    for board, k in zip(boards, lines_cleared):
        tracker.add(board, k)
    return tracker.metrics()


def style_of_frames(frames: list[dict], max_pieces: int | None = None) -> dict:
    """Style of an exported game (games/tetris/frames.py). Frame 0 (empty board) is skipped.

    max_pieces: only the first N pieces, so games of different lengths can
    be compared over the same stretch.
    """
    played = frames[1:] if max_pieces is None else frames[1:max_pieces + 1]
    return style_metrics((board_from_string(f["board"]) for f in played),
                         (len(f["cleared_rows"]) for f in played))
