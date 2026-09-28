"""Tetromino shapes, the rotation system, and the piece randomizer.

Coordinate convention used everywhere in the Tetris engine:
    (row, col), row 0 is the TOP of the board and rows grow DOWNWARD.
    This matches how NumPy prints arrays, so what you see in a debugger is
    what you see on screen.

A piece's shape is a small set of (row, col) offsets inside its "bounding box"
(the smallest square that contains every rotation of the piece). The piece's
position on the board is the board coordinate of that box's top-left corner.
Absolute cell = (piece_row + dr, piece_col + dc).

Two things are deliberately isolated behind tiny classes so they can be
swapped later (roadmap step 5) without touching the engine:
    * RotationSystem  -> today SimpleRotation (no wall kicks); later SRS.
    * Randomizer      -> today UniformRandomizer; later a 7-bag.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Piece ids
# ---------------------------------------------------------------------------
# Plain ints (not an Enum) because they are also the values written into the
# board array (0 = empty, 1..7 = a locked block from that piece type). Keeping
# the piece type in the board lets the text renderer draw letters, and costs
# nothing compared to storing just 0/1.
I, O, T, S, Z, J, L = range(1, 8)
PIECE_TYPES: tuple[int, ...] = (I, O, T, S, Z, J, L)
# Index 0 is "empty" so PIECE_NAMES[board_value] always works.
PIECE_NAMES: tuple[str, ...] = (".", "I", "O", "T", "S", "Z", "J", "L")

# Spawn orientation of each piece, drawn inside its bounding box.
# These match the orientations used by modern Tetris (flat side down for
# T/J/L, I lying horizontally), so an SRS drop-in later lines up with them.
# I uses a 4x4 box and O a 2x2 box; everything else is 3x3.
_SPAWN_DRAWINGS: dict[int, tuple[str, ...]] = {
    I: ("....",
        "####",
        "....",
        "...."),
    O: ("##",
        "##"),
    T: (".#.",
        "###",
        "..."),
    S: (".##",
        "##.",
        "..."),
    Z: ("##.",
        ".##",
        "..."),
    J: ("#..",
        "###",
        "..."),
    L: ("..#",
        "###",
        "..."),
}

# A shape is a tuple of four (dr, dc) offsets. Tuples (not arrays) because
# they are tiny, hashable (useful for de-duplicating placements later), and
# iterating 4 Python tuples is faster than NumPy for something this small.
Shape = tuple[tuple[int, int], ...]


def _drawing_to_cells(drawing: tuple[str, ...]) -> Shape:
    return tuple(
        (r, c)
        for r, line in enumerate(drawing)
        for c, ch in enumerate(line)
        if ch == "#"
    )


def _rotate_drawing_cw(drawing: tuple[str, ...]) -> tuple[str, ...]:
    """Rotate a square drawing 90 degrees clockwise.

    Clockwise rotation of an n x n grid: new[r][c] = old[n-1-c][r].
    """
    n = len(drawing)
    return tuple(
        "".join(drawing[n - 1 - c][r] for c in range(n)) for r in range(n)
    )


def bounding_box_size(piece: int) -> int:
    """Side length of the piece's square bounding box (4 for I, 2 for O, else 3)."""
    return len(_SPAWN_DRAWINGS[piece])


# ---------------------------------------------------------------------------
# Rotation systems
# ---------------------------------------------------------------------------
class SimpleRotation:
    """Classic rotation: spin the shape inside its bounding box, no wall kicks.

    A "wall kick" is a small nudge (e.g. one column right) the game tries when
    a rotation would overlap a wall or block. We don't do any: if the rotated
    piece doesn't fit exactly where it is, the rotation simply fails.

    Interface the engine relies on (an SRS class must provide the same):
        num_rotations          -> how many rotation states exist (4)
        cells(piece, rotation) -> Shape of that piece in that rotation state
        kicks(piece, from_rot, to_rot) -> list of (d_row, d_col) offsets to
            try, in order. The first offset where the piece fits wins.
    """

    num_rotations = 4

    def __init__(self) -> None:
        # Precompute all 7 x 4 shapes once. Rotation is then just a lookup,
        # which matters when we simulate millions of moves.
        self._shapes: dict[int, tuple[Shape, ...]] = {}
        for piece, drawing in _SPAWN_DRAWINGS.items():
            states = []
            for _ in range(self.num_rotations):
                states.append(_drawing_to_cells(drawing))
                drawing = _rotate_drawing_cw(drawing)
            self._shapes[piece] = tuple(states)

    def cells(self, piece: int, rotation: int) -> Shape:
        return self._shapes[piece][rotation % self.num_rotations]

    def kicks(self, piece: int, from_rot: int, to_rot: int) -> list[tuple[int, int]]:
        # Only "rotate in place". SRS would return up to 5 offsets here.
        return [(0, 0)]


# ---------------------------------------------------------------------------
# Randomizers
# ---------------------------------------------------------------------------
class UniformRandomizer:
    """Each new piece is one of the 7 types with equal probability.

    Takes an explicit numpy Generator instead of using the global `random`
    module. That single object is the only source of randomness in a game,
    so seed -> exact same piece sequence, which is what makes replays work.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def next_piece(self) -> int:
        # integers(low, high) excludes high, so this draws from 1..7.
        return int(self.rng.integers(I, L + 1))
