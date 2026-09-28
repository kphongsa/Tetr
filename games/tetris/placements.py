"""Placement actions: "put the current piece HERE" instead of pressing buttons.

A placement is a final resting spot for the active piece, identified by
(rotation, column) where column = the piece's LEFTMOST cell after rotating.
(Leftmost cell, not bounding-box corner: the box can hang off the wall, the
cells can't, so this keeps columns in 0..9.)

How legality is decided
-----------------------
We don't compute placements geometrically. We *perform* them with the
engine's own low-level moves, in the same order a simple human would:

    1. rotate at the spawn position (shortest way: 3 turns = 1 turn back)
    2. shift one column at a time toward the target
    3. drop straight down

If any move is blocked, that placement is unreachable. Rotations and shifts
call the engine directly, so placements can't disagree with the rules, and
a later button-press wrapper uses the same engine unchanged. The drop in
step 3 uses a faster equivalent calculation while *listing* placements (see
_next_filled_table); *applying* a placement uses the engine's real hard_drop,
and tests check the two always agree.

Consequence worth knowing: "tucks" and "spins" (sliding under an overhang
after dropping, or rotating into a slot) are NOT placements here. That's
standard for Tetris AI, and it keeps the action count small (<= 34).

Action ids
----------
Agents with a fixed-size output (e.g. a neural network with one output per
action) need a fixed action space, so each (rotation, column) pair gets an
id: action = rotation * 10 + column  ->  40 ids, 0..39. Most are illegal at
any given moment; the env reports which ones are legal via an action mask.
"""

from __future__ import annotations

from dataclasses import dataclass

from games.tetris.board import WIDTH
from games.tetris.engine import TetrisEngine

NUM_ROTATIONS = 4
NUM_ACTIONS = NUM_ROTATIONS * WIDTH  # 40


def encode_action(rotation: int, column: int) -> int:
    return rotation * WIDTH + column


def decode_action(action: int) -> tuple[int, int]:
    """Inverse of encode_action: returns (rotation, column)."""
    return divmod(action, WIDTH)


@dataclass(frozen=True)
class Placement:
    action: int
    rotation: int
    column: int  # leftmost cell's column
    cells: tuple[tuple[int, int], ...]  # final absolute (row, col) cells, sorted


def _leftmost(engine: TetrisEngine) -> int:
    return engine.col + min(dc for _, dc in engine.shape())


def _rotate_to(engine: TetrisEngine, target: int) -> bool:
    """Rotate from the current rotation to `target` the short way round."""
    n = engine.rotation_system.num_rotations
    turns = (target - engine.rotation) % n
    if turns == 0:
        return True
    if turns == n - 1:  # e.g. 3 clockwise == 1 counter-clockwise
        return engine.rotate(-1)
    return all(engine.rotate(+1) for _ in range(turns))  # all() stops at first failure


def _next_filled_table(grid) -> list[list[int]]:
    """table[c][r] = first row BELOW r in column c that holds a block (HEIGHT if none).

    Why this exists (speed): the first version found each landing spot by
    calling engine.soft_drop() row by row. The benchmark showed that was ~70%
    of all runtime (~12 drops x ~30 placements x 4 cells, each a NumPy
    scalar lookup). This table is built once per piece in plain Python
    (220 cells) and turns every landing into 4 lookups.
    """
    rows = grid.tolist()  # plain Python lists index ~10x faster than NumPy scalars
    height, width = len(rows), len(rows[0])
    table = [[height] * height for _ in range(width)]
    for c in range(width):
        nxt = height
        for r in range(height - 1, -1, -1):
            table[c][r] = nxt
            if rows[r][c]:
                nxt = r
    return table


def _landing_cells(engine: TetrisEngine, next_filled) -> tuple[tuple[int, int], ...]:
    """Final cells if the active piece dropped straight down from where it is.

    Each cell (r, c) can fall until just above the next block in its column;
    the piece falls by the smallest of those four distances. This gives the
    same result as repeated soft drops (a test checks it against the
    engine's real hard_drop), just much faster. No state is changed.
    """
    row, col = engine.row, engine.col
    shape = engine.shape()
    fall = min(next_filled[col + dc][row + dr] - (row + dr) - 1 for dr, dc in shape)
    return tuple(sorted((row + dr + fall, col + dc) for dr, dc in shape))


def legal_placements(engine: TetrisEngine) -> list[Placement]:
    """Every distinct reachable final placement of the active piece, sorted by action.

    Distinct means distinct final cells: an O placed with rotation 0 or 2
    ends up in exactly the same squares, so it's offered only once (with
    the lowest rotation). Agents shouldn't waste effort on duplicates.

    Temporarily moves the active piece around, and always restores it.
    """
    if engine.game_over:
        return []

    spawn_state = (engine.rotation, engine.row, engine.col)
    next_filled = _next_filled_table(engine.board.grid)
    seen_cells: set[tuple[tuple[int, int], ...]] = set()
    result: list[Placement] = []

    def record() -> None:
        cells = _landing_cells(engine, next_filled)
        if cells not in seen_cells:
            seen_cells.add(cells)
            column = _leftmost(engine)
            result.append(
                Placement(encode_action(engine.rotation, column), engine.rotation, column, cells)
            )

    try:
        for target_rot in range(engine.rotation_system.num_rotations):
            engine.rotation, engine.row, engine.col = spawn_state
            if not _rotate_to(engine, target_rot):
                continue
            rotated_state = (engine.rotation, engine.row, engine.col)

            # Sweep left from the rotated position (including it)...
            record()
            while engine.shift(-1):
                record()
            # ...then sweep right from it. Each step reached by shifting one
            # column at a time is exactly the path apply_placement will take.
            engine.rotation, engine.row, engine.col = rotated_state
            while engine.shift(+1):
                record()
    finally:
        engine.rotation, engine.row, engine.col = spawn_state

    result.sort(key=lambda p: p.action)
    return result


def apply_placement(engine: TetrisEngine, placement: Placement) -> int:
    """Carry out a placement with real moves, lock it, and return lines cleared.

    The caller must pass a placement from legal_placements() for the CURRENT
    piece. We still verify each move, so a stale/wrong placement fails loudly
    instead of quietly putting the piece somewhere else.
    """
    if not _rotate_to(engine, placement.rotation):
        raise ValueError(f"rotation blocked for {placement}")
    direction = 1 if placement.column > _leftmost(engine) else -1
    while _leftmost(engine) != placement.column:
        if not engine.shift(direction):
            raise ValueError(f"shift blocked for {placement}")
    return engine.hard_drop()
