"""Features and afterstates on small hand-drawn boards you can count by eye."""

import numpy as np
import pytest

from games.tetris.board import HEIGHT, WIDTH, Board
from games.tetris.engine import TetrisEngine
from games.tetris.features import (
    FEATURE_NAMES,
    afterstate,
    afterstates,
    board_features,
    column_heights,
)
from games.tetris.pieces import I, PIECE_TYPES
from games.tetris.placements import apply_placement, legal_placements


def draw(*rows: str) -> np.ndarray:
    """Build a 22x10 grid whose BOTTOM rows are the given drawings ('#' = block)."""
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for i, line in enumerate(rows):
        assert len(line) == WIDTH
        r = HEIGHT - len(rows) + i
        grid[r] = [ch == "#" for ch in line]
    return grid


def feats(grid, lines=0) -> dict:
    return dict(zip(FEATURE_NAMES, board_features(grid, lines)))


def test_empty_board_is_all_zero():
    grid = draw()
    assert list(column_heights(grid)) == [0] * 10
    assert feats(grid) == {"aggregate_height": 0, "lines_cleared": 0, "holes": 0, "bumpiness": 0}


def test_heights_holes_bumpiness_by_hand():
    grid = draw(
        "#.........",   # col 0 height 3
        "#.#.......",   # col 2 height 2, with a hole under it
        "##..#.....",   # col 1 height 1, col 4 height 1
    )
    #  heights: 3 1 2 0 1 0 0 0 0 0   -> aggregate 7
    #  holes:   col 2 row -1 is empty with a block above -> 1
    #  bumpiness: |3-1|+|1-2|+|2-0|+|0-1|+|1-0| = 2+1+2+1+1 = 7
    assert list(column_heights(grid)) == [3, 1, 2, 0, 1, 0, 0, 0, 0, 0]
    assert feats(grid) == {"aggregate_height": 7, "lines_cleared": 0, "holes": 1, "bumpiness": 7}


def test_multiple_holes_in_one_column_all_count():
    grid = draw(
        "#.........",
        "..........",
        "#.........",
        "..........",
    )
    # col 0 height 4 but only 2 blocks -> 2 holes. Bumpiness |4-0| = 4.
    assert feats(grid) == {"aggregate_height": 4, "lines_cleared": 0, "holes": 2, "bumpiness": 4}


def test_lines_cleared_is_passed_through():
    assert feats(draw(), lines=3)["lines_cleared"] == 3


def test_afterstate_places_cells_and_does_not_touch_input():
    board = draw("#########.")
    before = board.copy()
    # A vertical I in column 9 fills the gap: bottom row clears, 3 I cells remain.
    cells = [(HEIGHT - 4, 9), (HEIGHT - 3, 9), (HEIGHT - 2, 9), (HEIGHT - 1, 9)]
    grid, cleared = afterstate(board, cells)
    assert cleared == 1
    np.testing.assert_array_equal(board, before)  # input untouched
    np.testing.assert_array_equal(grid, draw(".........#", ".........#", ".........#"))


def test_afterstate_accepts_piece_id_grid_and_returns_0_1():
    board = draw("..........")
    board[HEIGHT - 1, 0] = 5  # a Z-piece block, as the engine stores it
    grid, _ = afterstate(board, [(HEIGHT - 1, 1)])
    assert set(np.unique(grid)) == {0, 1}


def test_afterstates_leave_real_engine_unchanged():
    b = Board(draw("###.######", "###.######"))
    e = TetrisEngine(np.random.default_rng(0), board=b, spawn_first=False)
    e.spawn(I)
    grid_before = e.board.grid.copy()
    state_before = (e.piece, e.rotation, e.row, e.col, e.score, e.lines)
    results = afterstates(e.board.grid, legal_placements(e))
    assert len(results) == len(legal_placements(e))
    np.testing.assert_array_equal(e.board.grid, grid_before)
    assert (e.piece, e.rotation, e.row, e.col, e.score, e.lines) == state_before
    # The vertical I into column 3 clears both bottom rows: the best move exists.
    assert max(a.lines_cleared for a in results) == 2


@pytest.mark.parametrize("piece", PIECE_TYPES)
def test_afterstate_matches_really_playing_the_placement(piece):
    """For every legal placement, the predicted afterstate == what the engine does."""
    base = draw(
        "..#.......",
        "#.##..#..#",
        "####.#####",
        "#####.####",
    )
    e = TetrisEngine(np.random.default_rng(0), board=Board(base.copy()), spawn_first=False)
    e.spawn(piece)
    for a in afterstates(e.board.grid, legal_placements(e)):
        real = TetrisEngine(np.random.default_rng(0), board=Board(base.copy()), spawn_first=False)
        real.spawn(piece)
        lines = apply_placement(real, a.placement)
        assert lines == a.lines_cleared
        np.testing.assert_array_equal((real.board.grid != 0).astype(np.uint8), a.grid)
