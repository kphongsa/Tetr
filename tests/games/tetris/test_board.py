import numpy as np
import pytest

from games.tetris.board import HEIGHT, HIDDEN_ROWS, WIDTH, Board
from games.tetris.pieces import I, O, T, SimpleRotation
from games.tetris.render_text import render

ROT = SimpleRotation()
T_UP = ROT.cells(T, 0)  # cells (0,1) (1,0) (1,1) (1,2): box cols 0..2 used
I_FLAT = ROT.cells(I, 0)  # row 1 of its 4x4 box, cols 0..3
I_VERT = ROT.cells(I, 1)  # col 2 of its 4x4 box, rows 0..3


def test_board_dimensions():
    b = Board()
    assert b.grid.shape == (22, 10) == (HEIGHT, WIDTH)
    assert b.visible.shape == (20, 10)
    assert b.grid.dtype == np.uint8


def test_rejects_wrong_grid_shape():
    with pytest.raises(ValueError):
        Board(np.zeros((20, 10), dtype=np.uint8))


def test_fits_on_empty_board():
    assert Board().fits(T_UP, 5, 3)


# --- walls -----------------------------------------------------------------

def test_left_wall():
    b = Board()
    assert b.fits(T_UP, 5, 0)  # leftmost cell at col 0: ok
    assert not b.fits(T_UP, 5, -1)  # leftmost cell at col -1: through the wall


def test_right_wall():
    b = Board()
    assert b.fits(T_UP, 5, WIDTH - 3)  # rightmost cell at col 9
    assert not b.fits(T_UP, 5, WIDTH - 2)  # rightmost cell at col 10


def test_box_may_hang_outside_as_long_as_cells_dont():
    # Vertical I only uses column 2 of its 4-wide box, so the box can extend
    # past the wall. Collision is about cells, not the box.
    b = Board()
    assert b.fits(I_VERT, 5, -2)  # cells in col 0
    assert not b.fits(I_VERT, 5, -3)  # cells in col -1
    assert b.fits(I_VERT, 5, WIDTH - 3)  # cells in col 9
    assert not b.fits(I_VERT, 5, WIDTH - 2)


# --- floor and ceiling -----------------------------------------------------

def test_floor():
    b = Board()
    # T_UP bottom cells are at box row 1, so box row HEIGHT-2 rests on the floor.
    assert b.fits(T_UP, HEIGHT - 2, 3)
    assert not b.fits(T_UP, HEIGHT - 1, 3)


def test_ceiling():
    b = Board()
    # Flat I sits on row 1 of its box, so the box can start at row -1.
    assert b.fits(I_FLAT, -1, 3)
    assert not b.fits(I_FLAT, -2, 3)


# --- other blocks ----------------------------------------------------------

def test_overlapping_a_locked_block():
    b = Board()
    b.grid[10, 4] = O
    # T_UP at (9, 3) covers (9,4) (10,3) (10,4) (10,5) -> hits (10,4).
    assert not b.fits(T_UP, 9, 3)
    # One row higher covers (8,4) (9,3) (9,4) (9,5) -> clear.
    assert b.fits(T_UP, 8, 3)


def test_empty_box_cells_dont_collide():
    # The T's box has empty corners; a block there must not block the piece.
    b = Board()
    b.grid[10, 3] = O  # the box's top-left corner when box is at (10, 3)
    assert b.fits(T_UP, 10, 3)


def test_place_writes_piece_id_and_then_collides():
    b = Board()
    b.place(T_UP, 5, 3, T)
    assert b.grid[5, 4] == T and b.grid[6, 3] == T
    assert int((b.grid != 0).sum()) == 4
    assert not b.fits(T_UP, 5, 3)


def test_copy_is_independent():
    b = Board()
    c = b.copy()
    c.grid[0, 0] = I
    assert b.grid[0, 0] == 0


# --- rendering -------------------------------------------------------------

def test_render_visible_only():
    b = Board()
    b.grid[HEIGHT - 1, 0] = I
    text = render(b, active_cells=[(HIDDEN_ROWS, 5)], score=100, lines=1, active_piece=T)
    lines = text.splitlines()
    assert lines[0] == "piece: T  score: 100  lines: 1"
    assert lines[1] == "|.....@....|"  # top visible row holds the active cell
    assert lines[-2] == "|I.........|"  # bottom row
    assert lines[-1] == "+----------+"
    assert len(lines) == 1 + 20 + 1


def test_render_with_hidden_rows():
    text = render(Board(), show_hidden=True)
    lines = text.splitlines()
    # 2 hidden rows, divider, 20 visible rows, floor. No status line.
    assert len(lines) == 2 + 1 + 20 + 1
    assert lines[2] == "+----------+"
