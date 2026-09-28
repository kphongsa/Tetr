import numpy as np
import pytest

from games.tetris.board import HEIGHT, WIDTH, Board
from games.tetris.engine import LINE_CLEAR_SCORES, GameOverError, TetrisEngine
from games.tetris.pieces import I, J, L, O, PIECE_TYPES, S, T, Z
from games.tetris.render_text import render_game

FILL = J  # arbitrary piece id used for pre-built "junk" blocks


def make_engine(board=None, seed=0):
    """Engine with no piece spawned yet, so a test can choose the piece."""
    return TetrisEngine(np.random.default_rng(seed), board=board, spawn_first=False)


def cells(engine):
    return set(engine.active_cells())


def leftmost(engine):
    return min(c for _, c in engine.active_cells())


def shift_to(engine, target_left_col):
    """Shift until the piece's leftmost cell is at the target column."""
    while leftmost(engine) < target_left_col:
        assert engine.shift(+1)
    while leftmost(engine) > target_left_col:
        assert engine.shift(-1)


def board_with_rows_filled_except_last_col(n_rows):
    """Bottom n_rows filled in columns 0..8, column 9 empty: a well for an I."""
    b = Board()
    b.grid[HEIGHT - n_rows:, : WIDTH - 1] = FILL
    return b


# --- spawn -------------------------------------------------------------------

def test_spawn_positions_are_centred_in_hidden_rows():
    e = make_engine()
    e.spawn(T)
    assert cells(e) == {(0, 4), (1, 3), (1, 4), (1, 5)}
    e.spawn(O)
    assert cells(e) == {(0, 4), (0, 5), (1, 4), (1, 5)}
    e.spawn(I)
    assert cells(e) == {(1, 3), (1, 4), (1, 5), (1, 6)}


@pytest.mark.parametrize("piece", PIECE_TYPES)
def test_every_piece_spawns_entirely_in_hidden_rows(piece):
    e = make_engine()
    assert e.spawn(piece)
    assert all(r in (0, 1) for r, _ in e.active_cells())


@pytest.mark.parametrize("piece", PIECE_TYPES)
def test_every_rotation_reachable_right_after_spawn(piece):
    # The placement layer rotates before moving, so this must hold.
    e = make_engine()
    e.spawn(piece)
    for _ in range(4):
        assert e.rotate(+1)
    for _ in range(4):
        assert e.rotate(-1)


def test_constructor_spawns_a_random_piece_by_default():
    e = TetrisEngine(np.random.default_rng(0))
    assert e.piece in PIECE_TYPES and not e.game_over


# --- shift -------------------------------------------------------------------

def test_shift_stops_at_left_wall():
    e = make_engine()
    e.spawn(T)  # cells span cols 3..5
    assert e.shift(-1) and e.shift(-1) and e.shift(-1)
    assert leftmost(e) == 0
    before = cells(e)
    assert not e.shift(-1)
    assert cells(e) == before  # failed move changes nothing


def test_shift_stops_at_right_wall():
    e = make_engine()
    e.spawn(I)  # cols 3..6
    for _ in range(3):
        assert e.shift(+1)
    assert not e.shift(+1)
    assert max(c for _, c in e.active_cells()) == WIDTH - 1


def test_shift_blocked_by_other_block():
    b = Board()
    b.grid[1, 2] = FILL
    e = make_engine(b)
    e.spawn(T)  # bottom row cells at (1,3)..(1,5)
    assert not e.shift(-1)


# --- rotate ------------------------------------------------------------------

def test_rotate_in_open_space_and_full_circle():
    e = make_engine()
    e.spawn(T)
    for _ in range(5):
        e.soft_drop()
    start = cells(e)
    assert e.rotate(+1)
    assert cells(e) != start
    assert e.rotate(+1) and e.rotate(+1) and e.rotate(+1)
    assert cells(e) == start
    assert e.rotate(-1) and e.rotate(+1)  # counter-clockwise undoes clockwise
    assert cells(e) == start


def test_rotating_vertical_i_against_left_wall_fails():
    e = make_engine()
    e.spawn(I)
    for _ in range(5):
        e.soft_drop()
    assert e.rotate(+1)  # vertical, in box column 2
    shift_to(e, 0)  # hug the left wall
    before = (cells(e), e.rotation)
    # Flat I would need box columns 0..3 = board columns -2..1: through the wall.
    # No wall kicks, so both directions fail and nothing changes.
    assert not e.rotate(+1)
    assert not e.rotate(-1)
    assert (cells(e), e.rotation) == before


def test_rotating_t_against_right_wall_fails():
    e = make_engine()
    e.spawn(T)
    for _ in range(5):
        e.soft_drop()
    assert e.rotate(-1)  # T pointing left: uses box columns 0..1
    shift_to(e, WIDTH - 2)  # cells in cols 8..9; box column 2 is col 10
    assert not e.rotate(+1)  # pointing up needs box column 2 -> off the board
    assert not e.rotate(-1)  # pointing down also needs it


def test_rotation_blocked_by_block():
    b = Board()
    e = make_engine(b)
    e.spawn(T)
    for _ in range(5):
        e.soft_drop()
    # Pointing-right T needs box cell (2,1) which is currently empty.
    b.grid[e.row + 2, e.col + 1] = FILL
    assert not e.rotate(+1)
    assert e.rotation == 0


# --- drops and lock ----------------------------------------------------------

def test_soft_drop_moves_one_row_and_stops_on_floor():
    e = make_engine()
    e.spawn(O)
    row0 = e.row
    assert e.soft_drop()
    assert e.row == row0 + 1
    while e.soft_drop():
        pass
    assert max(r for r, _ in e.active_cells()) == HEIGHT - 1


def test_hard_drop_locks_on_floor_and_spawns_next():
    e = make_engine()
    e.spawn(T)
    cleared = e.hard_drop()
    assert cleared == 0
    assert e.pieces_placed == 1
    # T pointing up on the floor: nub at row 20, flat part on row 21.
    assert e.board.grid[20, 4] == T
    assert list(e.board.grid[21, 3:6]) == [T, T, T]
    assert int((e.board.grid != 0).sum()) == 4
    # A fresh piece is back at the top.
    assert all(r in (0, 1) for r, _ in e.active_cells())


def test_hard_drop_lands_on_existing_blocks():
    b = Board()
    b.grid[15, 4] = FILL
    e = make_engine(b)
    e.spawn(O)  # cols 4..5
    e.hard_drop()
    assert b.grid[14, 4] == O and b.grid[13, 5] == O
    assert b.grid[15, 5] == 0  # didn't slide past the block


# --- line clears and scoring --------------------------------------------------

@pytest.mark.parametrize(
    "n_lines, name",
    [(1, "single"), (2, "double"), (3, "triple"), (4, "tetris")],
)
def test_line_clears(n_lines, name):
    e = make_engine(board_with_rows_filled_except_last_col(n_lines))
    e.spawn(I)
    assert e.rotate(+1)  # vertical
    shift_to(e, WIDTH - 1)
    assert e.hard_drop() == n_lines
    assert e.lines == n_lines
    assert e.score == LINE_CLEAR_SCORES[n_lines] == {1: 100, 2: 300, 3: 500, 4: 800}[n_lines]
    # Only the part of the I that stuck out above the filled rows remains,
    # and it has fallen to the bottom of column 9.
    remaining = 4 - n_lines
    assert int((e.board.grid != 0).sum()) == remaining
    if remaining:
        assert all(e.board.grid[HEIGHT - remaining:, WIDTH - 1] == I)


def test_non_adjacent_rows_clear_and_middle_row_falls():
    b = Board()
    b.grid[19, :9] = FILL
    b.grid[20, 1:9] = FILL  # col 0 AND col 9 empty: won't become full
    b.grid[21, :9] = FILL
    e = make_engine(b)
    e.spawn(I)
    e.rotate(+1)
    shift_to(e, 9)
    assert e.hard_drop() == 2  # rows 19 and 21
    assert e.score == 300
    # Old row 20 (with the I cell now filling col 9) is now the bottom row.
    assert b.grid[21, 0] == 0
    assert list(b.grid[21, 1:10]) == [FILL] * 8 + [I]
    # Top of the I (old row 18) fell 2 rows to row 20.
    assert b.grid[20, 9] == I
    assert int((b.grid != 0).sum()) == 8 + 1 + 1


def test_blocks_above_cleared_line_fall():
    b = board_with_rows_filled_except_last_col(1)
    b.grid[15, 0] = FILL  # floating marker block
    e = make_engine(b)
    e.spawn(I)
    e.rotate(+1)
    shift_to(e, 9)
    e.hard_drop()
    assert b.grid[16, 0] == FILL and b.grid[15, 0] == 0


def test_scores_accumulate():
    e = make_engine(board_with_rows_filled_except_last_col(1))
    e.spawn(I)
    e.rotate(+1)
    shift_to(e, 9)
    e.hard_drop()  # single; leaves 3 I cells in col 9, rows 19..21
    e.board.grid[18, :9] = FILL  # new almost-full row right above them
    e.spawn(I)
    e.rotate(+1)
    shift_to(e, 9)
    e.hard_drop()  # lands on rows 15..18 -> row 18 clears: another single
    assert e.score == 200 and e.lines == 2


# --- game over ---------------------------------------------------------------

def test_game_over_when_spawn_blocked():
    b = Board()
    b.grid[1, 4] = FILL
    e = make_engine(b)
    assert not e.spawn(T)
    assert e.game_over


def test_game_over_after_stack_reaches_spawn_area():
    b = Board()
    b.grid[2:, :9] = FILL  # visible area packed, col 9 empty so nothing clears
    e = make_engine(b, seed=5)
    e.spawn(O)  # can't move down at all: rows 2+ are full under it
    assert e.hard_drop() == 0
    # The O locked in the hidden rows at cols 4..5. Every piece's spawn
    # position overlaps col 4 or 5 in rows 0..1, so whatever comes next fails.
    assert e.game_over
    assert "GAME OVER" in render_game(e)


def test_locking_in_hidden_rows_is_not_itself_game_over():
    # Our rule: the game ends only when a NEW piece can't spawn. A piece that
    # locks partly/fully in the hidden rows is fine if the spawn spot is free.
    b = Board()
    b.grid[2:, :3] = FILL  # tall stack in cols 0..2 up to the hidden rows
    e = make_engine(b)
    e.spawn(O)
    shift_to(e, 0)  # O over cols 0..1, sitting in rows 0..1
    e.hard_drop()  # can't fall: locks entirely in the hidden rows
    assert b.grid[0, 0] == O
    assert not e.game_over  # spawn area (cols 3..6) is still free


def test_moves_after_game_over_raise():
    b = Board()
    b.grid[0:2, :] = FILL
    e = make_engine(b)
    e.spawn(T)
    for move in (lambda: e.shift(1), e.rotate, e.soft_drop, e.hard_drop, e.lock, e.spawn):
        with pytest.raises(GameOverError):
            move()


# --- reproducibility ---------------------------------------------------------

def test_same_seed_same_piece_sequence():
    def play(seed):
        e = TetrisEngine(np.random.default_rng(seed))
        seq = []
        for _ in range(30):
            seq.append(e.piece)
            e.hard_drop()
            if e.game_over:
                break
        return seq, e.board.grid.copy()

    seq_a, grid_a = play(42)
    seq_b, grid_b = play(42)
    assert seq_a == seq_b
    assert np.array_equal(grid_a, grid_b)
    assert play(43)[0] != seq_a
