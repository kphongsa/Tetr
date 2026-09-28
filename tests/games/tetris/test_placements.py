import numpy as np
import pytest

from games.tetris.board import HEIGHT, WIDTH, Board
from games.tetris.engine import TetrisEngine
from games.tetris.pieces import I, J, L, O, PIECE_TYPES, S, T, Z
from games.tetris.placements import (
    NUM_ACTIONS,
    apply_placement,
    decode_action,
    encode_action,
    legal_placements,
)

FILL = J


def engine_with(piece, board=None):
    e = TetrisEngine(np.random.default_rng(0), board=board, spawn_first=False)
    e.spawn(piece)
    return e


def test_action_encoding_round_trip():
    assert NUM_ACTIONS == 40
    seen = set()
    for rot in range(4):
        for col in range(WIDTH):
            a = encode_action(rot, col)
            assert decode_action(a) == (rot, col)
            seen.add(a)
    assert seen == set(range(40))


@pytest.mark.parametrize(
    "piece, expected",
    # Distinct final spots on an empty 10-wide board. A shape that is w cells
    # wide fits in 10 - w + 1 columns. E.g. T: 8 (flat up) + 9 + 8 + 9 = 34.
    # I/S/Z have only 2 distinct shapes, O has 1: duplicates are removed.
    [(I, 7 + 10), (O, 9), (T, 34), (S, 8 + 9), (Z, 8 + 9), (J, 34), (L, 34)],
)
def test_placement_counts_on_empty_board(piece, expected):
    assert len(legal_placements(engine_with(piece))) == expected


@pytest.mark.parametrize("piece", PIECE_TYPES)
def test_placements_rest_on_something_and_leave_piece_untouched(piece):
    b = Board()
    b.grid[15:, [0, 3, 4, 8]] = FILL  # some bumps to land on
    e = engine_with(piece, b)
    before = (e.rotation, e.row, e.col)
    placements = legal_placements(e)
    assert (e.rotation, e.row, e.col) == before  # enumeration restored the piece
    assert len({p.cells for p in placements}) == len(placements)  # no duplicates
    for p in placements:
        assert len(p.cells) == 4
        assert min(c for _, c in p.cells) == p.column
        assert all(b.grid[r, c] == 0 for r, c in p.cells)
        # "Resting": at least one cell sits on the floor or on a block.
        assert any(r == HEIGHT - 1 or b.grid[r + 1, c] != 0 for r, c in p.cells)


@pytest.mark.parametrize("piece", PIECE_TYPES)
def test_apply_placement_lands_exactly_where_promised(piece):
    b = Board()
    b.grid[18:, 2:5] = FILL
    for p in legal_placements(engine_with(piece, b)):
        e = engine_with(piece, b.copy())
        apply_placement(e, p)
        # Cells that changed = where the piece locked (no line can clear here).
        locked = {(int(r), int(c)) for r, c in zip(*np.nonzero(e.board.grid != b.grid))}
        assert locked == set(p.cells)


@pytest.mark.parametrize("seed", range(15))
def test_fast_landing_matches_real_hard_drop_on_random_boards(seed):
    # legal_placements computes landings with a lookup table; apply_placement
    # uses the engine's real moves. Random boards (with holes and overhangs
    # from random play) check that both always agree, for every placement.
    rng = np.random.default_rng([seed, 99])
    game = TetrisEngine(np.random.default_rng(seed))
    while not game.game_over:
        placements = legal_placements(game)
        for p in placements:
            trial = TetrisEngine(np.random.default_rng(0), board=game.board.copy(),
                                 spawn_first=False)
            trial.spawn(game.piece)
            apply_placement(trial, p)
            changed = {(int(r), int(c))
                       for r, c in zip(*np.nonzero(trial.board.grid != game.board.grid))}
            if trial.lines == 0:  # a line clear shifts rows; compare only simple cases
                assert changed == set(p.cells)
        apply_placement(game, placements[int(rng.integers(len(placements)))])


def test_wall_of_blocks_limits_reachable_columns():
    # A full-height wall in column 2 (hidden rows included) blocks shifting
    # left, even though the squares on the other side are empty. The legal
    # set shrinks: this is the "variable legal actions" case from CLAUDE.md.
    b = Board()
    b.grid[:, 2] = FILL
    placements = legal_placements(engine_with(T, b))
    assert placements
    assert all(p.column >= 3 for p in placements)


def test_high_stack_blocks_some_rotations():
    # Blocks at the top of the visible area (rows 2-3) in cols 4-5 stop the
    # I from turning vertical at spawn either way (clockwise it would occupy
    # col 5, counter-clockwise col 4), so only flat placements remain.
    b = Board()
    b.grid[2:4, 4:6] = FILL
    placements = legal_placements(engine_with(I, b))
    assert placements
    assert all(p.rotation == 0 for p in placements)


def test_no_placements_after_game_over():
    b = Board()
    b.grid[0:2, :] = FILL
    e = TetrisEngine(np.random.default_rng(0), board=b, spawn_first=False)
    e.spawn(T)
    assert e.game_over
    assert legal_placements(e) == []
