import numpy as np
import pytest

from games.tetris.pieces import (
    I, J, L, O, PIECE_TYPES, S, T, Z,
    SimpleRotation,
    UniformRandomizer,
    bounding_box_size,
)

ROT = SimpleRotation()


def normalized(shape):
    """Shift a shape so its min row/col is 0: compares shapes ignoring position."""
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return frozenset((r - min_r, c - min_c) for r, c in shape)


@pytest.mark.parametrize("piece", PIECE_TYPES)
def test_every_rotation_has_four_distinct_cells_inside_box(piece):
    size = bounding_box_size(piece)
    for rot in range(ROT.num_rotations):
        cells = ROT.cells(piece, rot)
        assert len(cells) == 4
        assert len(set(cells)) == 4
        assert all(0 <= r < size and 0 <= c < size for r, c in cells)


@pytest.mark.parametrize("piece", PIECE_TYPES)
def test_rotation_index_wraps_around(piece):
    # Rotating 4 times gets back to the start; index 4 == index 0.
    assert ROT.cells(piece, 4) == ROT.cells(piece, 0)
    assert ROT.cells(piece, -1) == ROT.cells(piece, 3)


def test_spawn_shapes():
    assert set(ROT.cells(I, 0)) == {(1, 0), (1, 1), (1, 2), (1, 3)}
    assert set(ROT.cells(O, 0)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert set(ROT.cells(T, 0)) == {(0, 1), (1, 0), (1, 1), (1, 2)}
    assert set(ROT.cells(S, 0)) == {(0, 1), (0, 2), (1, 0), (1, 1)}
    assert set(ROT.cells(Z, 0)) == {(0, 0), (0, 1), (1, 1), (1, 2)}
    assert set(ROT.cells(J, 0)) == {(0, 0), (1, 0), (1, 1), (1, 2)}
    assert set(ROT.cells(L, 0)) == {(0, 2), (1, 0), (1, 1), (1, 2)}


def test_t_rotates_clockwise():
    # Rotation 1 = one clockwise turn: T points right.
    assert set(ROT.cells(T, 1)) == {(0, 1), (1, 1), (2, 1), (1, 2)}
    # Rotation 2: T points down.
    assert set(ROT.cells(T, 2)) == {(1, 0), (1, 1), (1, 2), (2, 1)}
    # Rotation 3: T points left.
    assert set(ROT.cells(T, 3)) == {(0, 1), (1, 1), (2, 1), (1, 0)}


def test_i_is_vertical_after_one_rotation():
    assert set(ROT.cells(I, 1)) == {(0, 2), (1, 2), (2, 2), (3, 2)}


def test_o_never_changes():
    assert all(ROT.cells(O, r) == ROT.cells(O, 0) for r in range(4))


def test_number_of_distinct_shapes_per_piece():
    # Distinct shapes up to position. The placement layer (1c) relies on this
    # to avoid offering the same final placement twice.
    expected = {I: 2, O: 1, T: 4, S: 2, Z: 2, J: 4, L: 4}
    for piece, n in expected.items():
        shapes = {normalized(ROT.cells(piece, r)) for r in range(4)}
        assert len(shapes) == n, piece


def test_simple_rotation_has_no_kicks():
    assert ROT.kicks(T, 0, 1) == [(0, 0)]


def test_randomizer_is_reproducible_and_covers_all_pieces():
    a = UniformRandomizer(np.random.default_rng(123))
    b = UniformRandomizer(np.random.default_rng(123))
    seq_a = [a.next_piece() for _ in range(500)]
    seq_b = [b.next_piece() for _ in range(500)]
    assert seq_a == seq_b
    assert set(seq_a) == set(PIECE_TYPES)


def test_randomizer_differs_across_seeds():
    a = UniformRandomizer(np.random.default_rng(1))
    b = UniformRandomizer(np.random.default_rng(2))
    assert [a.next_piece() for _ in range(50)] != [b.next_piece() for _ in range(50)]
