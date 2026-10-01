"""Tetris frames: a real game exported frame by frame stays consistent with the engine."""

import numpy as np
import pytest

from agents.heuristic_agent import HeuristicAgent
from core.evaluate import evaluate
from core.replay import export_frames, make_replay
from games.tetris.env import TetrisEnv
from games.tetris.features import board_features, column_heights
from games.tetris.frames import TetrisFrameRecorder, board_before_clear, board_from_string, board_to_string
from games.tetris.render_text import render_frame


@pytest.fixture(scope="module")
def doc():
    # A heuristic game cut at 250 pieces: long enough to contain line clears.
    res = evaluate(TetrisEnv(), lambda s: HeuristicAgent(), [3], 250, ("lines", "score", "pieces"),
                   record_actions=True)
    g = res["games"][0]
    replay = make_replay("tetris", 3, g["actions"], lines=g["lines"], score=g["score"], pieces=g["pieces"])
    return export_frames(TetrisEnv(), replay, TetrisFrameRecorder())


def test_board_string_round_trip():
    grid = np.random.default_rng(0).integers(0, 8, size=(22, 10)).astype(np.uint8)
    assert np.array_equal(board_from_string(board_to_string(grid)), grid)


def test_frames_match_the_game(doc):
    frames = doc["frames"]
    assert len(frames) == 251 and [f["n"] for f in frames] == list(range(251))
    assert frames[0]["board"] == "0" * 220 and frames[0]["action"] is None
    assert frames[-1]["score"] == doc["metadata"]["score"]
    assert sum(len(f["cleared_rows"]) for f in frames) == frames[-1]["lines"] > 0
    for prev, f in zip(frames, frames[1:]):
        assert f["piece"] == prev["next"]  # the waiting piece is the next one placed
        grid = board_from_string(f["board"])
        assert f["height"] == column_heights(grid).max()
        assert f["holes"] == board_features(grid, 0)[2]
        # Before the clear: the new piece is on the previous board, and the
        # cleared rows are exactly the full rows.
        before = board_before_clear(prev["board"], f["piece"], f["cells"])
        assert sorted(np.flatnonzero((before != 0).all(axis=1))) == f["cleared_rows"]
        assert all(before[r, c] == f["piece"] for r, c in f["cells"])


def test_render_frame(doc):
    frames = doc["frames"]
    i = next(i for i, f in enumerate(frames) if f["cleared_rows"])
    text = render_frame(frames[i], frames[i - 1], flash=True)
    assert text.count("<- clear") == len(frames[i]["cleared_rows"]) and "@" in text
    assert "<- clear" not in render_frame(frames[i]) and f"piece {i}" in render_frame(frames[i])
