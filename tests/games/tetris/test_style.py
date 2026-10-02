"""Style metrics on hand-built boards, and agreement between the live and frames versions."""

import numpy as np
import pytest

from agents.heuristic_agent import HeuristicAgent
from core.evaluate import evaluate
from core.replay import export_frames, make_replay
from games.tetris.env import TetrisEnv
from games.tetris.frames import TetrisFrameRecorder
from games.tetris.style import METRIC_NAMES, StyleTracker, style_metrics, style_of_frames


def board(heights):
    """A 22x10 board whose columns are solid up to the given heights."""
    g = np.zeros((22, 10), dtype=np.uint8)
    for c, h in enumerate(heights):
        if h:
            g[22 - h:, c] = 1
    return g


def test_metrics_by_hand():
    flat = board([2] * 10)
    holey = board([4] + [0] * 9)
    holey[21, 0] = 0  # a hole under column 0
    m = style_metrics([flat, holey, flat, flat], [0, 1, 4, 2])
    assert set(m) == set(METRIC_NAMES)
    assert m["pieces"] == 4 and m["lines"] == 7 and m["max_stack_height"] == 4
    assert m["avg_stack_height"] == (2 + 4 + 2 + 2) / 4
    assert m["avg_holes"] == 1 / 4
    assert m["avg_bumpiness"] == 4 / 4  # only the holey board is bumpy (4 -> 0)
    # Shares are of LINES: 1 single, 1 double (2 lines), 1 tetris (4 lines) out of 7.
    assert (m["share_singles"], m["share_doubles"], m["share_triples"], m["share_tetrises"]) == (1 / 7, 2 / 7, 0, 4 / 7)
    assert m["pieces_per_clear"] == 4 / 3


def test_no_clears():
    m = StyleTracker()
    m.add(board([1] * 10), 0)
    out = m.metrics()
    assert out["pieces_per_clear"] is None and out["share_singles"] == 0.0 and out["lines"] == 0
    assert StyleTracker().metrics()["pieces"] == 0  # empty game doesn't crash


@pytest.mark.parametrize("cap", [None, 40])
def test_frames_and_live_agree(cap):
    res = evaluate(TetrisEnv(), lambda s: HeuristicAgent(), [5], 150, ("lines", "score", "pieces"), record_actions=True)
    g = res["games"][0]
    replay = make_replay("tetris", 5, g["actions"], lines=g["lines"], score=g["score"], pieces=g["pieces"])
    frames = export_frames(TetrisEnv(), replay, TetrisFrameRecorder())["frames"]
    env = TetrisEnv()
    env.reset(seed=5)
    live = StyleTracker()
    for a in g["actions"][:cap]:
        obs, _, _, _, info = env.step(a)
        live.add(obs["board"], info["lines_cleared"])
    assert style_of_frames(frames, cap) == live.metrics()
