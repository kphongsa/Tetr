"""make_gif: timeline logic, drawing, and GIF/MP4 writing on a short real game."""

import shutil

import pytest
from PIL import Image

from agents.heuristic_agent import HeuristicAgent
from core.evaluate import evaluate
from core.replay import make_replay
from games.tetris.draw import draw_panel, panel_size, side_by_side
from games.tetris.env import TetrisEnv
from scripts.export_frames import tetris_frames
from scripts.make_gif import render_ticks, save_gif, save_mp4, timeline


def test_timeline_single_and_flash():
    # 4 pieces (5 frames), a line clear at piece 2.
    assert timeline([5], 0, 4, 1, [set()]) == [[(i, False)] for i in range(5)]
    assert timeline([5], 0, 4, 1, [{2}]) == [[(0, False)], [(1, False)], [(2, True)], [(2, False)],
                                             [(3, False)], [(4, False)]]
    assert timeline([5], 0, 4, 2, [{2}]) == [[(0, False)], [(2, False)], [(4, False)]]  # no flash with --every


def test_timeline_compare_freezes_finished_game():
    # Game A has 3 frames (ended), B has 6; B clears at piece 4.
    ticks = timeline([3, 6], 0, 5, 1, [set(), {4}])
    assert ticks[-1] == [(2, False), (5, False)]  # A frozen on its last frame
    assert [(2, False), (4, True)] in ticks  # B flashes while A just stays put


@pytest.fixture(scope="module")
def doc():
    res = evaluate(TetrisEnv(), lambda s: HeuristicAgent(), [3], 40, ("lines", "score", "pieces"), record_actions=True)
    g = res["games"][0]
    return tetris_frames(make_replay("tetris", 3, g["actions"], lines=g["lines"], score=g["score"], pieces=g["pieces"]))


def test_draw_panel(doc):
    img = draw_panel(doc["frames"][10], doc["header"], cell=12, title="step 5")
    assert img.size == panel_size(doc["header"], 12) and img.width % 2 == 0 and img.height % 2 == 0
    pair = side_by_side([img, img])
    assert pair.height == img.height and pair.width > 2 * img.width and pair.width % 2 == 0


def test_save_gif(doc, tmp_path):
    ticks = timeline([len(doc["frames"])], 0, 20, 1, [set()])
    n = save_gif(render_ticks([doc], ticks, ["x"], 8), tmp_path / "g.gif", fps=10)
    with Image.open(tmp_path / "g.gif") as im:
        assert n == 21 and im.n_frames == 21


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_save_mp4(doc, tmp_path):
    ticks = timeline([len(doc["frames"]), len(doc["frames"])], 0, 10, 1, [set(), set()])
    assert save_mp4(render_ticks([doc, doc], ticks, ["a", "b"], 8), tmp_path / "v.mp4", fps=10) == 11
    assert (tmp_path / "v.mp4").stat().st_size > 0
