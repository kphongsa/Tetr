"""Every replay saved under runs/ re-simulates to its recorded score, lines and pieces.

Marked slow (~90 s for the ~520 replays from step 3), so it's skipped by a
plain `pytest`. Run it with:   pytest -m slow
runs/ is git-ignored, so on a fresh clone there's nothing to check (skipped).
"""

from pathlib import Path

import pytest

from core.replay import load_replay, verify_replay
from games.tetris.env import TetrisEnv
from scripts.verify_replays import replay_paths

PATHS = replay_paths(Path(__file__).resolve().parents[2] / "runs")


@pytest.mark.slow
@pytest.mark.skipif(not PATHS, reason="no saved replays under runs/")
def test_all_saved_replays_verify():
    failures = {}
    for path in PATHS:
        problems = verify_replay(TetrisEnv(), load_replay(path))
        if problems:
            failures[str(path)] = problems
    assert not failures, failures
