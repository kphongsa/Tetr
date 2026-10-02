"""Thoughts: the recorded options match the moves actually played, and a wrong agent is refused."""

import numpy as np
import pytest

from agents.afterstate_value_agent import AfterstateValueAgent, make_value_net
from core.replay import make_replay
from games.tetris.env import TetrisEnv
from games.tetris.features import FEATURE_NAMES, candidates
from scripts.export_frames import tetris_frames
from scripts.thoughts import add_thoughts, checkpoint_for_step, record_options


def agent(seed):
    net = make_value_net(len(FEATURE_NAMES), hidden=(16, 16), seed=seed)
    return AfterstateValueAgent(net, candidates, np.random.default_rng(0))


def game(a, seed=3, pieces=40):
    env = TetrisEnv(max_pieces=pieces)
    obs, info = env.reset(seed=seed)
    actions, done = [], False
    while not done:
        actions.append(a.act(obs, info))
        obs, _, terminated, truncated, info = env.step(actions[-1])
        done = terminated or truncated
    return make_replay("tetris", seed, actions, score=info["score"], lines=info["lines"], pieces=info["pieces"])


def test_options_match_the_game(tmp_path):
    a = agent(0)
    replay = game(a)
    options = record_options(replay, a, k=3)
    assert len(options) == len(replay["actions"])
    assert [o[0]["a"] for o in options] == replay["actions"]  # #1 = the move played
    doc = add_thoughts(tetris_frames(replay), replay, a, tmp_path / "x.pt", k=3)
    frames = doc["frames"]
    assert "options" not in frames[-1]  # nothing left to place after the last move
    # The #1 option's cells are exactly where the next frame says the piece landed.
    assert all(f["options"][0]["cells"] == nxt["cells"] for f, nxt in zip(frames, frames[1:]))
    assert doc["thoughts"]["k"] == 3


def test_wrong_agent_is_refused():
    replay = game(agent(0))
    with pytest.raises(ValueError, match="top choice"):
        record_options(replay, agent(1))


def test_checkpoint_for_step(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "step000000500.pt").write_bytes(b"")
    assert checkpoint_for_step(tmp_path, 500).name == "step000000500.pt"
    assert checkpoint_for_step(tmp_path, 600) is None
