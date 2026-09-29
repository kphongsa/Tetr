import pytest
import torch

from core.checkpoint import load_checkpoint, save_checkpoint
from core.replay import load_replay, make_replay, replay_episode, save_replay


def test_checkpoint_round_trip(tmp_path):
    payload = {"w": torch.arange(3.0), "n": 7, "rng": {"state": 2**70, "name": "PCG64"}}
    save_checkpoint(tmp_path / "c" / "latest.pt", payload)
    back = load_checkpoint(tmp_path / "c" / "latest.pt")
    assert torch.equal(back["w"], payload["w"]) and back["n"] == 7 and back["rng"] == payload["rng"]
    assert not list((tmp_path / "c").glob("*.tmp"))  # temp file renamed away


class Steps:
    def reset(self, seed=None):
        self.n = seed
        return 0, {"n": self.n}

    def step(self, action):
        self.n -= 1
        return 0, 0.0, self.n == 0, False, {"n": self.n}


def test_replay_round_trip_and_mismatch(tmp_path):
    r = make_replay("steps", 3, [0, 0, 0], step=10)
    save_replay(tmp_path / "r.json", r)
    back = load_replay(tmp_path / "r.json")
    assert back == r
    assert replay_episode(Steps(), back, ("n",)) == {"terminated": True, "truncated": False, "n": 0}
    with pytest.raises(ValueError):  # more actions than the game allows
        replay_episode(Steps(), make_replay("steps", 2, [0, 0, 0]))
