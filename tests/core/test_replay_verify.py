"""verify_replay / export_frames: generic, so tested with a tiny fake game."""

import pytest

from core.replay import export_frames, make_replay, verify_replay


class Countdown:
    """Fake game: seed = number of steps until game over; info["n"] counts down."""

    def reset(self, seed=None):
        self.n = seed
        return self.n, {"n": self.n}

    def step(self, action):
        self.n -= 1
        return self.n, 0.0, self.n == 0, False, {"n": self.n}


class Recorder:
    def header(self):
        return {"kind": "countdown"}

    def start(self, env, obs, info):
        return {"n": info["n"]}

    def step(self, env, action, obs, info):
        return {"n": info["n"], "a": action}


def test_verify_ok_and_mismatches():
    assert verify_replay(Countdown(), make_replay("c", 3, [0, 0, 0], n=0, terminated=True, step=5)) == []
    # Recorded result differs from the re-simulation -> one problem per key.
    problems = verify_replay(Countdown(), make_replay("c", 3, [0, 0, 0], n=1, terminated=False))
    assert len(problems) == 2 and problems[0].startswith("n:")
    # More actions than the game lasts.
    assert "game ended" in verify_replay(Countdown(), make_replay("c", 2, [0, 0, 0]))[0]


def test_make_replay_records_git_unless_given():
    assert "git" in make_replay("c", 1, [0])["metadata"]
    assert make_replay("c", 1, [0], git="abc")["metadata"]["git"] == "abc"


def test_export_frames():
    doc = export_frames(Countdown(), make_replay("c", 2, [7, 8], n=0, terminated=True), Recorder())
    assert doc["format"] == "frames" and doc["header"] == {"kind": "countdown"}
    assert doc["frames"] == [{"n": 2}, {"n": 1, "a": 7}, {"n": 0, "a": 8}]
    with pytest.raises(ValueError):  # never export a game that doesn't match its record
        export_frames(Countdown(), make_replay("c", 2, [7, 8], n=5), Recorder())
