"""The generic loop, tested on a fake game and a fake learner (no Tetris, no torch)."""

import csv

import pytest

from core.train import LoopConfig, linear_epsilon, make_run_dir, train


class CountdownEnv:
    """Game lasts `seed % 5 + 1` steps, reward 1 per step."""

    def reset(self, seed=None):
        self.left = seed % 5 + 1
        return 0, {"left": self.left}

    def step(self, action):
        self.left -= 1
        return 0, 1.0, self.left == 0, False, {"left": self.left}


class FakeLearner:
    stat_names = ("loss",)

    def __init__(self):
        self.epsilon = 1.0
        self.seen = []  # (terminated, truncated) per observe

    def act(self, obs, info):
        return 0

    def observe(self, action, reward, terminated, truncated, next_obs, next_info):
        self.seen.append((terminated, truncated))

    def update(self):
        return {"loss": 0.5} if len(self.seen) > 3 else None  # "buffer filling" at first


def test_epsilon_schedule():
    cfg = LoopConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    assert linear_epsilon(0, cfg) == 1.0
    assert linear_epsilon(50, cfg) == pytest.approx(0.55)
    assert linear_epsilon(100, cfg) == linear_epsilon(10_000, cfg) == 0.1


def test_loop_logs_one_row_per_episode(tmp_path):
    cfg = LoopConfig(total_steps=20, max_episode_steps=None, train_seed_start=0,
                     epsilon_decay_steps=10, print_every_episodes=1000)
    learner = FakeLearner()
    rows = train(CountdownEnv(), learner, cfg, tmp_path, info_keys=("left",), tensorboard=False)
    assert rows[-1]["step"] >= 20 and rows[-2]["step"] < 20  # stops after the episode crossing 20
    assert [r["seed"] for r in rows] == list(range(len(rows)))  # seed = start + episode
    assert [r["steps"] for r in rows[:5]] == [1, 2, 3, 4, 5]
    with open(tmp_path / "log.csv") as f:
        logged = list(csv.DictReader(f))
    assert len(logged) == len(rows)
    assert logged[0]["loss"] == ""  # no update yet
    assert float(logged[-1]["loss"]) == 0.5
    assert float(logged[-1]["epsilon"]) == cfg.epsilon_end


def test_truncation_is_passed_to_the_learner(tmp_path):
    # Games of 5 steps capped at 3: learner must see truncated=True, terminated=False.
    cfg = LoopConfig(total_steps=3, max_episode_steps=3, train_seed_start=4, print_every_episodes=1000)
    learner = FakeLearner()
    rows = train(CountdownEnv(), learner, cfg, tmp_path, tensorboard=False)
    assert learner.seen[-1] == (False, True)
    assert rows[0]["terminated"] == 0


def test_run_dir_refuses_to_mix_runs(tmp_path):
    make_run_dir(tmp_path, "a", {"x": 1})
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "a", {"x": 2})
    d = make_run_dir(tmp_path, "a", {"x": 3}, fresh=True)
    assert '"x": 3' in (d / "config.json").read_text()
