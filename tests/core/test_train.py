"""The generic loop, tested on a fake game and a fake learner (no Tetris)."""

import csv

import pytest

from core.checkpoint import load_checkpoint
from core.replay import load_replay, replay_episode
from core.train import EvalSpec, LoopConfig, linear_epsilon, make_run_dir, train


class CountdownEnv:
    """Game lasts `seed % 5 + 1` steps, reward 1 per step. Optionally 'crashes'
    (raises KeyboardInterrupt, like Ctrl+C) at a given global step."""

    def __init__(self, interrupt_at=None):
        self.interrupt_at = interrupt_at
        self.total = 0

    def reset(self, seed=None):
        self.left = seed % 5 + 1
        return 0, {"left": self.left}

    def step(self, action):
        self.total += 1
        if self.interrupt_at is not None and self.total == self.interrupt_at:
            raise KeyboardInterrupt
        self.left -= 1
        return 0, 1.0, self.left == 0, False, {"left": self.left}


class FakeLearner:
    """'Learns' by counting updates; its whole state is that counter."""

    stat_names = ("loss",)

    def __init__(self):
        self.epsilon = 1.0
        self.seen = []  # (terminated, truncated) per observe
        self.updates = 0

    def act(self, obs, info):
        return 0

    def observe(self, action, reward, terminated, truncated, next_obs, next_info):
        self.seen.append((terminated, truncated))

    def update(self):
        if len(self.seen) <= 3:
            return None  # "buffer filling"
        self.updates += 1
        return {"loss": 0.5}

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state["updates"]

    def eval_agent(self):
        return self


def cfg(**kw):
    base = dict(total_steps=20, max_episode_steps=None, train_seed_start=0, epsilon_decay_steps=10,
                print_every_episodes=1000, checkpoint_every_episodes=3, eval_every_episodes=None)
    base.update(kw)
    return LoopConfig(**base)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_epsilon_schedule():
    c = LoopConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)
    assert linear_epsilon(0, c) == 1.0
    assert linear_epsilon(50, c) == pytest.approx(0.55)
    assert linear_epsilon(100, c) == linear_epsilon(10_000, c) == 0.1


def test_loop_logs_one_row_per_episode(tmp_path):
    c = cfg()
    res = train(CountdownEnv(), FakeLearner(), c, tmp_path, info_keys=("left",), tensorboard=False)
    rows = res.rows
    assert rows[-1]["step"] >= 20 and rows[-2]["step"] < 20  # stops after the episode crossing 20
    assert [r["seed"] for r in rows] == list(range(len(rows)))  # seed = start + episode
    assert [r["steps"] for r in rows[:5]] == [1, 2, 3, 4, 5]
    logged = read_csv(tmp_path / "log.csv")
    assert len(logged) == len(rows)
    assert logged[0]["loss"] == ""  # no update yet
    assert float(logged[-1]["loss"]) == 0.5
    assert float(logged[-1]["epsilon"]) == c.epsilon_end
    assert not res.interrupted


def test_truncation_is_passed_to_the_learner(tmp_path):
    # Games of 5 steps capped at 3: learner must see truncated=True, terminated=False.
    learner = FakeLearner()
    res = train(CountdownEnv(), learner, cfg(total_steps=3, max_episode_steps=3, train_seed_start=4),
                tmp_path, tensorboard=False)
    assert learner.seen[-1] == (False, True)
    assert res.rows[0]["terminated"] == 0


def test_run_dir_refuses_to_mix_runs(tmp_path):
    make_run_dir(tmp_path, "a", {"x": 1})
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "a", {"x": 2})
    d = make_run_dir(tmp_path, "a", {"x": 3}, fresh=True)
    assert '"x": 3' in (d / "config.json").read_text()


def test_checkpoint_holds_progress_and_learner_state(tmp_path):
    res = train(CountdownEnv(), FakeLearner(), cfg(), tmp_path, tensorboard=False, config={"name": "t"})
    ck = load_checkpoint(tmp_path / "checkpoints" / "latest.pt")
    assert ck["progress"]["step"] == res.progress.step
    assert ck["progress"]["episode"] == len(res.rows)
    assert ck["learner"]["updates"] == res.progress.updates > 0
    assert ck["config"] == {"name": "t"}
    assert "torch_rng" in ck and "epsilon" in ck


def test_interrupt_then_resume_continues_without_losing_or_duplicating(tmp_path):
    # Reference: an uninterrupted run.
    full = train(CountdownEnv(), FakeLearner(), cfg(total_steps=40), tmp_path / "full", tensorboard=False)

    # Same run, "Ctrl+C" at global step 17 (in the middle of an episode).
    run = tmp_path / "run"
    first = train(CountdownEnv(interrupt_at=17), FakeLearner(), cfg(total_steps=40), run, tensorboard=False)
    assert first.interrupted
    ck = load_checkpoint(run / "checkpoints" / "latest.pt")
    assert ck["progress"]["step"] == 16  # the step that raised never completed
    assert ck["progress"]["episode"] == len(first.rows)  # the unfinished episode will be replayed

    learner = FakeLearner()
    second = train(CountdownEnv(), learner, cfg(total_steps=40), run, tensorboard=False, resume=True)
    assert not second.interrupted
    assert learner.updates > ck["learner"]["updates"]  # continued from the saved count, not from 0
    logged = read_csv(run / "log.csv")
    episodes = [int(r["episode"]) for r in logged]
    assert episodes == list(range(len(episodes)))  # no gaps, no duplicates
    # Same games in the same order as the uninterrupted run:
    assert [r["seed"] for r in logged] == [str(r["seed"]) for r in full.rows[: len(logged)]]


def test_resume_drops_rows_logged_after_the_checkpoint(tmp_path):
    # A hard crash (no Ctrl+C save) loses everything after the last checkpoint,
    # but those episodes are already in log.csv. Simulate: finish a run, rewind
    # its checkpoint to episode 3, resume. Rows 3+ must not appear twice.
    import torch

    c = cfg(total_steps=15)  # episodes of 1..5 steps -> 5 episodes
    train(CountdownEnv(), FakeLearner(), c, tmp_path, tensorboard=False)
    path = tmp_path / "checkpoints" / "latest.pt"
    ck = load_checkpoint(path)
    ck["progress"].update(step=6, episode=3)
    torch.save(ck, path)
    train(CountdownEnv(), FakeLearner(), c, tmp_path, tensorboard=False, resume=True)
    episodes = [int(r["episode"]) for r in read_csv(tmp_path / "log.csv")]
    assert episodes == list(range(len(episodes)))


def test_periodic_evaluation_best_checkpoint_and_replays(tmp_path):
    spec = EvalSpec(env=CountdownEnv(), seeds=[2, 3, 4], max_steps=None, info_keys=("left",),
                    reference={"left_mean": 0.0}, game="countdown")
    c = cfg(total_steps=30, eval_every_episodes=4, best_metric="left")
    res = train(CountdownEnv(), FakeLearner(), c, tmp_path, eval_spec=spec, tensorboard=False)
    evals = read_csv(tmp_path / "eval.csv")
    assert len(evals) == len(res.evals) >= 2
    assert evals[0]["ref_left_mean"] == "0.0" and evals[0]["games"] == "3"
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    replays = sorted((tmp_path / "replays").glob("*.json"))
    assert len(replays) == 2 * len(evals)  # 2 per evaluation
    r = load_replay(replays[0])
    assert r["seed"] == 2 and r["metadata"]["step"] == int(evals[0]["step"])
    assert replay_episode(CountdownEnv(), r)["terminated"]


def test_ctrl_c_during_evaluation_still_saves_and_resumes(tmp_path):
    # Once the agent is good, one evaluation takes minutes, so that's where a
    # Ctrl+C is most likely to land. The eval env "crashes" on its 2nd step.
    spec = EvalSpec(env=CountdownEnv(interrupt_at=2), seeds=[4], max_steps=None, info_keys=("left",))
    c = cfg(total_steps=40, eval_every_episodes=4, checkpoint_every_episodes=1000)
    res = train(CountdownEnv(), FakeLearner(), c, tmp_path, eval_spec=spec, tensorboard=False)
    assert res.interrupted
    ck = load_checkpoint(tmp_path / "checkpoints" / "latest.pt")
    # The 4 episodes before the evaluation are finished and saved, nothing lost.
    assert ck["progress"]["episode"] == 4 == len(res.rows)
    assert ck["progress"]["step"] == res.rows[-1]["step"]

    resumed = train(CountdownEnv(), FakeLearner(), cfg(total_steps=40, checkpoint_every_episodes=1000),
                    tmp_path, tensorboard=False, resume=True)
    assert resumed.rows[0]["episode"] == 4 and not resumed.interrupted
