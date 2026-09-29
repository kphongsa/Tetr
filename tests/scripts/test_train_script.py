"""End-to-end: the real Tetris training setup, tiny sizes."""

import csv
import json
from pathlib import Path

import pytest
import torch

from core.checkpoint import load_checkpoint
from core.replay import load_replay, replay_episode
from core.train import train
from games.tetris.env import TetrisEnv
from scripts.train import INFO_KEYS, build, eval_spec, full_config

ROOT = Path(__file__).resolve().parents[2]


def tiny_config(total_steps=300):
    cfg = json.loads((ROOT / "configs" / "smoke.json").read_text())
    cfg["loop"].update(total_steps=total_steps, epsilon_decay_steps=200, print_every_episodes=1000,
                       checkpoint_every_episodes=5, eval_every_episodes=6, eval_games=2)
    cfg["learner"].update(learning_starts=100, batch_size=16)
    return cfg


def run(run_dir, config, resume=False, with_eval=False):
    run_dir.mkdir(exist_ok=True)
    env, learner, loop = build(config)
    spec = eval_spec(loop, "test") if with_eval else None
    result = train(env, learner, loop, run_dir, INFO_KEYS, eval_spec=spec, tensorboard=False,
                   resume=resume, config=config)
    return result, learner


def without_timing(rows):
    return [{k: v for k, v in r.items() if k not in ("steps_per_sec", "elapsed_sec")} for r in rows]


def test_training_is_reproducible(tmp_path):
    a, _ = run(tmp_path / "a", tiny_config())
    b, _ = run(tmp_path / "b", tiny_config())
    assert without_timing(a.rows) == without_timing(b.rows)
    assert a.rows[-1]["loss"] != ""  # learning actually happened


def test_training_seeds_must_avoid_eval_seeds():
    cfg = tiny_config()
    cfg["loop"]["train_seed_start"] = 10_050
    with pytest.raises(ValueError):
        build(cfg)


def test_unknown_config_key_is_an_error():
    cfg = tiny_config()
    cfg["learner"]["learning_rate"] = 0.1  # typo for "lr"
    with pytest.raises(TypeError):
        full_config(cfg)


def test_learner_state_round_trip():
    _, learner, _ = build(tiny_config())
    env = TetrisEnv()
    obs, info = env.reset(seed=1_000_000)
    for _ in range(200):  # enough to start updating
        a = learner.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        learner.observe(a, r, term, trunc, obs, info)
        learner.update()
        if term:
            obs, info = env.reset(seed=1_000_001)
    state = learner.state_dict()

    _, fresh, _ = build(tiny_config())
    fresh.load_state_dict(state)
    for p, q in zip(learner.net.parameters(), fresh.net.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(learner.target_net.parameters(), fresh.target_net.parameters()):
        assert torch.equal(p, q)
    assert fresh.updates == learner.updates > 0
    assert fresh.optimizer.state_dict()["state"][0]["step"] == learner.optimizer.state_dict()["state"][0]["step"]
    assert fresh.agent.rng.random() == learner.agent.rng.random()  # rng continues identically


def test_stop_and_resume_continues_progress(tmp_path):
    run_dir = tmp_path / "run"
    first, learner = run(run_dir, tiny_config(total_steps=150), with_eval=True)
    ck = load_checkpoint(run_dir / "checkpoints" / "latest.pt")
    assert ck["progress"]["step"] == first.progress.step >= 150

    # "Restart the program" with a bigger budget: new objects, loaded from disk.
    second, resumed = run(run_dir, tiny_config(total_steps=400), resume=True, with_eval=True)
    assert second.rows[0]["episode"] == ck["progress"]["episode"]  # continues, doesn't restart at 0
    assert second.rows[0]["step"] > ck["progress"]["step"]
    assert second.progress.step >= 400
    assert resumed.updates > ck["learner"]["updates"]

    with open(run_dir / "log.csv") as f:
        episodes = [int(r["episode"]) for r in csv.DictReader(f)]
    assert episodes == list(range(len(episodes)))


def test_eval_replays_resimulate_exactly(tmp_path):
    result, _ = run(tmp_path / "run", tiny_config(), with_eval=True)
    assert result.evals and (tmp_path / "run" / "checkpoints" / "best.pt").exists()
    paths = sorted((tmp_path / "run" / "replays").glob("*.json"))
    assert len(paths) == 2 * len(result.evals)
    for path in paths[:2]:
        r = load_replay(path)
        assert r["game"] == "tetris" and r["seed"] in (10_000, 10_001)
        final = replay_episode(TetrisEnv(), r, INFO_KEYS)
        assert {k: final[k] for k in INFO_KEYS} == {k: r["metadata"][k] for k in INFO_KEYS}
