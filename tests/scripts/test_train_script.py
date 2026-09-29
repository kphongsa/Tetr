import json
from pathlib import Path

import pytest

from core.train import train
from scripts.train import build, full_config

ROOT = Path(__file__).resolve().parents[2]


def tiny_config():
    cfg = json.loads((ROOT / "configs" / "smoke.json").read_text())
    cfg["loop"].update(total_steps=300, epsilon_decay_steps=200, print_every_episodes=1000)
    cfg["learner"].update(learning_starts=100, batch_size=16)
    return cfg


def run(tmp_path):
    tmp_path.mkdir()
    env, learner, loop = build(tiny_config())
    rows = train(env, learner, loop, tmp_path, ("lines", "score", "pieces"), tensorboard=False)
    timing = ("steps_per_sec", "elapsed_sec")
    return [{k: v for k, v in r.items() if k not in timing} for r in rows]


def test_training_is_reproducible(tmp_path):
    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert a == b
    assert a[-1]["loss"] != ""  # learning actually happened


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
