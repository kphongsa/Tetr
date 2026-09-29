import json
from pathlib import Path

import pytest
import torch

from core.train import train
from scripts.compare import paired
from scripts.train import build, load_eval_agent

ROOT = Path(__file__).resolve().parents[2]


def test_paired_differences():
    learned = [{"seed": 1, "score": 10}, {"seed": 2, "score": 5}, {"seed": 3, "score": 7}]
    base = [{"seed": 3, "score": 7}, {"seed": 1, "score": 4}, {"seed": 2, "score": 9}]  # any order
    p = paired(learned, base, "score")
    assert p["mean_diff"] == pytest.approx((6 - 4 + 0) / 3)
    assert (p["wins"], p["ties"], p["losses"], p["games"]) == (1, 1, 1, 3)


def test_full_config_is_valid():
    cfg = json.loads((ROOT / "configs" / "full.json").read_text())
    build(cfg)  # unknown keys / eval-seed overlap would raise


def test_load_eval_agent_restores_trained_weights(tmp_path):
    cfg = json.loads((ROOT / "configs" / "smoke.json").read_text())
    cfg["loop"].update(total_steps=200, print_every_episodes=1000, eval_every_episodes=None)
    cfg["learner"].update(learning_starts=50, batch_size=16)
    env, learner, loop = build(cfg)
    train(env, learner, loop, tmp_path, tensorboard=False, config=cfg)
    agent, ck = load_eval_agent(tmp_path / "checkpoints" / "latest.pt")
    assert agent.epsilon == 0.0
    for p, q in zip(agent.net.parameters(), learner.net.parameters()):
        assert torch.equal(p, q)
