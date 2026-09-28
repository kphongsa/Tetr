import numpy as np

from agents.random_agent import RandomAgent


def test_only_picks_legal_actions_and_uses_all_of_them():
    agent = RandomAgent(np.random.default_rng(0))
    info = {"legal_actions": [3, 17, 25]}
    picks = [agent.act(None, info) for _ in range(300)]
    assert set(picks) == {3, 17, 25}


def test_same_seed_same_choices():
    info = {"legal_actions": list(range(20))}
    a = RandomAgent(np.random.default_rng(9))
    b = RandomAgent(np.random.default_rng(9))
    assert [a.act(None, info) for _ in range(50)] == [b.act(None, info) for _ in range(50)]
