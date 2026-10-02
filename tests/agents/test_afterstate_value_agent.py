import numpy as np
import torch

from agents.afterstate_value_agent import AfterstateValueAgent, ValueMLP, make_value_net
from core.evaluate import evaluate
from games.tetris.env import TetrisEnv
from games.tetris.features import FEATURE_NAMES, candidates

N = len(FEATURE_NAMES)


def make_agent(epsilon=0.0, seed=0, rng_seed=0):
    net = make_value_net(N, hidden=(16, 16), seed=seed)
    return AfterstateValueAgent(net, candidates, np.random.default_rng(rng_seed), epsilon=epsilon)


def play(agent, seed, max_pieces=60):
    env = TetrisEnv(max_pieces=max_pieces)
    obs, info = env.reset(seed=seed)
    actions, done = [], False
    while not done:
        a = agent.act(obs, info)
        assert info["action_mask"][a]  # always legal
        actions.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated
    return actions


def test_network_output_shape():
    net = ValueMLP(N, hidden=(8,))
    assert net(torch.zeros(5, N)).shape == (5,)
    assert net(torch.zeros(1, N)).shape == (1,)


def test_same_seed_same_initial_weights():
    a, b, c = (make_value_net(N, (16,), seed=s) for s in (1, 1, 2))
    for pa, pb, pc in zip(a.parameters(), b.parameters(), c.parameters()):
        assert torch.equal(pa, pb)
    assert not all(torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_q_values_are_scaled_reward_plus_discounted_value():
    agent = make_agent()
    agent.gamma, agent.reward_scale = 0.9, 0.01
    obs, info = TetrisEnv().reset(seed=0)
    c = candidates(obs, info)
    expected = 0.01 * c.rewards + 0.9 * agent.values(c.features)
    assert np.allclose(agent.q_values(c), expected)
    assert agent.values(c.features).shape == (len(c.actions),)


def test_greedy_is_deterministic():
    # epsilon = 0: same net + same game = same moves, whatever the agent rng.
    assert play(make_agent(rng_seed=1), seed=4) == play(make_agent(rng_seed=2), seed=4)


def test_greedy_picks_highest_q():
    agent = make_agent()
    obs, info = TetrisEnv().reset(seed=0)
    idx, c = agent.choose(obs, info)
    assert idx == int(np.argmax(agent.q_values(c)))


def test_full_exploration_depends_on_agent_rng():
    # epsilon = 1: every move random (but legal); reproducible from the agent rng.
    a1 = play(make_agent(epsilon=1.0, rng_seed=1), seed=4)
    a2 = play(make_agent(epsilon=1.0, rng_seed=1), seed=4)
    a3 = play(make_agent(epsilon=1.0, rng_seed=2), seed=4)
    assert a1 == a2 and a1 != a3


def test_works_with_evaluation_harness():
    agent = make_agent()
    res = evaluate(TetrisEnv(), lambda seed: agent, seeds=[1, 2, 3], max_steps=50,
                   info_keys=("lines", "score", "pieces"))
    s = res["summary"]
    assert s["games"] == 3 and s["steps"]["max"] <= 50


def test_ranked_choices_first_is_what_it_plays():
    agent = make_agent()
    env = TetrisEnv()
    obs, info = env.reset(seed=4)
    for _ in range(20):
        ranked = agent.ranked_choices(obs, info, 5)
        assert len(ranked) == min(5, len(info["legal_actions"]))
        assert ranked[0][0] == agent.act(obs, info)
        assert [q for _, q in ranked] == sorted((q for _, q in ranked), reverse=True)
        obs, _, terminated, _, info = env.step(ranked[0][0])
        if terminated:
            break
