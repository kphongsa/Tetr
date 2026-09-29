import numpy as np
import pytest
import torch

from agents.afterstate_value_agent import AfterstateTDLearner, AfterstateValueAgent, TDConfig, make_value_net
from games.tetris.env import TetrisEnv
from games.tetris.features import FEATURE_NAMES, candidates

D = len(FEATURE_NAMES)


def make_learner(**overrides):
    cfg = TDConfig(hidden=(8,), buffer_capacity=100, batch_size=4, learning_starts=4, **overrides)
    net = make_value_net(D, cfg.hidden, seed=0)
    agent = AfterstateValueAgent(net, candidates, np.random.default_rng(0), gamma=cfg.gamma,
                                 reward_scale=cfg.reward_scale)
    return AfterstateTDLearner(agent, cfg, n_features=D, max_candidates=40)


def test_td_target_terminated_vs_bootstrap_and_padding():
    lr = make_learner(gamma=0.9, reward_scale=0.01, survival_bonus=0.1, game_over_value=-2.0)
    n = 40
    next_x = np.zeros((2, n, D), np.float32)
    next_x[1, 0], next_x[1, 1] = [0.1, 0, 0, 0], [0.5, 0.25, 0, 0.3]
    next_r = np.zeros((2, n), np.float32)
    next_r[1, 1] = 100.0
    next_r[1, 5] = 1e6  # a padding row with a huge reward: must be ignored
    mask = np.zeros((2, n), bool)
    mask[1, :2] = True
    batch = {"x": np.zeros((2, D), np.float32), "terminated": np.array([True, False]),
             "next_x": next_x, "next_r": next_r, "next_mask": mask}

    target = lr.td_targets(batch)
    with torch.no_grad():
        v = lr.target_net(torch.from_numpy(next_x[1, :2])).numpy()
    expected = max(0.01 * 0 + 0.1 + 0.9 * v[0], 0.01 * 100 + 0.1 + 0.9 * v[1])
    assert target[0].item() == -2.0  # game over: no bootstrapping
    assert target[1].item() == pytest.approx(expected, rel=1e-6)


def play_steps(lr, steps, seed=0, max_pieces=None):
    env = TetrisEnv(max_pieces=max_pieces)
    obs, info = env.reset(seed=seed)
    for _ in range(steps):
        a = lr.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        lr.observe(a, r, term, trunc, obs, info)
        if term or trunc:
            return term, trunc
    return False, False


def test_observe_stores_next_candidates_and_terminal():
    lr = make_learner()
    lr.epsilon = 1.0  # random play dies fast
    term, _ = play_steps(lr, 500)
    assert term
    last = lr.buffer.size - 1
    assert lr.buffer.data["terminated"][last] and not lr.buffer.data["next_mask"][last].any()
    assert lr.buffer.data["next_mask"][0].sum() >= 9  # first move: next piece has 9+ placements


def test_truncated_transition_still_bootstraps():
    lr = make_learner()
    term, trunc = play_steps(lr, 10, max_pieces=3)
    assert trunc and not term
    last = lr.buffer.size - 1
    assert not lr.buffer.data["terminated"][last] and lr.buffer.data["next_mask"][last].any()


def test_reward_mismatch_is_caught():
    lr = make_learner()
    obs, info = TetrisEnv().reset(seed=0)
    a = lr.act(obs, info)
    with pytest.raises(AssertionError):
        lr.observe(a, 999, False, False, obs, info)


def full_buffer_loss(lr):
    batch = {k: v[: lr.buffer.size] for k, v in lr.buffer.data.items()}
    with torch.no_grad():
        pred = lr.net(torch.from_numpy(batch["x"]))
        return torch.nn.functional.huber_loss(pred, lr.td_targets(batch)).item()


def test_updates_lower_the_loss_on_fixed_data_and_target_is_frozen():
    # 15 random pieces and no game over. (A game-over transition would add an
    # error that CAN'T be learned away: whether the next piece fails to spawn
    # is random, so a -2 target sits among lookalike afterstates worth ~0.)
    lr = make_learner(lr=0.01, target_update_every=10_000)
    lr.epsilon = 1.0
    assert play_steps(lr, 15) == (False, False)
    assert lr.update() is not None
    before = [p.clone() for p in lr.target_net.parameters()]
    loss_before = full_buffer_loss(lr)
    for _ in range(300):
        lr.update()
    # Measured on ALL stored transitions (single sampled batches are too noisy
    # to compare). The target network is frozen, so this is plain regression.
    assert full_buffer_loss(lr) < 0.5 * loss_before
    # online network changed, target network didn't (no sync yet)...
    assert all(torch.equal(a, b) for a, b in zip(before, lr.target_net.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(lr.net.parameters(), lr.target_net.parameters()))
    lr.sync_target()  # ...until we sync
    assert all(torch.equal(a, b) for a, b in zip(lr.net.parameters(), lr.target_net.parameters()))


def test_no_update_while_buffer_fills():
    lr = make_learner()
    assert lr.update() is None
