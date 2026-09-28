import numpy as np
import pytest

from agents.random_agent import RandomAgent
from games.tetris.env import TetrisEnv


def play_actions(seed, actions):
    """Replay: fresh env, given seed, given actions. Returns every step's output."""
    env = TetrisEnv()
    obs, info = env.reset(seed=seed)
    history = [(obs["board"].copy(), obs["piece"], info["legal_actions"])]
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        history.append((obs["board"].copy(), obs["piece"], info["legal_actions"],
                        reward, terminated, info["score"], info["lines"]))
    return history


def random_game(seed):
    """Play a whole game with the random agent; return the actions taken."""
    env = TetrisEnv()
    agent = RandomAgent(np.random.default_rng([seed, 1]))
    obs, info = env.reset(seed=seed)
    actions, terminated = [], False
    while not terminated:
        a = agent.act(obs, info)
        actions.append(a)
        obs, _, terminated, _, info = env.step(a)
    return actions, info


def histories_equal(h1, h2):
    if len(h1) != len(h2):
        return False
    for s1, s2 in zip(h1, h2):
        if not np.array_equal(s1[0], s2[0]) or s1[1:] != s2[1:]:
            return False
    return True


# --- interface ---------------------------------------------------------------

def test_reset_returns_obs_and_info():
    env = TetrisEnv()
    obs, info = env.reset(seed=0)
    assert obs["board"].shape == (22, 10) and obs["board"].dtype == np.uint8
    assert obs["board"].sum() == 0
    assert 1 <= obs["piece"] <= 7
    assert info["action_mask"].shape == (40,) and info["action_mask"].dtype == bool
    assert info["seed"] == 0 and info["score"] == 0


def test_mask_matches_legal_actions_and_placements():
    env = TetrisEnv()
    _, info = env.reset(seed=3)
    assert list(np.flatnonzero(info["action_mask"])) == info["legal_actions"]
    assert [p.action for p in info["placements"]] == info["legal_actions"]


def test_step_places_one_piece():
    env = TetrisEnv()
    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(info["legal_actions"][0])
    assert obs["board"].sum() == 4
    assert reward == 0 and not terminated and not truncated
    assert info["pieces"] == 1


def test_illegal_action_raises():
    env = TetrisEnv()
    _, info = env.reset(seed=0)
    illegal = int(np.flatnonzero(~info["action_mask"])[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_step_before_reset_and_after_game_over_raise():
    env = TetrisEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    actions, info = random_game(seed=0)
    env.reset(seed=0)
    for a in actions:
        _, _, terminated, _, info = env.step(a)
    assert terminated and info["legal_actions"] == [] and not info["action_mask"].any()
    with pytest.raises(RuntimeError):
        env.step(actions[0])


def test_reward_sums_to_score():
    env = TetrisEnv()
    agent = RandomAgent(np.random.default_rng(0))
    for seed in range(30):
        obs, info = env.reset(seed=seed)
        done = False
        game_total = 0
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, info))
            game_total += r
            done = term or trunc
        assert game_total == info["score"]


@pytest.mark.parametrize("seed", range(7))  # different seeds -> different first pieces
def test_line_clear_gives_reward(seed):
    # Random play almost never clears a line, so set one up: take any legal
    # placement on the empty floor, then fill the rest of the bottom row
    # around it. That placement stays legal and now completes the row.
    env = TetrisEnv()
    _, info = env.reset(seed=seed)
    p = info["placements"][0]
    bottom = env.engine.board.grid[21]
    bottom[:] = 1
    for r, c in p.cells:
        if r == 21:
            bottom[c] = 0
    obs, reward, _, _, info = env.step(p.action)
    assert reward == 100 and info["lines_cleared"] == 1 and info["score"] == 100


def test_truncation_at_max_pieces():
    env = TetrisEnv(max_pieces=3)
    obs, info = env.reset(seed=0)
    for i in range(3):
        obs, _, terminated, truncated, info = env.step(info["legal_actions"][0])
    assert truncated and not terminated


# --- reproducibility ---------------------------------------------------------

def test_same_seed_same_actions_identical_game():
    actions, _ = random_game(seed=11)
    h1 = play_actions(11, actions)
    h2 = play_actions(11, actions)
    assert histories_equal(h1, h2)
    assert h1[-1][4] is True  # the replay also ends in game over


def test_different_seed_gives_different_pieces():
    env = TetrisEnv()
    seq = {}
    for seed in (1, 2):
        env.reset(seed=seed)
        pieces = []
        for _ in range(10):
            pieces.append(env.engine.piece)
            env.engine.hard_drop()
        seq[seed] = pieces
    assert seq[1] != seq[2]


def test_unseeded_reset_reports_a_seed_that_replays_the_game():
    env = TetrisEnv()
    agent = RandomAgent(np.random.default_rng(5))
    obs, info = env.reset()  # no seed given
    seed = info["seed"]
    assert isinstance(seed, int)
    actions, done = [], False
    while not done:
        a = agent.act(obs, info)
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    final_score = info["score"]
    replay = play_actions(seed, actions)
    assert replay[-1][5] == final_score


# --- robustness --------------------------------------------------------------

def test_random_agent_plays_many_games_without_crashing():
    for seed in range(300):
        actions, info = random_game(seed)
        assert len(actions) == info["pieces"] > 0
