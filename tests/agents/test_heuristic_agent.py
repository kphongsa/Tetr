import numpy as np
import pytest

from agents.heuristic_agent import LEE_WEIGHTS, HeuristicAgent
from agents.random_agent import RandomAgent
from games.tetris.board import HEIGHT, WIDTH, Board
from games.tetris.engine import TetrisEngine
from games.tetris.env import TetrisEnv
from games.tetris.pieces import I, O
from games.tetris.placements import decode_action, legal_placements


def obs_info(piece, rows=()):
    """Build (obs, info) like the env would, for a hand-drawn board and a chosen piece."""
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for i, line in enumerate(rows):
        grid[HEIGHT - len(rows) + i] = [ch == "#" for ch in line]
    e = TetrisEngine(np.random.default_rng(0), board=Board(grid), spawn_first=False)
    e.spawn(piece)
    return {"board": grid, "piece": piece}, {"placements": legal_placements(e)}


def test_takes_the_obvious_double_line_clear():
    obs, info = obs_info(I, ["####.#####", "####.#####"])
    rotation, column = decode_action(HeuristicAgent().act(obs, info))
    assert column == 4 and rotation % 2 == 1  # vertical I into the gap


def test_ties_go_to_lowest_action_id():
    # Empty board, O piece: column 0 and column 8 give mirror-image boards with
    # identical scores. The tie must go to the lower action id (column 0).
    obs, info = obs_info(O)
    agent = HeuristicAgent()
    scores = dict(agent.score_placements(obs, info))
    assert scores[0] == scores[8]
    assert agent.act(obs, info) == 0


def test_weights_must_match_feature_count():
    with pytest.raises(ValueError):
        HeuristicAgent(weights=(1.0, 2.0))
    assert len(LEE_WEIGHTS) == 4


def play(agent, seed, max_pieces):
    env = TetrisEnv(max_pieces=max_pieces)
    obs, info = env.reset(seed=seed)
    actions, done = [], False
    while not done:
        a = agent.act(obs, info)
        actions.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated
    return actions, info


def test_is_deterministic():
    a1, _ = play(HeuristicAgent(), seed=5, max_pieces=100)
    a2, _ = play(HeuristicAgent(), seed=5, max_pieces=100)
    assert a1 == a2


def test_crushes_random_on_same_seed():
    _, h = play(HeuristicAgent(), seed=0, max_pieces=300)
    _, r = play(RandomAgent(np.random.default_rng([0, 1])), seed=0, max_pieces=300)
    # 300 pieces = 1200 cells = at most 120 lines. A decent player clears most
    # of that; random almost never clears a single line.
    assert h["lines"] >= 80
    assert h["lines"] > r["lines"] + 50
