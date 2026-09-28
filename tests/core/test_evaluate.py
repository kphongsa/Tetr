import numpy as np
import pytest

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from core.evaluate import StepLimit, evaluate, summarize
from games.tetris.env import TetrisEnv

METRICS = ("lines", "score", "pieces")


class CountdownEnv:
    """Tiny fake game (no Tetris) to test the harness in isolation.

    The game lasts `seed` steps, then terminates. Reward 1 per step.
    """

    def reset(self, seed=None):
        self.left = seed
        return 0, {"left": self.left}

    def step(self, action):
        self.left -= 1
        return 0, 1.0, self.left == 0, False, {"left": self.left}


class NullAgent:
    def act(self, obs, info):
        return 0


def test_harness_works_on_a_non_tetris_env():
    r = evaluate(CountdownEnv(), lambda s: NullAgent(), seeds=[3, 5, 10])
    assert [g["steps"] for g in r["games"]] == [3, 5, 10]
    assert r["summary"]["return"]["mean"] == pytest.approx(6.0)
    assert r["summary"]["capped"] == 0


def test_step_limit_truncates_at_cap():
    r = evaluate(CountdownEnv(), lambda s: NullAgent(), seeds=[3, 5, 10], max_steps=4)
    assert [g["steps"] for g in r["games"]] == [3, 4, 4]
    assert [g["truncated"] for g in r["games"]] == [False, True, True]
    assert r["summary"]["capped"] == 2


def test_real_game_over_on_the_cap_step_is_terminated_not_truncated():
    # The game ends by itself on step 4, exactly when the cap is reached.
    env = StepLimit(CountdownEnv(), max_steps=4)
    env.reset(seed=4)
    for _ in range(3):
        env.step(0)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated


def test_step_limit_resets_its_counter():
    env = StepLimit(CountdownEnv(), max_steps=2)
    for _ in range(2):  # two episodes in a row: the second must also get 2 steps
        env.reset(seed=10)
        assert env.step(0)[3] is False
        assert env.step(0)[3] is True


def test_cap_truncates_a_tetris_game():
    # The heuristic easily survives 25 pieces, so every game must be cut at exactly 25.
    agent = HeuristicAgent()
    r = evaluate(TetrisEnv(), lambda s: agent, seeds=[0, 1, 2], max_steps=25, info_keys=METRICS)
    for g in r["games"]:
        assert g["pieces"] == 25 and g["steps"] == 25
        assert g["truncated"] and not g["terminated"]
    assert r["summary"]["capped"] == 3


def strip_timing(result):
    s = {k: v for k, v in result["summary"].items()
         if k not in ("seconds", "games_per_second", "steps_per_second")}
    return result["games"], s


@pytest.mark.parametrize("name", ["random", "heuristic"])
def test_same_agent_same_seeds_same_results(name):
    def make(seed):
        if name == "random":
            return RandomAgent(np.random.default_rng([seed, 1]))
        return HeuristicAgent()

    seeds = [7, 8, 9, 10]
    r1 = evaluate(TetrisEnv(), make, seeds, max_steps=60, info_keys=METRICS)
    r2 = evaluate(TetrisEnv(), make, seeds, max_steps=60, info_keys=METRICS)
    assert strip_timing(r1) == strip_timing(r2)


def test_game_result_doesnt_depend_on_game_order():
    # Per-game agent rngs mean seed 9's game is the same whether it's played first or last.
    make = lambda seed: RandomAgent(np.random.default_rng([seed, 1]))  # noqa: E731
    a = evaluate(TetrisEnv(), make, [7, 8, 9], info_keys=METRICS)["games"]
    b = evaluate(TetrisEnv(), make, [9, 8, 7], info_keys=METRICS)["games"]
    assert a == b[::-1]


def test_summarize():
    s = summarize([1, 2, 3, 10])
    assert s == {"mean": 4.0, "median": 2.5, "std": pytest.approx(np.std([1, 2, 3, 10])),
                 "min": 1.0, "max": 10.0}
