import numpy as np

from core.cem import CEMConfig, cem, unit


def sphere_fitness(target):
    """Toy problem: score = -(distance to a hidden target)^2. Best possible = 0 at target."""
    def fitness(candidates, round_idx):
        return -((candidates - target) ** 2).sum(axis=1)
    return fitness


def test_finds_a_hidden_target():
    target = np.array([3.0, -2.0, 0.5])
    cfg = CEMConfig(rounds=30, population=50, init_std=5.0, extra_std=0.5, normalize=False)
    result = cem(sphere_fitness(target), 3, np.random.default_rng(0), cfg)
    assert np.allclose(result["mean"], target, atol=0.1)


def test_finds_a_direction_when_normalized():
    # With normalize=True only direction matters: score = cosine with a hidden direction.
    target = unit(np.array([-1.0, 0.5, -2.0, -0.3]))
    cfg = CEMConfig(rounds=25, population=40, normalize=True)
    result = cem(lambda c, r: c @ target, 4, np.random.default_rng(1), cfg)
    assert np.isclose(np.linalg.norm(result["mean"]), 1.0)
    assert result["mean"] @ target > 0.99


def test_candidates_are_unit_length_when_normalized():
    seen = []

    def fitness(c, r):
        seen.append(c)
        return np.zeros(len(c))

    cem(fitness, 4, np.random.default_rng(0), CEMConfig(rounds=2, population=5))
    assert np.allclose(np.linalg.norm(np.vstack(seen), axis=1), 1.0)


def test_same_seed_same_run():
    f = sphere_fitness(np.array([1.0, 2.0]))
    cfg = CEMConfig(rounds=5, population=10, normalize=False)
    a = cem(f, 2, np.random.default_rng(42), cfg)
    b = cem(f, 2, np.random.default_rng(42), cfg)
    assert a["history"] == b["history"]


def test_history_and_extra_spread_decay():
    cfg = CEMConfig(rounds=4, population=10, extra_std=0.3, normalize=False)
    # Constant fitness: every candidate ties. std must still include the extra
    # spread, which reaches 0 on the final round.
    result = cem(lambda c, r: np.zeros(len(c)), 2, np.random.default_rng(0), cfg)
    assert [h["round"] for h in result["history"]] == [0, 1, 2, 3]
    assert result["history"][-1]["std"] == result["std"].tolist()
    first_std = np.array(result["history"][0]["std"])
    assert (first_std >= 0.3 * (1 - 1 / 4) - 1e-12).all()
