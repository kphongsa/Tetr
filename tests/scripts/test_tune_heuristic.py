import numpy as np

from agents.heuristic_agent import LEE_WEIGHTS
from scripts.evaluate import EVAL_FIRST_SEED
from scripts.tune_heuristic import make_fitness, round_seeds


def test_tuning_seeds_never_touch_evaluation_seeds():
    # Even a very long tuning run (100 rounds x 50 games) stays below the eval seeds.
    for r in range(100):
        assert max(round_seeds(r, 50)) < EVAL_FIRST_SEED


def test_rounds_use_fresh_seeds():
    assert set(round_seeds(0, 3)).isdisjoint(round_seeds(1, 3))


def test_every_candidate_in_a_round_plays_the_same_games():
    # Identical weights in the same round must get identical fitness: they
    # face the same piece sequences, so any difference would mean unfair seeds.
    fitness = make_fitness(games=2, max_pieces=40)
    w = np.array(LEE_WEIGHTS)
    scores = fitness(np.vstack([w, w, w]), round_idx=0)
    assert scores[0] == scores[1] == scores[2]


def test_fitness_prefers_sensible_weights_over_reversed_ones():
    fitness = make_fitness(games=2, max_pieces=60)
    w = np.array(LEE_WEIGHTS)
    good, bad = fitness(np.vstack([w, -w]), round_idx=0)
    assert good > bad
