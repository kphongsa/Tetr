"""Cross-entropy method (CEM): a simple search for a good parameter vector.

The idea in plain language
--------------------------
Keep a "cloud" of guesses: a normal (bell-curve) distribution over parameter
vectors, described by a mean (the centre of the cloud) and a standard
deviation per dimension (how wide it is). Each round:

  1. sample   Draw `population` vectors from the cloud.
  2. score    Run each one (for us: play a few games) to get a fitness number.
  3. select   Keep the top `elite_frac` of them: the "elite".
  4. refit    Move the cloud's centre to the elite's average, and set its
              width to the elite's spread. The cloud drifts toward good
              regions and shrinks as the elite agree with each other.
  5. widen    Add a little extra width, so the cloud doesn't shrink onto a
              lucky-but-mediocre spot too early. The extra shrinks to 0 by the
              last round, so the search can settle at the end.

Why this is "search", not deep RL: CEM never looks inside a game. It doesn't
know which move was good or bad, only each candidate's final total. It also
has no gradient (no "which direction improves the score" computed from the
model); it just tries, keeps what worked, and tries nearby. That works well
for a handful of parameters (4 weights here) and badly for millions
(a neural network), which is why step 3 needs real RL.

Game-agnostic: the caller supplies `fitness(candidates, round) -> scores`.
Scoring the whole batch at once lets the caller give every candidate in a
round the SAME games (a fair comparison), and later spread the batch over
several CPU cores without changing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CEMConfig:
    rounds: int = 15
    population: int = 30
    elite_frac: float = 0.2
    init_std: float = 1.0
    # Extra spread added to the refitted std. Starts here and shrinks linearly
    # to 0 at the final round (Szita & Lorincz 2006 found decreasing noise
    # works best for Tetris).
    extra_std: float = 0.2
    # Scale-free parameters: for an "argmax of weights . features" player,
    # doubling all weights picks exactly the same moves. Projecting every
    # sample onto the unit sphere (length 1) removes that useless direction,
    # so the search only explores directions that change behaviour.
    normalize: bool = True


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def cem(
    fitness: Callable[[np.ndarray, int], np.ndarray],
    dim: int,
    rng: np.random.Generator,
    config: CEMConfig = CEMConfig(),
    init_mean: np.ndarray | None = None,
    on_round: Callable[[dict], None] | None = None,
) -> dict:
    """Run CEM. Returns {"mean": final centre, "std": final width, "history": [per-round dicts]}.

    fitness(candidates, round_idx): candidates is (population, dim); must
    return one score per row, higher = better.
    on_round(record): called after each round (for logging / saving progress).
    """
    n_elite = max(1, int(round(config.population * config.elite_frac)))
    mean = np.zeros(dim) if init_mean is None else np.asarray(init_mean, dtype=np.float64)
    std = np.full(dim, config.init_std)
    history = []

    for r in range(config.rounds):
        candidates = mean + std * rng.standard_normal((config.population, dim))
        if config.normalize:
            candidates = unit(candidates)
        scores = np.asarray(fitness(candidates, r), dtype=np.float64)

        # Stable sort on -score: ties keep sampling order, so runs are reproducible.
        order = np.argsort(-scores, kind="stable")
        elite = candidates[order[:n_elite]]

        extra = config.extra_std * (1 - (r + 1) / config.rounds)
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + extra

        record = {
            "round": r,
            "best_score": float(scores[order[0]]),
            "mean_score": float(scores.mean()),
            "elite_mean_score": float(scores[order[:n_elite]].mean()),
            "best_candidate": candidates[order[0]].tolist(),
            "mean": (unit(mean) if config.normalize else mean).tolist(),
            "std": std.tolist(),
        }
        history.append(record)
        if on_round is not None:
            on_round(record)

    final = unit(mean) if config.normalize else mean
    return {"mean": final, "std": std, "history": history}
