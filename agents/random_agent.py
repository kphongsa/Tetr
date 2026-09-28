"""Picks uniformly at random among the currently legal actions.

Useful as a sanity check (does the env survive thousands of games?) and as
the absolute floor any real agent must beat.

It has its OWN rng, separate from the game's. The game's randomness (piece
sequence) must depend only on the game seed; if the agent shared that rng,
changing the agent would change which pieces appear, and replays of
"seed + actions" would break.

Game-agnostic on purpose: it only reads info["legal_actions"], so it will
work unchanged on a Slay the Spire env that follows the same interface.
"""

from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, obs, info) -> int:
        legal = info["legal_actions"]
        return legal[int(self.rng.integers(len(legal)))]
