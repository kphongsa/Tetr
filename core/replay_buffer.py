"""Replay buffer: a fixed-size memory of past transitions, sampled at random for training.

Why not just learn from each move as it happens?
  1. Consecutive moves are nearly identical (same board plus one piece).
     A network trained on them in order overfits to "the last few seconds"
     and forgets the rest, which makes learning wobble or collapse.
     Random samples from a big memory mix early, mid and late game.
  2. Reuse: each transition is sampled many times, so fewer games are needed.

Game-agnostic: the caller declares the fields ("spec") once, e.g.

    ReplayBuffer(100_000, {"x": ((4,), np.float32), "done": ((), np.bool_)}, rng)

and then adds/samples dicts of arrays. It knows nothing about Tetris.

Storage is one pre-allocated NumPy array per field, used as a ring: once
full, the newest transition overwrites the oldest. Pre-allocating means the
memory cost is paid (and visible) up front, not as a slow creep.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Spec = dict[str, tuple[tuple[int, ...], Any]]  # field name -> (shape of ONE item, dtype)


class ReplayBuffer:
    def __init__(self, capacity: int, spec: Spec, rng: np.random.Generator) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = rng
        self.data = {name: np.zeros((capacity, *shape), dtype=dtype) for name, (shape, dtype) in spec.items()}
        self.size = 0  # how many slots hold real data
        self.pos = 0  # where the next add() writes

    def __len__(self) -> int:
        return self.size

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.data.values())

    def add(self, **fields: Any) -> None:
        if fields.keys() != self.data.keys():
            raise KeyError(f"expected fields {sorted(self.data)}, got {sorted(fields)}")
        for name, value in fields.items():
            self.data[name][self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """`batch_size` random transitions (with replacement), one array per field.

        With replacement = the same transition can appear twice in a batch.
        With a buffer of thousands and batches of ~100 that's rare and
        harmless, and it's simpler and faster than sampling without.
        """
        if self.size == 0:
            raise ValueError("buffer is empty")
        idx = self.rng.integers(self.size, size=batch_size)
        return {name: arr[idx] for name, arr in self.data.items()}
