"""Learned player: a small neural network rates every afterstate, pick the best.

Same recipe as the heuristic agent (step 2), with the hand-set weighted sum
replaced by a neural network whose parameters are LEARNED (step 3b):

    for each legal action a:
        r(a)  = immediate reward of a (points it scores right now)
        V(a)  = network's estimate of all FUTURE reward after a's afterstate
        Q(a)  = reward_scale * r(a) + gamma * V(a)
    play argmax Q, except with probability epsilon play a random legal action

Why split the immediate reward out instead of letting V predict it too:
the game tells us r(a) exactly, so there's no reason to make the network
guess it. It also matters for step 3e, where the network will see the raw
board: after a line clear the cleared rows are gone, so the network couldn't
even see that points were just scored.

Game-agnostic on purpose. The agent never imports a game: it's handed a
`candidates_fn(obs, info)` that returns, for every legal action, its id,
a feature vector (network input) and its immediate reward. For Tetris
that's games.tetris.features.candidates; for Slay the Spire it would be a
different function and the same agent.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
from torch import nn


class CandidateSet(Protocol):
    """What candidates_fn must return (a structural type: any object with these fields)."""

    actions: list[int]
    features: np.ndarray  # float32 (n, n_features)
    rewards: np.ndarray  # (n,) raw immediate reward, same units as env.step's reward


class ValueMLP(nn.Module):
    """Multi-layer perceptron: feature vector in, one number (the value) out.

    Layers: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear(1).
    A Linear layer is a weighted sum (like the heuristic's) but with many
    outputs; ReLU (max(0, x)) between them lets the network represent
    curved, non-additive judgments such as "holes matter more when the stack
    is high", which a single weighted sum can't.
    """

    def __init__(self, n_inputs: int, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_inputs) -> (batch,): one value per row."""
        return self.net(x).squeeze(-1)


def make_value_net(n_inputs: int, hidden: Sequence[int], seed: int) -> ValueMLP:
    """Build a ValueMLP whose random starting weights depend only on `seed`.

    PyTorch initializes layers from its GLOBAL random generator. fork_rng()
    saves that generator's state and restores it afterwards, so seeding it
    here gives reproducible weights without leaking a side effect onto
    anything else that uses torch randomness (CLAUDE.md rule 3 in spirit).
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return ValueMLP(n_inputs, hidden)


class AfterstateValueAgent:
    def __init__(
        self,
        net: nn.Module,
        candidates_fn: Callable[[Any, dict], CandidateSet],
        rng: np.random.Generator,
        epsilon: float = 0.0,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
    ) -> None:
        self.net = net
        self.candidates_fn = candidates_fn
        self.rng = rng  # the agent's OWN rng (for exploration), never the game's
        self.epsilon = epsilon
        self.gamma = gamma
        self.reward_scale = reward_scale

    @torch.no_grad()  # just playing, not learning: skip the bookkeeping gradients need
    def values(self, features: np.ndarray) -> np.ndarray:
        """Network value of each row of `features`, as a NumPy array."""
        return self.net(torch.from_numpy(features)).numpy()

    def q_values(self, cands: CandidateSet) -> np.ndarray:
        """Q(a) = scaled immediate reward + gamma * value of the afterstate, per candidate."""
        return self.reward_scale * cands.rewards + self.gamma * self.values(cands.features)

    def choose(self, obs, info) -> tuple[int, CandidateSet]:
        """Index of the chosen candidate, plus the candidates themselves.

        The training loop (step 3b) needs both: the chosen afterstate's
        features are what gets a TD update. Returning them avoids computing
        every afterstate twice.
        """
        cands = self.candidates_fn(obs, info)
        # Epsilon-greedy: explore with probability epsilon. We draw the coin
        # flip even when epsilon is 0, so the rng advances identically
        # regardless of epsilon (keeps runs comparable and easy to reason about).
        explore = self.rng.random() < self.epsilon
        if explore:
            return int(self.rng.integers(len(cands.actions))), cands
        # np.argmax returns the FIRST maximum; candidates are sorted by action
        # id, so exact ties go to the lowest id, same rule as the heuristic.
        return int(np.argmax(self.q_values(cands))), cands

    def act(self, obs, info) -> int:
        idx, cands = self.choose(obs, info)
        return cands.actions[idx]
