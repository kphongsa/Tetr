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

import copy
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.replay_buffer import ReplayBuffer


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
        candidates_fn: Callable[[Any, dict], Any],  # returns a CandidateSet-like object
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

    def choose(self, obs, info, cands: CandidateSet | None = None) -> tuple[int, CandidateSet]:
        """Index of the chosen candidate, plus the candidates themselves.

        The learner needs both: the chosen afterstate's features are what
        gets a TD update. Returning them (and accepting precomputed ones)
        avoids computing every afterstate twice.
        """
        if cands is None:
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

    def ranked_choices(self, obs, info, k: int) -> list[tuple[int, float]]:
        """The k best placements as (action, Q), best first: "what was it thinking".

        Q is the number the greedy agent maximizes. A stable sort keeps
        exact ties in action-id order, so the first entry is always what
        choose() would pick with epsilon = 0 (which also takes the lowest id).
        Used by the step 4e viewer overlay; doesn't touch the rng.
        """
        cands = self.candidates_fn(obs, info)
        q = self.q_values(cands)
        order = np.argsort(-q, kind="stable")[:k]
        return [(int(cands.actions[i]), float(q[i])) for i in order]


# ----------------------------------------------------------------------
# Learning (step 3b)
# ----------------------------------------------------------------------
@dataclass
class TDConfig:
    """Hyperparameters of the network and its TD learning. All chosen, none learned."""

    hidden: tuple[int, ...] = (64, 64)  # hidden layer sizes of the MLP
    net_seed: int = 0  # starting weights
    rng_seed: int = 0  # rng for exploration and replay sampling
    gamma: float = 0.99  # discount factor
    lr: float = 1e-3  # learning rate (Adam)
    # Learning-rate decay: the rate falls in a straight line from `lr` to
    # `lr_end` over the first `lr_decay_steps` updates, then stays at lr_end.
    # None = constant rate. Why: once the agent is decent, big weight changes
    # mostly reshuffle which placement wins (policy churn); small late steps
    # let it settle on a good policy instead of wandering around it.
    lr_end: float | None = None
    lr_decay_steps: int = 0
    # Inputs are all 0/1 (e.g. a raw board): store them bit-packed in the
    # replay buffer, 8 inputs per byte. For 220-cell boards with 40 candidates
    # per transition that's ~1.2 KB instead of ~35 KB (float32), so 100k
    # transitions take ~120 MB of memory instead of ~3.5 GB.
    binary_inputs: bool = False
    batch_size: int = 128
    buffer_capacity: int = 100_000  # replay buffer size, in transitions
    learning_starts: int = 1_000  # don't update until the buffer holds this many
    target_update_every: int = 1_000  # copy online -> target network every N updates
    huber_delta: float = 1.0  # Huber loss: squared below this error, linear above
    # Reward shaping: the learner's view only. Logs and evaluation use the real score.
    reward_scale: float = 0.01  # 100/300/500/800 points -> 1/3/5/8
    survival_bonus: float = 0.1  # added to every placement's reward
    game_over_value: float = -2.0  # TD target for an afterstate that led to game over


class AfterstateTDLearner:
    """Teaches an AfterstateValueAgent's network from its own games: TD(0) + replay + target network.

    What gets stored per move (one "transition")
    --------------------------------------------
    After playing action a_t from state s_t we land in afterstate s'_t; then
    the next piece arrives, giving state s_{t+1}. We store:

        x            features of s'_t (the afterstate we chose)
        terminated   did the game end right after s'_t?
        next_x       features of EVERY legal afterstate from s_{t+1}  (padded)
        next_r       their immediate raw rewards                      (padded)
        next_mask    which of the padded rows are real

    TD target for V(s'_t)
    ---------------------
        terminated:  game_over_value
        otherwise:   max over a' of [ scale*r(a') + bonus + gamma * V_target(s'_{t+1}(a')) ]

    "Otherwise" includes TRUNCATED (training cap reached): the game could
    have continued, so we bootstrap from the next position exactly as usual.
    Only a real game over has no future. So `truncated` isn't even needed
    for the target; what matters is whether the game truly ended.

    Why store every next candidate instead of recomputing later? The max has
    to use the CURRENT target network each time a transition is replayed,
    and recomputing afterstates at sample time would be the slowest part of
    training. 40 x 4 floats per transition is small (~70 MB for 100k).
    """

    # value_max: the largest prediction in the batch. Its per-episode average
    # is what scripts/run_status.py compares against the theoretical ceiling
    # (runaway values are a classic sign of TD learning going unstable).
    stat_names: tuple[str, ...] = ("loss", "value_mean", "value_max", "target_mean", "lr")

    def __init__(self, agent: AfterstateValueAgent, cfg: TDConfig, n_features: int, max_candidates: int) -> None:
        self.agent = agent
        self.cfg = cfg
        self.net = agent.net
        # The target network: a frozen copy used ONLY to compute TD targets.
        self.target_net = copy.deepcopy(self.net)
        self.target_net.requires_grad_(False)
        # Adam: the optimizer. It turns gradients into weight changes, with a
        # per-weight step size that adapts to how noisy that weight's gradient is.
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.n_features = n_features
        # What one stored input looks like: float32 values, or packed bits.
        width, dtype = ((n_features + 7) // 8, np.uint8) if cfg.binary_inputs else (n_features, np.float32)
        self.buffer = ReplayBuffer(
            cfg.buffer_capacity,
            {
                "x": ((width,), dtype),
                "terminated": ((), np.bool_),
                "next_x": ((max_candidates, width), dtype),
                "next_r": ((max_candidates,), np.float32),
                "next_mask": ((max_candidates,), np.bool_),
            },
            agent.rng,  # one rng for the whole learner: exploration + sampling
        )
        self.max_candidates = max_candidates
        self.updates = 0
        self._pending: tuple[np.ndarray, float] | None = None  # (x, raw reward) of the move just chosen
        self._cached: tuple[dict, CandidateSet] | None = None  # (info, its candidates)

    # The loop sets epsilon on the learner; the agent is what actually uses it.
    @property
    def epsilon(self) -> float:
        return self.agent.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self.agent.epsilon = value

    # ------------------------------------------------------------------
    def act(self, obs, info) -> int:
        # observe() already computed the candidates for this exact info dict.
        # `is` (identity) guarantees they're never reused for a different state.
        cached = self._cached[1] if self._cached is not None and self._cached[0] is info else None
        idx, cands = self.agent.choose(obs, info, cached)
        self._pending = (cands.features[idx], float(cands.rewards[idx]))
        return cands.actions[idx]

    def observe(self, action, reward, terminated, truncated, next_obs, next_info) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called without a preceding act()")
        x, expected_reward = self._pending
        self._pending = None
        # Cheap, always-on consistency check: the reward the agent predicted
        # for its move must be exactly what the game paid. If afterstates ever
        # drift from real play, training stops here instead of quietly
        # learning from wrong data.
        if float(reward) != expected_reward:
            raise AssertionError(f"predicted reward {expected_reward} but env paid {reward}")

        n, d = self.max_candidates, x.shape[0]
        next_x = np.zeros((n, d), dtype=np.float32)
        next_r = np.zeros(n, dtype=np.float32)
        next_mask = np.zeros(n, dtype=bool)
        self._cached = None
        if not terminated:
            cands = self.agent.candidates_fn(next_obs, next_info)
            k = len(cands.actions)
            next_x[:k], next_r[:k], next_mask[:k] = cands.features, cands.rewards, True
            self._cached = (next_info, cands)
        self.buffer.add(x=self._pack(x), terminated=terminated, next_x=self._pack(next_x), next_r=next_r,
                        next_mask=next_mask)

    # Bit-packing (only when cfg.binary_inputs): np.packbits turns every 8
    # zeros/ones along the last axis into one byte; unpackbits reverses it
    # (count= drops the padding bits when n_features isn't a multiple of 8).
    def _pack(self, x: np.ndarray) -> np.ndarray:
        return np.packbits(x > 0.5, axis=-1) if self.cfg.binary_inputs else x

    def _unpack(self, x: np.ndarray) -> np.ndarray:
        if not self.cfg.binary_inputs:
            return x
        return np.unpackbits(x, axis=-1, count=self.n_features).astype(np.float32)

    # ------------------------------------------------------------------
    def td_targets(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        """TD target for each transition in the batch (see the class docstring)."""
        cfg = self.cfg
        next_x = torch.from_numpy(batch["next_x"])
        b, n = next_x.shape[:2]
        mask = torch.from_numpy(batch["next_mask"])
        with torch.no_grad():  # targets are fixed numbers to aim at, not something to learn through
            # Run the network only on real rows (next_x[mask] -> (k, d)); padding
            # is ~35% of rows and this forward pass is the costliest part of an update.
            v_next = torch.zeros(b, n)
            v_next[mask] = self.target_net(next_x[mask])
            q_next = cfg.reward_scale * torch.from_numpy(batch["next_r"]) + cfg.survival_bonus + cfg.gamma * v_next
            # Padding rows aren't real moves: make sure max() can never pick them.
            q_next = q_next.masked_fill(~mask, float("-inf"))
            best = q_next.max(dim=1).values
            # For terminated rows every entry is masked, so `best` is -inf there;
            # torch.where discards it and uses game_over_value instead.
            terminated = torch.from_numpy(batch["terminated"])
            return torch.where(terminated, torch.full_like(best, cfg.game_over_value), best)

    def current_lr(self) -> float:
        cfg = self.cfg
        if cfg.lr_end is None or cfg.lr_decay_steps <= 0:
            return cfg.lr
        frac = min(1.0, self.updates / cfg.lr_decay_steps)
        return cfg.lr + frac * (cfg.lr_end - cfg.lr)

    def update(self) -> dict[str, float] | None:
        """One gradient step on a random batch. None while the buffer is still filling."""
        cfg = self.cfg
        if len(self.buffer) < max(cfg.learning_starts, cfg.batch_size):
            return None
        batch = self.buffer.sample(cfg.batch_size)
        batch["x"], batch["next_x"] = self._unpack(batch["x"]), self._unpack(batch["next_x"])
        target = self.td_targets(batch)
        pred = self.net(torch.from_numpy(batch["x"]))
        loss = F.huber_loss(pred, target, delta=cfg.huber_delta)

        lr = self.current_lr()
        for group in self.optimizer.param_groups:  # Adam reads its step size from here every step
            group["lr"] = lr
        self.optimizer.zero_grad()  # gradients add up by default; clear the previous step's
        loss.backward()  # compute d(loss)/d(weight) for every weight
        self.optimizer.step()  # move every weight a little in the direction that lowers the loss

        self.updates += 1
        if self.updates % cfg.target_update_every == 0:
            self.sync_target()
        # .item() pulls a plain Python number out of a 1-element tensor.
        return {"loss": loss.item(), "value_mean": pred.mean().item(), "value_max": pred.max().item(),
                "target_mean": target.mean().item(), "lr": lr}

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())

    # ------------------------------------------------------------------
    # Checkpointing and evaluation (step 3c)
    # ------------------------------------------------------------------
    def eval_agent(self) -> AfterstateValueAgent:
        """A greedy (epsilon = 0) player that shares the CURRENT network.

        It gets its own rng so evaluating never advances the training rng:
        a run with evaluations every 50 episodes trains exactly like one
        with evaluations every 100.
        """
        return AfterstateValueAgent(self.net, self.agent.candidates_fn, np.random.default_rng(0),
                                    epsilon=0.0, gamma=self.agent.gamma, reward_scale=self.agent.reward_scale)

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue learning, EXCEPT the replay buffer.

        Why not the buffer: at 100k transitions it's ~70 MB, which would make
        every checkpoint slow and large. On resume the buffer starts empty
        and learning pauses for `learning_starts` steps while it refills with
        the current (already good) policy's moves. The cost: a resumed run
        is not bit-identical to an uninterrupted one, and there's a short
        blip right after resuming while the network trains on a small,
        recent-only memory.
        """
        return {
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            # Adam keeps running averages per weight; without them the first
            # steps after resuming would be badly sized.
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "rng": self.agent.rng.bit_generator.state,  # exploration + replay sampling
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
        self.agent.rng.bit_generator.state = state["rng"]
        self._pending = None
        self._cached = None
