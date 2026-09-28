"""Evaluation harness: play an agent on a fixed list of seeds and summarize the results.

Game-agnostic on purpose (no imports from games/): it only uses the
Gymnasium-style interface every env in this project follows,

    obs, info = env.reset(seed=seed)
    obs, reward, terminated, truncated, info = env.step(action)

and agents that follow `action = agent.act(obs, info)`. So the same harness
will evaluate a Slay the Spire agent later without changes.

Why fixed seeds
---------------
A game's randomness (the piece sequence in Tetris) comes only from its seed.
Evaluating two agents on the SAME seeds means they face the exact same
games, so a difference in score is due to the agents, not to luck. It also
makes every evaluation reproducible: same agent + same seeds = same numbers.

Why a step cap (truncation)
---------------------------
A good agent can play for a very long time (strong Tetris bots survive
millions of pieces). Without a cap, one evaluation could run for hours. So
each game is cut off after `max_steps` steps and reported as *truncated*
(cut short, not lost). The catch: if most games hit the cap, every good
agent gets the same capped score and the evaluation can't tell them apart.
`summary["capped"]` counts capped games so you can notice that.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class Env(Protocol):
    """What the harness needs from an environment (a structural type: no base class needed)."""

    def reset(self, seed: int | None = None) -> tuple[Any, dict]: ...
    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]: ...


class Agent(Protocol):
    def act(self, obs: Any, info: dict) -> int: ...


class StepLimit:
    """Wrap any env so each episode ends (truncated=True) after `max_steps` steps.

    This is Gymnasium's "TimeLimit" wrapper idea. It lives in core/ rather
    than relying on an env option (TetrisEnv has max_pieces) because other
    games won't have that option, and the harness must cap every game the
    same way. For Tetris, one step = one piece, so max_steps = max pieces.

    If the game really ends on the same step the cap is reached, we report
    terminated (a real loss) and NOT truncated: Gymnasium's convention is
    that truncated means "the game could have gone on".
    """

    def __init__(self, env: Env, max_steps: int | None) -> None:
        self.env = env
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[Any, dict]:
        self.steps = 0
        return self.env.reset(seed=seed)

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if self.max_steps is not None and self.steps >= self.max_steps and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


def play_episode(env: Env, agent: Agent, seed: int, info_keys: Sequence[str] = ()) -> dict:
    """Play one game to the end. Returns a row: seed, return, steps, how it ended, and info_keys."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(agent.act(obs, info))
        total_reward += float(reward)
        steps += 1
    row: dict[str, Any] = {
        "seed": seed,
        "return": total_reward,  # "return" = total reward over the episode (RL term)
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    for key in info_keys:
        row[key] = info[key]  # read from the FINAL info (end-of-game totals)
    return row


def summarize(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        # std = standard deviation: how spread out the per-game results are.
        # Tetris scores vary a lot between games, so expect a large std.
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def evaluate(
    env: Env,
    make_agent: Callable[[int], Agent],
    seeds: Sequence[int],
    max_steps: int | None = None,
    info_keys: Sequence[str] = (),
) -> dict:
    """Play one game per seed and summarize.

    make_agent(seed) is called once per game and must return a ready agent.
    A factory (rather than one shared agent) lets agents with their own
    randomness start each game from a seed-derived rng, so game #7's result
    doesn't depend on which games were played before it. Deterministic
    agents can just ignore the seed: `lambda seed: agent`.

    Returns {"games": [row per game], "summary": {...}}. summary has one
    stats dict per metric ("return", "steps", and each info key), plus
    counts and speed.
    """
    capped_env = StepLimit(env, max_steps)
    rows = []
    start = time.perf_counter()
    for seed in seeds:
        rows.append(play_episode(capped_env, make_agent(seed), seed, info_keys))
    elapsed = time.perf_counter() - start

    total_steps = sum(r["steps"] for r in rows)
    summary: dict[str, Any] = {
        metric: summarize([r[metric] for r in rows])
        for metric in ("return", "steps", *info_keys)
    }
    summary.update(
        games=len(rows),
        capped=sum(r["truncated"] for r in rows),
        max_steps=max_steps,
        seconds=elapsed,
        games_per_second=len(rows) / elapsed if elapsed > 0 else float("inf"),
        steps_per_second=total_steps / elapsed if elapsed > 0 else float("inf"),
    )
    return {"games": rows, "summary": summary}


def git_commit() -> dict:
    """Current commit hash and whether there are uncommitted changes.

    Saved alongside results so a number can always be traced back to the
    exact code that produced it. "dirty" = tracked files had uncommitted
    edits, so the hash alone doesn't fully describe the code. Untracked
    files are ignored: they're typically fresh outputs (results/*.json),
    and code only matters once it's tracked anyway.
    """
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except OSError:  # git not installed
        return {"commit": None, "dirty": None}
