"""Generic self-play training loop: play episodes, feed every transition to a learner, log.

Game-agnostic (no imports from games/ or agents/). It needs:
  - an env with the Gymnasium-style reset/step interface (core.evaluate.Env),
  - a learner (see the Learner protocol below) that chooses moves and learns.

One loop iteration = one env step (for Tetris: one piece):

    learner.epsilon = schedule(step)          exploration fades over time
    action = learner.act(obs, info)           pick a move (sometimes random)
    next_obs, reward, ... = env.step(action)  play it
    learner.observe(...)                      remember what happened
    stats = learner.update()                  one small learning step (or None)

Run folder
----------
Each run lives in runs/<name>/ (git-ignored):
    config.json   the full config the run was started with
    log.csv       one row per episode (open it in a spreadsheet, or pandas)
    tb/           TensorBoard event files (same numbers, as live charts)

Seeds: episode i is played on seed `train_seed_start + i`. That makes the
sequence of training games reproducible, and a single number to check
against the evaluation seeds (training must never see those games).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from core.evaluate import Env, StepLimit


@dataclass
class LoopConfig:
    """Settings of the generic loop. Every field is a hyperparameter you choose."""

    total_steps: int = 20_000  # stop after the episode during which this many steps are reached
    max_episode_steps: int | None = 1_000  # training games are cut off (truncated) here
    train_seed_start: int = 1_000_000  # episode i uses seed train_seed_start + i
    # Exploration schedule: epsilon falls in a straight line from start to end
    # over the first `epsilon_decay_steps` steps, then stays at end.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 10_000
    print_every_episodes: int = 20  # console progress line (the CSV logs every episode)


class Learner(Protocol):
    epsilon: float
    stat_names: tuple[str, ...]  # keys of the dict update() returns (e.g. "loss"); fixed CSV columns

    def act(self, obs: Any, info: dict) -> int: ...
    def observe(self, action: int, reward: float, terminated: bool, truncated: bool,
                next_obs: Any, next_info: dict) -> None: ...
    def update(self) -> dict[str, float] | None: ...


def linear_epsilon(step: int, cfg: LoopConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / cfg.epsilon_decay_steps
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


# ----------------------------------------------------------------------
# Config files
# ----------------------------------------------------------------------
def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def format_config(config: dict) -> str:
    """Pretty-print a (nested) config so every run shows exactly what it used."""
    return json.dumps(config, indent=2)


def make_run_dir(runs_root: Path, name: str, config: dict, fresh: bool = False) -> Path:
    """Create runs/<name>/ and save config.json in it.

    Refuses to reuse an existing run folder unless fresh=True (which deletes
    it first): silently mixing two runs' logs in one CSV would make the
    curves meaningless. (Resuming an interrupted run comes in step 3c.)
    """
    run_dir = Path(runs_root) / name
    if run_dir.exists():
        if not fresh:
            raise FileExistsError(f"{run_dir} already exists; pick another --name or pass --fresh to delete it")
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(format_config(config))
    return run_dir


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
class RunLogger:
    """Writes one row per episode to log.csv and the same numbers to TensorBoard.

    CSV is the durable, simple record. TensorBoard is for watching curves
    live while training runs. The x-axis in TensorBoard is the total env step
    count, not the episode number, because later episodes are much longer:
    plotting by step shows learning per unit of experience.
    """

    def __init__(self, run_dir: Path, columns: Sequence[str], tensorboard: bool = True) -> None:
        self.columns = list(columns)
        self._file = open(run_dir / "log.csv", "w", newline="")
        self._csv = csv.DictWriter(self._file, fieldnames=self.columns)
        self._csv.writeheader()
        self._tb = None
        if tensorboard:
            # Imported here so the rest of the loop (and its tests) work even
            # without the tensorboard package installed.
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(run_dir / "tb"))

    def log(self, row: dict[str, Any], step: int) -> None:
        self._csv.writerow(row)
        self._file.flush()  # so the CSV is readable while the run is still going
        if self._tb is not None:
            for key, value in row.items():
                if key not in ("episode", "step", "seed") and value is not None and value != "":
                    self._tb.add_scalar(key, float(value), step)

    def close(self) -> None:
        self._file.close()
        if self._tb is not None:
            self._tb.close()


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------
def train(
    env: Env,
    learner: Learner,
    cfg: LoopConfig,
    run_dir: Path,
    info_keys: Sequence[str] = (),
    tensorboard: bool = True,
) -> list[dict]:
    """Train until cfg.total_steps. Returns the per-episode rows (also written to disk).

    info_keys: end-of-episode values to log from the final info (for Tetris
    "lines", "score", "pieces"). These are the REAL game results, not the
    shaped reward the learner may be using internally.
    """
    stat_keys = list(learner.stat_names)
    env = StepLimit(env, cfg.max_episode_steps)
    rows: list[dict] = []
    logger: RunLogger | None = None
    step = 0
    episode = 0
    start = time.perf_counter()
    try:
        while step < cfg.total_steps:
            seed = cfg.train_seed_start + episode
            obs, info = env.reset(seed=seed)
            ep_steps = 0
            ep_return = 0.0
            ep_stats: dict[str, list[float]] = {}
            ep_start = time.perf_counter()
            terminated = truncated = False
            while not (terminated or truncated):
                learner.epsilon = linear_epsilon(step, cfg)
                action = learner.act(obs, info)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                learner.observe(action, reward, terminated, truncated, next_obs, next_info)
                stats = learner.update()
                if stats:
                    for k, v in stats.items():
                        ep_stats.setdefault(k, []).append(v)
                obs, info = next_obs, next_info
                ep_return += float(reward)
                ep_steps += 1
                step += 1
            ep_seconds = time.perf_counter() - ep_start

            row: dict[str, Any] = {
                "episode": episode,
                "step": step,
                "seed": seed,
                "steps": ep_steps,
                "return": ep_return,  # raw env reward (for Tetris = score)
                "terminated": int(terminated),
                **{k: info[k] for k in info_keys},
                "epsilon": round(learner.epsilon, 4),
                # Mean of each learner stat over this episode's updates; blank
                # before learning starts (the buffer is still filling up).
                **{k: float(np.mean(ep_stats[k])) if k in ep_stats else "" for k in stat_keys},
                "steps_per_sec": ep_steps / ep_seconds if ep_seconds > 0 else 0.0,
                "elapsed_sec": time.perf_counter() - start,
            }
            if logger is None:
                logger = RunLogger(run_dir, list(row), tensorboard=tensorboard)
            logger.log(row, step)
            rows.append(row)

            if episode % cfg.print_every_episodes == 0 or step >= cfg.total_steps:
                recent = rows[-cfg.print_every_episodes:]
                parts = [f"ep {episode:>6}", f"step {step:>8}"]
                for k in info_keys:
                    parts.append(f"{k} {np.mean([r[k] for r in recent]):>7.1f}")
                for k in stat_keys:
                    vals = [r[k] for r in recent if r[k] != ""]
                    if vals:
                        parts.append(f"{k} {np.mean(vals):.4f}")
                parts.append(f"eps {learner.epsilon:.3f}")
                parts.append(f"{np.mean([r['steps_per_sec'] for r in recent]):>5.0f} steps/s")
                print("  ".join(parts), flush=True)
            episode += 1
    finally:
        if logger is not None:
            logger.close()
    return rows


def config_dict(cfg: Any) -> dict:
    """A dataclass config as a plain dict (for saving/printing)."""
    return asdict(cfg)
