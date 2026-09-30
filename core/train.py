"""Generic self-play training loop: play, learn, log, evaluate, checkpoint, resume.

Game-agnostic (no imports from games/ or agents/). It needs:
  - an env with the Gymnasium-style reset/step interface (core.evaluate.Env),
  - a learner (see the Learner protocol below) that chooses moves and learns.

One loop iteration = one env step (for Tetris: one piece):

    learner.epsilon = schedule(step)          exploration fades over time
    action = learner.act(obs, info)           pick a move (sometimes random)
    next_obs, reward, ... = env.step(action)  play it
    learner.observe(...)                      remember what happened
    stats = learner.update()                  one small learning step (or None)

After each episode: log a row; every `eval_every_episodes` evaluate greedily
(epsilon = 0) on fixed seeds and save sample replays; every
`checkpoint_every_episodes` save a checkpoint.

Run folder (runs/<name>/, git-ignored)
--------------------------------------
    config.json           the full config
    log.csv               one row per training episode
    eval.csv              one row per evaluation (with reference numbers alongside)
    tb/                   TensorBoard event files (same numbers, as live charts)
    checkpoints/latest.pt everything needed to resume
    checkpoints/best.pt   same, from the evaluation with the best mean `best_metric`
    replays/              sample evaluation games (seed + actions), per evaluation

Seeds: training episode i is played on seed `train_seed_start + i`, so the
sequence of training games is reproducible and easy to keep disjoint from
the evaluation seeds.

Resuming (resume=True)
----------------------
Loads checkpoints/latest.pt: learner state (network, target network,
optimizer, rng), step/episode counters, best score so far, torch rng state.
Log rows written after that checkpoint are dropped, so the CSVs have no
duplicates. The replay buffer is NOT saved (see the learner for why); it
starts empty and refills before learning continues.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch

from core.checkpoint import load_checkpoint, save_checkpoint
from core.evaluate import Agent, Env, StepLimit, evaluate
from core.replay import make_replay, save_replay


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
    checkpoint_every_episodes: int = 50  # save checkpoints/latest.pt
    eval_every_episodes: int | None = 200  # None = never evaluate during training
    eval_games: int = 20  # how many of the evaluation seeds to play each time
    replays_per_eval: int = 2  # sample games saved per evaluation (the first seeds)
    best_metric: str = "score"  # best.pt = highest mean of this eval metric
    # Also save checkpoints/step<N>.pt at every evaluation, so the best few
    # can be re-checked on more games later (a 20-50 game evaluation is noisy).
    keep_eval_checkpoints: bool = False


@dataclass
class EvalSpec:
    """How to evaluate during training. Built by the (game-specific) script."""

    env: Env
    seeds: Sequence[int]  # already cut to eval_games
    max_steps: int | None  # same cap as the baseline, so numbers are comparable
    info_keys: Sequence[str]
    # Numbers to log next to ours, e.g. {"score_mean": 67036.0}: the baseline
    # agent's results on exactly these seeds.
    reference: dict[str, float] = field(default_factory=dict)
    game: str = "game"  # written into replay files
    replay_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Progress:
    """Counters that must survive a restart."""

    step: int = 0  # total env steps taken in training
    episode: int = 0  # index of the NEXT training episode to play
    updates: int = 0  # informational (the learner keeps its own)
    best: float | None = None  # best mean eval metric so far
    seconds: float = 0.0  # training wall-clock time so far, across restarts


@dataclass
class TrainResult:
    rows: list[dict]  # training-episode rows logged in this call
    evals: list[dict]  # evaluation rows logged in this call
    progress: Progress
    interrupted: bool  # stopped by Ctrl+C (a checkpoint was saved)


class Learner(Protocol):
    epsilon: float
    stat_names: tuple[str, ...]  # keys of the dict update() returns (e.g. "loss"); fixed CSV columns

    def act(self, obs: Any, info: dict) -> int: ...
    def observe(self, action: int, reward: float, terminated: bool, truncated: bool,
                next_obs: Any, next_info: dict) -> None: ...
    def update(self) -> dict[str, float] | None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
    def eval_agent(self) -> Agent: ...  # a greedy (epsilon = 0) player sharing the current network


def linear_epsilon(step: int, cfg: LoopConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / cfg.epsilon_decay_steps
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


# ----------------------------------------------------------------------
# Config files and run folders
# ----------------------------------------------------------------------
def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def format_config(config: dict) -> str:
    """Pretty-print a (nested) config so every run shows exactly what it used."""
    return json.dumps(config, indent=2)


def make_run_dir(runs_root: Path, name: str, config: dict, fresh: bool = False) -> Path:
    """Create runs/<name>/ and save config.json in it.

    Refuses to reuse an existing run folder unless fresh=True (which deletes
    it first): silently mixing two runs' logs would make the curves
    meaningless. To continue a run, resume it instead.
    """
    run_dir = Path(runs_root) / name
    if run_dir.exists():
        if not fresh:
            raise FileExistsError(f"{run_dir} already exists; resume it, pick another --name, "
                                  "or pass --fresh to delete it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(format_config(config))
    return run_dir


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
class CsvLog:
    """Append rows to a CSV with fixed columns. Flushed after every row, so the
    file is readable (e.g. in a spreadsheet) while the run is still going."""

    def __init__(self, path: Path, columns: Sequence[str], append: bool) -> None:
        self.columns = list(columns)
        exists = append and path.exists()
        self._file = open(path, "a" if exists else "w", newline="")
        self._csv = csv.DictWriter(self._file, fieldnames=self.columns)
        if not exists:
            self._csv.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._csv.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def drop_rows_after(path: Path, column: str, limit: float) -> None:
    """Keep only rows with row[column] < limit. Used on resume: rows logged
    after the checkpoint describe progress that the resumed run will redo."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = [r for r in reader if float(r[column]) < limit]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


class TensorBoard:
    """Thin wrapper: does nothing when disabled, so tests don't need tensorboard."""

    def __init__(self, run_dir: Path, enabled: bool, purge_step: int | None) -> None:
        self._w = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            # purge_step: on resume, tell TensorBoard to hide points logged
            # after the checkpoint (the same idea as drop_rows_after).
            self._w = SummaryWriter(log_dir=str(run_dir / "tb"), purge_step=purge_step)

    def log(self, prefix: str, row: dict[str, Any], step: int, skip: Sequence[str]) -> None:
        if self._w is None:
            return
        for key, value in row.items():
            if key not in skip and value is not None and value != "":
                self._w.add_scalar(f"{prefix}/{key}", float(value), step)

    def close(self) -> None:
        if self._w is not None:
            self._w.close()


# ----------------------------------------------------------------------
# Evaluation during training
# ----------------------------------------------------------------------
def run_evaluation(learner: Learner, spec: EvalSpec, cfg: LoopConfig, progress: Progress,
                   run_dir: Path) -> tuple[dict, float]:
    """Greedy evaluation on the fixed seeds. Returns (eval.csv row, best-metric value).

    Also saves the first `replays_per_eval` games as replay files. They are
    always the SAME seeds, so across a run you can watch how the agent plays
    the exact same piece sequence at different stages of training.
    """
    agent = learner.eval_agent()
    res = evaluate(spec.env, lambda seed: agent, spec.seeds, spec.max_steps, spec.info_keys,
                   record_actions=cfg.replays_per_eval > 0)
    s = res["summary"]
    row: dict[str, Any] = {"step": progress.step, "episode": progress.episode, "games": s["games"],
                           "capped": s["capped"]}
    for k in spec.info_keys:
        row[f"{k}_mean"] = s[k]["mean"]
        row[f"{k}_median"] = s[k]["median"]
    for k, v in spec.reference.items():
        row[f"ref_{k}"] = v
    row["seconds"] = s["seconds"]

    for game in res["games"][: cfg.replays_per_eval]:
        replay = make_replay(
            spec.game, game["seed"], game.pop("actions"),
            step=progress.step, episode=progress.episode,
            **{k: game[k] for k in spec.info_keys},
            terminated=game["terminated"], truncated=game["truncated"], max_steps=spec.max_steps,
            **spec.replay_metadata,
        )
        save_replay(run_dir / "replays" / f"step{progress.step:09d}_seed{game['seed']}.json", replay)
    return row, float(s[cfg.best_metric]["mean"])


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------
def checkpoint_payload(learner: Learner, progress: Progress, config: dict | None, epsilon: float) -> dict:
    return {
        "learner": learner.state_dict(),
        "progress": asdict(progress),
        "epsilon": epsilon,  # informational: the schedule recomputes it from the step
        "torch_rng": torch.get_rng_state(),
        "config": config,
    }


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------
def train(
    env: Env,
    learner: Learner,
    cfg: LoopConfig,
    run_dir: Path,
    info_keys: Sequence[str] = (),
    eval_spec: EvalSpec | None = None,
    tensorboard: bool = True,
    resume: bool = False,
    config: dict | None = None,
) -> TrainResult:
    """Train until cfg.total_steps. Ctrl+C saves a checkpoint and returns (interrupted=True).

    info_keys: end-of-episode values to log from the final info (for Tetris
    "lines", "score", "pieces"). These are the REAL game results, not the
    shaped reward the learner may use internally.
    config: the full run config, stored inside checkpoints.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    progress = Progress()
    if resume:
        payload = load_checkpoint(ckpt_dir / "latest.pt")
        learner.load_state_dict(payload["learner"])
        progress = Progress(**payload["progress"])
        torch.set_rng_state(payload["torch_rng"])
        drop_rows_after(run_dir / "log.csv", "episode", progress.episode)
        drop_rows_after(run_dir / "eval.csv", "step", progress.step + 1)
        print(f"resumed from step {progress.step}, episode {progress.episode}, "
              f"best {cfg.best_metric} {progress.best}", flush=True)

    stat_keys = list(learner.stat_names)
    columns = ["episode", "step", "seed", "steps", "return", "terminated", *info_keys, "epsilon",
               *stat_keys, "steps_per_sec", "elapsed_sec"]
    log = CsvLog(run_dir / "log.csv", columns, append=resume)
    eval_log: CsvLog | None = None
    tb = TensorBoard(run_dir, tensorboard, purge_step=progress.step if resume else None)
    env = StepLimit(env, cfg.max_episode_steps)

    def save(name: str) -> None:
        progress.updates = int(getattr(learner, "updates", 0))
        save_checkpoint(ckpt_dir / name, checkpoint_payload(learner, progress, config, learner.epsilon))

    rows: list[dict] = []
    evals: list[dict] = []
    interrupted = False
    session_start = time.perf_counter()
    seconds_before = progress.seconds
    try:
        while progress.step < cfg.total_steps:
            seed = cfg.train_seed_start + progress.episode
            obs, info = env.reset(seed=seed)
            ep_steps = 0
            ep_return = 0.0
            ep_stats: dict[str, list[float]] = {}
            ep_start = time.perf_counter()
            terminated = truncated = False
            while not (terminated or truncated):
                learner.epsilon = linear_epsilon(progress.step, cfg)
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
                progress.step += 1
            ep_seconds = time.perf_counter() - ep_start
            progress.seconds = seconds_before + time.perf_counter() - session_start

            row: dict[str, Any] = {
                "episode": progress.episode,
                "step": progress.step,
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
                "elapsed_sec": progress.seconds,
            }
            log.write(row)
            tb.log("train", row, progress.step, skip=("episode", "step", "seed"))
            rows.append(row)
            progress.episode += 1  # this episode is done; the next one is progress.episode

            if progress.episode % cfg.print_every_episodes == 0 or progress.step >= cfg.total_steps:
                print_progress(rows[-cfg.print_every_episodes:], info_keys, stat_keys, progress, learner.epsilon)

            last = progress.step >= cfg.total_steps
            if eval_spec is not None and cfg.eval_every_episodes and (
                    progress.episode % cfg.eval_every_episodes == 0 or last):
                eval_row, metric = run_evaluation(learner, eval_spec, cfg, progress, run_dir)
                if eval_log is None:
                    eval_log = CsvLog(run_dir / "eval.csv", list(eval_row), append=True)
                eval_log.write(eval_row)
                tb.log("eval", eval_row, progress.step, skip=("step", "episode"))
                evals.append(eval_row)
                if cfg.keep_eval_checkpoints:
                    save(f"step{progress.step:09d}.pt")
                is_best = progress.best is None or metric > progress.best
                if is_best:
                    progress.best = metric
                    save("best.pt")
                print_eval(eval_row, eval_spec, cfg, is_best)
            if progress.episode % cfg.checkpoint_every_episodes == 0 or last:
                save("latest.pt")
    except KeyboardInterrupt:
        # The episode in progress is abandoned: `progress.episode` still points
        # at it, so a resumed run replays that seed from the start. Steps
        # already taken stay counted (the network did learn from them).
        interrupted = True
        progress.seconds = seconds_before + time.perf_counter() - session_start
        save("latest.pt")
        print(f"\ninterrupted: saved checkpoint at step {progress.step}, episode {progress.episode}", flush=True)
    finally:
        log.close()
        if eval_log is not None:
            eval_log.close()
        tb.close()
    return TrainResult(rows, evals, progress, interrupted)


def print_progress(recent: list[dict], info_keys: Sequence[str], stat_keys: Sequence[str],
                   progress: Progress, epsilon: float) -> None:
    parts = [f"ep {progress.episode - 1:>6}", f"step {progress.step:>8}"]
    for k in info_keys:
        parts.append(f"{k} {np.mean([r[k] for r in recent]):>7.1f}")
    for k in stat_keys:
        vals = [r[k] for r in recent if r[k] != ""]
        if vals:
            parts.append(f"{k} {np.mean(vals):.4f}")
    parts.append(f"eps {epsilon:.3f}")
    parts.append(f"{np.mean([r['steps_per_sec'] for r in recent]):>5.0f} steps/s")
    print("  ".join(parts), flush=True)


def print_eval(row: dict, spec: EvalSpec, cfg: LoopConfig, is_best: bool) -> None:
    parts = [f"EVAL step {row['step']}: {row['games']} games (epsilon 0)"]
    for k in spec.info_keys:
        ref = spec.reference.get(f"{k}_mean")
        parts.append(f"{k} {row[f'{k}_mean']:.1f}" + (f" (ref {ref:.1f})" if ref is not None else ""))
    parts.append(f"capped {row['capped']}/{row['games']}, {row['seconds']:.0f}s")
    if is_best:
        parts.append(f"-> new best {cfg.best_metric}, saved best.pt")
    print("  ".join(parts), flush=True)
