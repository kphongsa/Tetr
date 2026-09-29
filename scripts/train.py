"""Train the afterstate value network on Tetris by self-play.

Run from the repo root:
    python -m scripts.train --config configs/smoke.json            # ~1 minute sanity run
    python -m scripts.train --config configs/smoke.json --fresh    # same, deleting an old runs/smoke/
    python -m scripts.train --config configs/smoke.json --name try2
    python -m scripts.train --resume smoke                         # continue runs/smoke/ where it stopped
    python -m scripts.train --resume smoke --total-steps 50000     # ...and train for longer than planned

Ctrl+C saves a checkpoint and exits; --resume continues from it.

Writes runs/<name>/{config.json, log.csv, eval.csv, tb/, checkpoints/, replays/}.
Watch live with
    tensorboard --logdir runs
then open http://localhost:6006 in a browser.

Evaluation during training uses SELECTION seeds 10100.. (the first
`eval_games` of them) and the piece cap of results/baseline_heuristic.json,
and logs the tuned heuristic's results on those same games alongside ours
(ref_* columns). Why not the official evaluation seeds 10000..10099: best.pt
is the checkpoint that did best on these games. If they were also the final
test games, picking the best of many checkpoints on them would flatter the
final number a little (the "winner's curse"). Separate seeds keep the final
comparison honest.

This script is the only place where Tetris and the generic parts meet:
it builds the Tetris env and candidates function and hands them to the
game-agnostic agent, learner and training loop.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agents.afterstate_value_agent import AfterstateTDLearner, AfterstateValueAgent, TDConfig, make_value_net
from agents.heuristic_agent import HeuristicAgent
from core.checkpoint import load_checkpoint
from core.evaluate import evaluate, git_commit
from core.train import EvalSpec, LoopConfig, format_config, load_config, make_run_dir, train
from games.tetris.env import TetrisEnv
from games.tetris.features import FEATURE_NAMES, candidates
from scripts.evaluate import EVAL_FIRST_SEED, EVAL_GAMES

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline_heuristic.json"
SELECTION_REFERENCE = ROOT / "results" / "selection_reference.json"
# Seeds for in-training evaluation / choosing best.pt. Disjoint from tuning
# (0..44), the official evaluation seeds (10000..10099) and training (1,000,000+).
SELECTION_FIRST_SEED = 10_100
SELECTION_MAX_GAMES = 100  # seeds 10100..10199 are reserved for this
INFO_KEYS = ("lines", "score", "pieces")


def build(config: dict) -> tuple[TetrisEnv, AfterstateTDLearner, LoopConfig]:
    """Turn a config dict into ready-to-train objects. Unknown keys raise TypeError."""
    loop = LoopConfig(**config["loop"])
    td_kwargs = dict(config["learner"])
    td_kwargs["hidden"] = tuple(td_kwargs.get("hidden", TDConfig.hidden))  # JSON lists -> tuple
    td = TDConfig(**td_kwargs)

    # Training must never play the evaluation games (that would be training on the test).
    last_reserved = SELECTION_FIRST_SEED + SELECTION_MAX_GAMES - 1
    if loop.train_seed_start <= last_reserved:
        raise ValueError(f"train_seed_start must be above the evaluation and selection seeds "
                         f"({EVAL_FIRST_SEED}..{last_reserved})")
    if loop.eval_games > SELECTION_MAX_GAMES:
        raise ValueError(f"eval_games must be at most {SELECTION_MAX_GAMES}")

    torch.set_num_threads(config.get("torch_threads", 4))
    env = TetrisEnv()  # the loop's StepLimit applies max_episode_steps
    net = make_value_net(len(FEATURE_NAMES), td.hidden, seed=td.net_seed)
    agent = AfterstateValueAgent(net, candidates, np.random.default_rng(td.rng_seed),
                                 gamma=td.gamma, reward_scale=td.reward_scale)
    learner = AfterstateTDLearner(agent, td, n_features=len(FEATURE_NAMES), max_candidates=env.num_actions)
    return env, learner, loop


def full_config(config: dict) -> dict:
    """The config with every default filled in, so the saved copy is complete."""
    _, learner, loop = build(config)
    out = copy.deepcopy(config)
    out["loop"] = asdict(loop)
    out["learner"] = asdict(learner.cfg)
    out.setdefault("torch_threads", 4)
    return out


def load_eval_agent(checkpoint_path: Path) -> tuple[AfterstateValueAgent, dict]:
    """Greedy agent from a checkpoint (rebuilt from the config stored inside it), plus the checkpoint."""
    ck = load_checkpoint(checkpoint_path)
    _, learner, _ = build(ck["config"])
    learner.load_state_dict(ck["learner"])
    return learner.eval_agent(), ck


def tuned_reference(seeds: list[int], cap: int, baseline_path: Path = BASELINE,
                    cache_path: Path = SELECTION_REFERENCE) -> dict[int, dict]:
    """The tuned heuristic's result on each selection seed (seed -> game row).

    The heuristic is deterministic, so each game only ever needs playing once.
    Results are cached in results/selection_reference.json; only seeds that
    are missing (or a changed cap / weights) cause games to be played.
    """
    baseline = json.loads(Path(baseline_path).read_text())
    key = {"cap": cap, "weights": baseline["weights"]}
    cache = json.loads(Path(cache_path).read_text()) if Path(cache_path).exists() else {}
    games = {int(k): v for k, v in cache.get("games", {}).items()} if cache.get("key") == key else {}
    missing = [s for s in seeds if s not in games]
    if missing:
        print(f"playing the tuned heuristic on {len(missing)} selection seed(s) (once; cached) ...", flush=True)
        res = evaluate(TetrisEnv(), lambda seed: HeuristicAgent(baseline["weights"]), missing, cap, INFO_KEYS)
        games.update({g["seed"]: g for g in res["games"]})
        Path(cache_path).write_text(json.dumps({
            "description": "Tuned heuristic on the selection seeds (in-training evaluation). Cache for scripts/train.py.",
            "key": key, "git": git_commit(), "games": {str(k): games[k] for k in sorted(games)}}, indent=1))
    return {s: games[s] for s in seeds}


def eval_spec(loop: LoopConfig, run_name: str, baseline_path: Path = BASELINE,
              reference_path: Path = SELECTION_REFERENCE) -> EvalSpec:
    """Evaluation on the selection seeds with the baseline's cap, plus the tuned heuristic on the same games."""
    baseline = json.loads(Path(baseline_path).read_text())
    seeds = list(range(SELECTION_FIRST_SEED, SELECTION_FIRST_SEED + loop.eval_games))
    cap = baseline["max_pieces"]
    games = tuned_reference(seeds, cap, baseline_path, reference_path)
    reference = {f"{k}_mean": float(np.mean([games[s][k] for s in seeds])) for k in INFO_KEYS}
    return EvalSpec(
        env=TetrisEnv(),
        seeds=seeds,
        max_steps=cap,
        info_keys=INFO_KEYS,
        reference=reference,
        game="tetris",
        replay_metadata={"agent": "afterstate_value", "run": run_name, "git": git_commit()["commit"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="start a new run from this config")
    parser.add_argument("--resume", metavar="NAME", help="continue runs/NAME/ from its latest checkpoint")
    parser.add_argument("--name", help="run name for a new run (default: the config's 'name')")
    parser.add_argument("--fresh", action="store_true", help="delete runs/<name>/ first if it exists")
    parser.add_argument("--total-steps", type=int, help="override loop.total_steps (e.g. to extend a run)")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    if (args.config is None) == (args.resume is None):
        parser.error("give exactly one of --config (new run) or --resume NAME")

    if args.resume:
        run_dir = ROOT / "runs" / args.resume
        if not (run_dir / "checkpoints" / "latest.pt").exists():
            parser.error(f"no checkpoint to resume in {run_dir}")
        config = load_config(run_dir / "config.json")
    else:
        config = load_config(args.config)
        if args.name:
            config["name"] = args.name
        config = full_config(config)
    if args.total_steps is not None:
        config["loop"]["total_steps"] = args.total_steps

    print(f"run {config['name']!r}, config:\n{format_config(config)}\n")
    if args.resume:
        (run_dir / "config.json").write_text(format_config(config))  # records any --total-steps change
    else:
        run_dir = make_run_dir(ROOT / "runs", config["name"], config, fresh=args.fresh)

    env, learner, loop = build(config)
    spec = eval_spec(loop, config["name"])
    print(f"replay buffer: {learner.buffer.nbytes() / 1e6:.0f} MB, "
          f"network: {sum(p.numel() for p in learner.net.parameters())} parameters")
    print(f"evaluating every {loop.eval_every_episodes} episodes on seeds {spec.seeds[0]}..{spec.seeds[-1]}, "
          f"cap {spec.max_steps}; tuned heuristic on these games: "
          + ", ".join(f"{k} {v:.1f}" for k, v in spec.reference.items()) + "\n")

    result = train(env, learner, loop, run_dir, INFO_KEYS, eval_spec=spec,
                   tensorboard=not args.no_tensorboard, resume=bool(args.resume), config=config)
    rel = run_dir.relative_to(ROOT)
    if result.interrupted:
        print(f"resume with:  python -m scripts.train --resume {config['name']}")
    else:
        print(f"\ndone. logs in {rel}, checkpoints in {rel / 'checkpoints'}")


if __name__ == "__main__":
    main()
