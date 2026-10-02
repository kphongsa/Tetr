"""Export chosen replays as frames into site/data/, for the web viewer in site/.

Run from the repo root:
    python -m scripts.build_site                       # default selection (below)
    python -m scripts.build_site --runs lr_decay raw   # only these runs
    python -m scripts.build_site --per-run 30          # more training steps per run
    python -m scripts.build_site --all                 # every replay (~46 MB of JSON!)
    python -m scripts.build_site --thoughts 0          # without the "AI's options" overlay

Then serve the folder and open the viewer:
    python -m http.server 8000 --directory site       ->  http://localhost:8000

Default selection, per run: up to --per-run evaluations spread evenly over
training, always including the first, the last, and the one best.pt came
from (highest mean of the run's best_metric in eval.csv); both saved seeds
of each. That keeps site/data/ to a few MB while still covering early, best
and late training.

Output:
    site/data/index.json             the list the viewer's picker shows
    site/data/<run>/<replay>.json    one frames file per game (format:
                                     games/tetris/frames.py)
site/data/ is rebuilt from scratch every time (it's generated, git-ignored).

Thoughts (step 4e): for every game whose training step still has a
checkpoint, the network's top --thoughts options per piece are added
(scripts/thoughts.py), for the viewer's "Show AI's options" overlay.
Adds ~60% to the file size and ~10 s per 1,000 pieces.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from core.replay import load_replay
from scripts.export_frames import tetris_frames
from scripts.thoughts import add_thoughts, checkpoint_for_step

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["lr_decay", "full", "raw"]


def best_step(run_dir: Path) -> int | None:
    """Training step of the evaluation best.pt was saved from (None if unknown)."""
    try:
        metric = json.loads((run_dir / "config.json").read_text())["loop"].get("best_metric", "score")
        rows = list(csv.DictReader((run_dir / "eval.csv").open()))
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if not rows:
        return None
    # Ties: the first one wins, same as training (a new best must be strictly better).
    best = max(rows, key=lambda r: float(r[f"{metric}_mean"]))
    return int(best["step"])


def pick_steps(steps: list[int], per_run: int, must: list[int]) -> list[int]:
    """Up to per_run steps evenly spaced through `steps` (sorted), plus `must`."""
    steps = sorted(set(steps))
    if per_run >= len(steps):
        return steps
    if per_run <= 1:
        chosen = {steps[-1]}
    else:
        # Evenly spaced positions in the LIST of evaluations (not in step
        # numbers): evaluations are denser early in training, where play
        # changes fastest, so this keeps more early snapshots.
        chosen = {steps[round(i * (len(steps) - 1) / (per_run - 1))] for i in range(per_run)}
    chosen |= {s for s in must if s in steps}
    return sorted(chosen)


def build(runs_root: Path, runs: list[str], out: Path, per_run: int = 10, everything: bool = False,
          thoughts: int = 0) -> list[dict]:
    """Export the selected replays of `runs` into `out`; returns the index entries.

    thoughts > 0: add the agent's top-`thoughts` options per piece wherever
    the checkpoint from the replay's training step exists.
    """
    if out.exists():
        shutil.rmtree(out)
    entries = []
    for run in runs:
        run_dir = runs_root / run
        paths = sorted((run_dir / "replays").glob("*.json"))
        if not paths:
            print(f"skipping {run}: no replays in {run_dir / 'replays'}")
            continue
        replays = {p: load_replay(p) for p in paths}
        best = best_step(run_dir)
        steps = [r["metadata"]["step"] for r in replays.values()]
        keep = set(steps) if everything else set(pick_steps(steps, per_run, [best] if best else []))
        for path, replay in replays.items():
            meta = replay["metadata"]
            if meta["step"] not in keep:
                continue
            doc = tetris_frames(replay)  # re-simulates AND verifies; raises on mismatch
            checkpoint = checkpoint_for_step(run_dir, meta["step"]) if thoughts else None
            if checkpoint is not None:
                from scripts.train import load_eval_agent  # imports torch: only when needed
                agent, _ = load_eval_agent(checkpoint)
                add_thoughts(doc, replay, agent, checkpoint, thoughts)
            rel = Path(run) / path.name
            (out / run).mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(json.dumps(doc, separators=(",", ":")))
            entries.append({
                "id": f"{run}/{path.stem}",
                "file": rel.as_posix(),  # forward slashes: it's a URL path, not a Windows path
                "run": run,
                "step": meta["step"],
                "seed": replay["seed"],
                "pieces": meta["pieces"],
                "lines": meta["lines"],
                "score": meta["score"],
                "best": meta["step"] == best,
                "thoughts": checkpoint is not None,
            })
        mine = [e for e in entries if e["run"] == run]
        print(f"{run}: {len(mine)} games from {len(keep)} training steps"
              + (f" (best.pt = step {best:,})" if best else "")
              + (f", {sum(e['thoughts'] for e in mine)} with the AI's options" if thoughts else ""))

    entries.sort(key=lambda e: (e["run"], e["step"], e["seed"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps({"replays": entries}, indent=1))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--per-run", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="every replay of the chosen runs")
    parser.add_argument("--out", type=Path, default=ROOT / "site" / "data")
    parser.add_argument("--thoughts", type=int, default=4, help="top-k options per piece (0 = off)")
    args = parser.parse_args()
    entries = build(ROOT / "runs", args.runs, args.out, args.per_run, args.all, args.thoughts)
    total = sum(f.stat().st_size for f in args.out.rglob("*.json"))
    print(f"{len(entries)} games -> {args.out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
