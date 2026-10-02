"""How did the play style change over training? Metrics per training step, plotted.

Run from the repo root:
    python -m scripts.style_report runs/lr_decay                      # saved replays only (2 per step)
    python -m scripts.style_report runs/lr_decay --fresh-games 10     # + 10 new games per checkpoint
    python -m scripts.style_report runs/raw --fresh-games 10 --checkpoint-every 5

Two sources of games:
    saved   the 2 replays each evaluation saved during training (seeds
            10100/10101). Free, but 2 games is very few: one lucky or
            unlucky piece sequence moves the average a lot.
    fresh   (--fresh-games K) load every per-evaluation checkpoint
            (checkpoints/step*.pt, kept when keep_eval_checkpoints was on)
            and play K new greedy games on seeds 20000.. (outside the
            eval/selection seed ranges; nothing is selected on these
            games, so any seeds would do). Games stop at --max-pieces
            (default 1000): style settles long before that, and it keeps
            the late-training games from taking ages. These games are
            saved as replays in <run>/style_games/ and reused next time.

Outputs, in the run folder:
    style_report.png    one small chart per metric against training step
    style_report.csv    one row per game (source, step, seed, metrics)
    style_summary.json  per step: mean of each metric over the games

Metric definitions: games/tetris/style.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from core.evaluate import evaluate
from core.replay import load_replay, make_replay, save_replay
from games.tetris.env import TetrisEnv
from games.tetris.style import METRIC_NAMES
from scripts.build_site import best_step
from scripts.replay_style import style

ROOT = Path(__file__).resolve().parents[1]
FRESH_SEED_START = 20_000
SHARE_KEYS = ("share_singles", "share_doubles", "share_triples", "share_tetrises")

# (metric, chart title) for the PNG, in reading order.
PANELS = [
    ("avg_stack_height", "Average stack height (rows)"),
    ("avg_holes", "Average holes"),
    ("avg_bumpiness", "Average bumpiness"),
    ("share_singles", "Share of lines from singles"),
    ("share_doubles", "Share of lines from doubles"),
    ("share_triples", "Share of lines from triples"),
    ("share_tetrises", "Share of lines from tetrises"),
    ("pieces_per_clear", "Pieces per line clear"),
    ("pieces", "Game length (pieces)"),
]
FRESH_COLOR = "#2a78d6"  # blue: fresh games (dots) and their per-step mean (line)
SAVED_COLOR = "#eb6834"  # orange: the 2 replays saved during training


def checkpoint_steps(run_dir: Path) -> dict[int, Path]:
    """step -> checkpoints/step<N>.pt (only the per-evaluation snapshots)."""
    return {int(p.stem[4:]): p for p in sorted((run_dir / "checkpoints").glob("step*.pt"))}


def fresh_games(run_dir: Path, games: int, max_pieces: int, every: int) -> list[dict]:
    """Style rows of K new games per checkpoint, playing (and caching) the missing ones."""
    from scripts.train import INFO_KEYS, load_eval_agent  # imports torch: only when needed

    cache = run_dir / "style_games"
    steps = checkpoint_steps(run_dir)
    if not steps:
        raise SystemExit(f"no per-evaluation checkpoints (checkpoints/step*.pt) in {run_dir}")
    chosen = sorted(steps)[::every]
    if sorted(steps)[-1] not in chosen:
        chosen.append(sorted(steps)[-1])  # always include the end of training
    seeds = list(range(FRESH_SEED_START, FRESH_SEED_START + games))
    rows = []
    t0 = time.perf_counter()
    for i, step in enumerate(chosen):
        paths = {s: cache / f"step{step:09d}_seed{s}.json" for s in seeds}
        # A cached game counts only if it was played with the same piece cap.
        missing = [s for s, p in paths.items()
                   if not p.exists() or load_replay(p)["metadata"].get("max_steps") != max_pieces]
        if missing:
            agent, _ = load_eval_agent(steps[step])
            res = evaluate(TetrisEnv(), lambda seed: agent, missing, max_pieces, INFO_KEYS, record_actions=True)
            for g in res["games"]:
                save_replay(paths[g["seed"]], make_replay(
                    "tetris", g["seed"], g["actions"], step=step, run=run_dir.name, source="style_report",
                    max_steps=max_pieces, terminated=g["terminated"], truncated=g["truncated"],
                    **{k: g[k] for k in INFO_KEYS}))
        for s, p in paths.items():
            rows.append({"source": "fresh", "seed": s, **style(load_replay(p))})
        print(f"  checkpoint {i + 1}/{len(chosen)} (step {step:,}): "
              f"{'played ' + str(len(missing)) if missing else 'cached'}  [{time.perf_counter() - t0:.0f} s]",
              flush=True)
    return rows


def saved_games(run_dir: Path) -> list[dict]:
    return [{"source": "saved", "seed": (r := load_replay(p))["seed"], **style(r)}
            for p in sorted((run_dir / "replays").glob("*.json"))]


def mean_or_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict]) -> list[dict]:
    """Per (source, step): the mean of each metric over that step's games.

    Clear shares only average games that cleared at least one line (a game
    with no lines has no "mix" to speak of; counting it as 0% everywhere
    would drag every share down).
    """
    out = []
    for source, step in sorted({(r["source"], r["step"]) for r in rows}):
        group = [r for r in rows if r["source"] == source and r["step"] == step]
        summary: dict = {"source": source, "step": step, "games": len(group)}
        for k in METRIC_NAMES:
            vals = [r[k] for r in group if not (k in SHARE_KEYS and r["lines"] == 0)]
            summary[k] = mean_or_none(vals)
        out.append(summary)
    return out


def plot(rows: list[dict], summary: list[dict], best: int | None, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # draw straight to a file, no window
    import matplotlib.pyplot as plt

    # "Small multiples": one small chart per metric, all sharing the same
    # x axis (training step), instead of one crowded chart with 9 lines.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for ax, (key, label) in zip(axes.flat, PANELS):
        for source, color in (("fresh", FRESH_COLOR), ("saved", SAVED_COLOR)):
            pts = [(r["step"], r[key]) for r in rows if r["source"] == source and r[key] is not None
                   and not (key in SHARE_KEYS and r["lines"] == 0)]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=10, color=color, alpha=0.35, linewidths=0)
            means = [(s["step"], s[key]) for s in summary if s["source"] == source and s[key] is not None]
            # The line goes through the mean of the most-sampled source
            # (fresh if there is any, else the saved replays).
            if means and (source == "fresh" or not any(r["source"] == "fresh" for r in rows)):
                xs, ys = zip(*means)
                ax.plot(xs, ys, color=color, linewidth=2)
        if best:
            ax.axvline(best, color="#888780", linestyle="--", linewidth=1)
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_xscale("log")  # evaluations are dense early in training, sparse later
        ax.grid(True, color="#e8e7e2", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if key in SHARE_KEYS:
            ax.set_ylim(-0.03, 1.03)
        if key == "pieces":
            ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("training step (log scale)")

    # One legend for the whole figure (the same two sources in every panel).
    from matplotlib.lines import Line2D
    present = {r["source"] for r in rows}
    handles = []
    if "fresh" in present:
        handles.append(Line2D([], [], color=FRESH_COLOR, marker="o", linewidth=2,
                              label="fresh games (dots) and their mean (line)"))
    if "saved" in present:
        # Without fresh games, the line is the saved replays' mean.
        handles.append(Line2D([], [], color=SAVED_COLOR, marker="o", linewidth=0 if "fresh" in present else 2,
                              label="replays saved during training"))
    if best:
        handles.append(Line2D([], [], color="#888780", linestyle="--", label="best.pt"))
    # Title on the first line, legend on its own line under it (not overlapping).
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=13)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.965), frameon=False, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def table(summary: list[dict], source: str) -> str:
    lines = [f"{'step':>9} {'games':>5} {'pieces':>7} {'height':>6} {'holes':>6} {'bump':>5}"
             f" {'1s':>5} {'2s':>5} {'3s':>5} {'4s':>5} {'pcs/clr':>7}"]
    for s in (x for x in summary if x["source"] == source):
        f = lambda v, w, p: f"{v:{w}.{p}f}" if v is not None else f"{'-':>{w}}"  # noqa: E731
        shares = " ".join(f"{100 * s[k]:4.0f}%" if s[k] is not None else f"{'-':>5}" for k in SHARE_KEYS)
        lines.append(f"{s['step']:>9,} {s['games']:>5} {f(s['pieces'], 7, 0)} {f(s['avg_stack_height'], 6, 1)} "
                     f"{f(s['avg_holes'], 6, 1)} {f(s['avg_bumpiness'], 5, 1)} {shares} {f(s['pieces_per_clear'], 7, 1)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fresh-games", type=int, default=0, help="new games per checkpoint (0 = saved replays only)")
    parser.add_argument("--max-pieces", type=int, default=1000, help="piece cap for fresh games")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="use every Nth checkpoint (raw has 200)")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir

    rows = saved_games(run_dir)
    if args.fresh_games:
        print(f"fresh games: {args.fresh_games} per checkpoint, cap {args.max_pieces} pieces")
        rows += fresh_games(run_dir, args.fresh_games, args.max_pieces, args.checkpoint_every)
    summary = summarize(rows)
    best = best_step(run_dir)

    with (run_dir / "style_report.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "step", "seed", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "style_summary.json").write_text(json.dumps(
        {"run": run_dir.name, "best_step": best, "fresh_games": args.fresh_games,
         "fresh_max_pieces": args.max_pieces, "summary": summary}, indent=1))
    n_fresh = sum(r["source"] == "fresh" for r in rows)
    title = (f"Play style over training: run {run_dir.name}  "
             f"({len(rows) - n_fresh} saved replays" + (f", {n_fresh} fresh games capped at {args.max_pieces} pieces)" if n_fresh else ")"))
    plot(rows, summary, best, title, run_dir / "style_report.png")

    for source in ("saved", "fresh"):
        if any(s["source"] == source for s in summary):
            print(f"\n{source} games, mean per training step:")
            print(table(summary, source))
    print(f"\n-> {run_dir / 'style_report.png'}, style_report.csv, style_summary.json")


if __name__ == "__main__":
    main()
