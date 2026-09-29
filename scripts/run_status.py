"""Status report for a training run, safe to run while training is going.

Run from the repo root (in a second terminal while training):
    python -m scripts.run_status runs/full
    python -m scripts.run_status runs/full --window 100 --patience 8

It only READS files the training loop writes (log.csv, eval.csv,
config.json) plus results/baseline_heuristic.json, so it never disturbs the
run. It prints:
  - progress: episodes, pieces, elapsed time, pieces/sec, rough time left;
  - training games: average lines/score recently vs earlier in the run;
  - evaluations: latest and best, next to the random / Lee / tuned baselines;
  - the network's predicted values and recent loss;
  - HEALTH WARNINGS in plain words;
and saves learning_curve.png in the run folder.

Why the numbers can look inconsistent
-------------------------------------
Training games and evaluation games are different things:
  - training games use exploration (epsilon) and are cut off at
    loop.max_episode_steps pieces (2,000 in configs/full.json);
  - evaluation games are greedy (no exploration), use the fixed SELECTION
    seeds 10100.., and are capped at 10,000 pieces like the baseline.
So only the EVALUATION numbers are comparable to the baselines. The baseline
rows marked "(100 seeds)" are on the official seeds 10000..10099, a different
set of games; the tuned heuristic on the same selection seeds is the fairest
comparison. The official comparison is scripts/compare.py after training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline_heuristic.json"

# Upper bound on lines per piece in the long run: every piece adds 4 cells,
# every cleared line removes 10, and the board can't grow forever, so on
# average at most 4/10 = 0.4 lines per piece.
MAX_LINES_PER_PIECE = 0.4
# The most points one line can earn: a 4-line clear (tetris) pays 800 = 200 per line.
MAX_POINTS_PER_LINE = 200


# ----------------------------------------------------------------------
# Reading the run's files
# ----------------------------------------------------------------------
def _num(v: str) -> float | None:
    """CSV cell -> float, or None for a blank cell (e.g. loss before learning starts)."""
    return None if v == "" else float(v)


def read_rows(path: Path) -> list[dict[str, Any]]:  # values: float, or None for blank cells
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def column(rows: list[dict], key: str) -> np.ndarray:
    """One column as an array, skipping rows where it's blank or missing."""
    return np.array([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)


# ----------------------------------------------------------------------
# Baselines and the value ceiling
# ----------------------------------------------------------------------
def load_baselines(path: Path = BASELINE) -> dict[str, dict[str, float]]:
    """Mean lines/score/pieces of each step 2 agent over all 100 evaluation seeds."""
    b = json.loads(Path(path).read_text())
    summaries = {**b["other_summaries"], b["agent"]: b["summary"]}
    return {name: {k: s[k]["mean"] for k in ("lines", "score", "pieces")} for name, s in summaries.items()}


def value_ceilings(learner_cfg: dict, baselines: dict[str, dict[str, float]]) -> tuple[float, float]:
    """(hard ceiling, heuristic-level value) for the network's predictions.

    The network predicts V = sum over future pieces of gamma^k * (shaped reward).
    Shaped reward per piece = reward_scale * points + survival_bonus. If every
    future piece earned the most possible reward r_max, V would be
    r_max * (1 + gamma + gamma^2 + ...) = r_max / (1 - gamma).

      hard ceiling: r_max from the best sustainable rate (0.4 lines/piece, all
                    tetrises). No correct prediction can be above this.
      heuristic level: the same formula with the tuned heuristic's actual
                    points per piece. A good agent's values sit near or below it.
    """
    scale, bonus, gamma = learner_cfg["reward_scale"], learner_cfg["survival_bonus"], learner_cfg["gamma"]
    horizon = 1.0 / (1.0 - gamma)  # 100 pieces for gamma = 0.99: "how far ahead" the value looks
    hard = (bonus + scale * MAX_LINES_PER_PIECE * MAX_POINTS_PER_LINE) * horizon
    tuned = baselines.get("heuristic (tuned)")
    typical = (bonus + scale * tuned["score"] / tuned["pieces"]) * horizon if tuned else float("nan")
    return hard, typical


# ----------------------------------------------------------------------
# Health checks (pure functions, so they're easy to test)
# ----------------------------------------------------------------------
def health_warnings(log: list[dict], evals: list[dict], hard_ceiling: float,
                    window: int = 50, patience: int = 5) -> list[str]:
    warnings: list[str] = []
    losses = column(log, "loss")

    # 1. Loss is nan/inf: the network's numbers have blown up; nothing after this is meaningful.
    bad = [r for r in log if r.get("loss") is not None and not math.isfinite(r["loss"])]
    if bad:
        warnings.append(f"LOSS IS NAN/INF in {len(bad)} episode(s), first at episode {int(bad[0]['episode'])}. "
                        "Training has diverged (numbers blew up). Stop it and tell Claude.")

    # 2. Loss jumped >10x: compare the median of the most recent episodes with
    # the median of a longer stretch just before them. Medians, not means, so
    # one noisy episode doesn't trigger it.
    finite = losses[np.isfinite(losses)]
    recent_n = max(5, window // 5)
    if len(finite) >= recent_n + window:
        recent = float(np.median(finite[-recent_n:]))
        before = float(np.median(finite[-(recent_n + window):-recent_n]))
        if before > 0 and recent > 10 * before:
            warnings.append(f"LOSS JUMPED {recent / before:.0f}x: median {recent:.4f} over the last {recent_n} "
                            f"episodes vs {before:.4f} over the {window} before. Often the start of divergence.")

    # 3. Predicted values above what's possible.
    vmax = column(log, "value_max")
    if len(vmax) == 0:
        vmax = column(log, "value_mean")  # older runs didn't log value_max
    if len(vmax) and float(vmax[-window:].max()) > hard_ceiling:
        warnings.append(f"PREDICTED VALUES TOO HIGH: max {float(vmax[-window:].max()):.1f} > ceiling "
                        f"{hard_ceiling:.1f}. The network believes in more future reward than the game can "
                        "pay: overestimation feeding on itself.")

    # 4. No progress in evaluation lines for `patience` evaluations.
    lines = column(evals, "lines_mean")
    if len(lines) > patience:
        best_before = float(lines[:-patience].max())
        best_recent = float(lines[-patience:].max())
        if best_recent <= best_before:
            warnings.append(f"NO IMPROVEMENT in the last {patience} evaluations: best recent lines "
                            f"{best_recent:.1f} vs {best_before:.1f} earlier. Could be a plateau "
                            "(fine near the end of a run) or forgetting.")
    return warnings


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def _window_stats(rows: list[dict]) -> str:
    if not rows:
        return "-"
    capped = sum(1 for r in rows if r["terminated"] == 0)
    return (f"lines {np.mean(column(rows, 'lines')):7.1f}  score {np.mean(column(rows, 'score')):9,.0f}  "
            f"pieces {np.mean(column(rows, 'pieces')):6.0f}  eps {rows[-1]['epsilon']:.3f}  "
            f"(episodes {int(rows[0]['episode'])}-{int(rows[-1]['episode'])}, {capped} hit training cap)")


def report(run_dir: Path, window: int = 50, patience: int = 5, baseline_path: Path = BASELINE) -> str:
    config: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    log = read_rows(run_dir / "log.csv")
    evals = read_rows(run_dir / "eval.csv")
    baselines = load_baselines(baseline_path)
    hard, typical = value_ceilings(config["learner"], baselines)
    total = config["loop"]["total_steps"]
    out: list[str] = [f"=== run {config['name']!r} ({run_dir}) ==="]
    if not log:
        return "\n".join(out + ["no episodes logged yet"])

    # --- progress ---
    last = log[-1]
    step, elapsed = last["step"], last["elapsed_sec"]
    rate = step / elapsed if elapsed else 0.0  # includes evaluation time: the honest overall rate
    train_rate = float(np.mean(column(log[-window:], "steps_per_sec")))
    # Time left uses the RECENT rate (last `window` episodes, evaluations
    # included): early in a run games are short and evaluations dominate, so
    # the whole-run average badly overestimates how long the rest will take.
    # Games that took over 5 minutes mean the process was paused (laptop
    # sleep/suspend); their time is left out so a pause doesn't distort it.
    def paused(r: dict) -> bool:
        return bool(r.get("steps_per_sec")) and r["steps"] / r["steps_per_sec"] > 300

    first = log[-window - 1] if len(log) > window else log[0]
    recent = [r for r in log if r["episode"] > first["episode"]]
    pause_secs = sum(r["steps"] / r["steps_per_sec"] for r in recent if paused(r))
    pause_steps = sum(r["steps"] for r in recent if paused(r))
    span = elapsed - first["elapsed_sec"] - pause_secs
    recent_rate = (step - first["step"] - pause_steps) / span if span > 0 else rate
    age = time.time() - (run_dir / "log.csv").stat().st_mtime
    out.append(f"progress   episode {int(last['episode']) + 1:,}, pieces {int(step):,} / {total:,} "
               f"({100 * step / total:.1f}%), training time {_fmt_time(elapsed)}")
    out.append(f"speed      {rate:.0f} pieces/s whole run, {recent_rate:.0f} recently (incl. evaluations), "
               f"{train_rate:.0f} while playing; ~{_fmt_time((total - step) / recent_rate) if recent_rate else '?'} "
               "left at the recent rate")
    stalls = [r for r in log if paused(r)]
    if stalls:
        r = stalls[-1]
        out.append(f"           note: {len(stalls)} game(s) took over 5 minutes (latest: episode "
                   f"{int(r['episode'])}, {_fmt_time(r['steps'] / r['steps_per_sec'])}); the laptop probably "
                   "slept. Harmless, just slower.")
    out.append(f"           log.csv last written {_fmt_time(age)} ago"
               + ("  <- is training still running?" if age > 600 else ""))

    # --- training games ---
    out.append("\ntraining games (exploring, capped at "
               f"{config['loop']['max_episode_steps']} pieces; NOT comparable to baselines)")
    out.append(f"  {'first ' + str(window):<12} {_window_stats(log[:window])}")
    if len(log) > 2 * window:
        out.append(f"  {'previous ' + str(window):<12} {_window_stats(log[-2 * window:-window])}")
    out.append(f"  {'last ' + str(window):<12} {_window_stats(log[-window:])}")

    # --- evaluations ---
    out.append("\nevaluation (greedy, selection seeds 10100.., 10,000-piece cap)")
    if evals:
        n = int(evals[-1]["games"])
        best = max(evals, key=lambda r: r["score_mean"])  # same rule as best.pt (best_metric = score)
        rows = [("latest", evals[-1]), ("best", best)]
        out.append(f"  {'':34} {'lines':>8} {'score':>9} {'pieces':>7}  capped   at step")
        for label, r in rows:
            out.append(f"  {label + f' ({n} seeds)':34} {r['lines_mean']:8.1f} {r['score_mean']:9,.0f} "
                       f"{r['pieces_mean']:7.0f}  {int(r['capped'])}/{n}   {int(r['step']):>9,}")
        if evals[-1].get("ref_score_mean") is not None:
            r = evals[-1]
            out.append(f"  {f'heuristic (tuned), same {n} seeds':34} {r['ref_lines_mean']:8.1f} "
                       f"{r['ref_score_mean']:9,.0f} {r['ref_pieces_mean']:7.0f}   <- fairest comparison")
        for name, b in baselines.items():
            out.append(f"  {name + ' (100 seeds)':34} {b['lines']:8.1f} {b['score']:9,.0f} {b['pieces']:7.0f}")
        out.append(f"  {len(evals)} evaluations so far; eval lines trend: "
                   + " ".join(f"{v:.0f}" for v in column(evals, "lines_mean")[-10:]) + "  (last 10)")
    else:
        out.append("  none yet (first one after loop.eval_every_episodes episodes)")

    # --- network ---
    recent = log[-window:]
    vmean = column(recent, "value_mean")
    vmax = column(recent, "value_max")
    if len(vmax) == 0:
        vmax = vmean  # older runs didn't log value_max; the largest episode mean is a lower bound
    loss = column(recent, "loss")
    early_loss = column(log[:window], "loss")
    out.append(f"\nnetwork (last {window} episodes)")
    if len(loss):
        out.append(f"  predicted value: mean {vmean.mean():.2f}, max {vmax.max():.2f}"
                   f"   (ceiling {hard:.0f} = best possible; ~{typical:.0f} = tuned-heuristic-level play)")
        out.append(f"  loss: median {np.median(loss):.4f}, latest {loss[-1]:.4f}   "
                   + (f"(first {window} logged: median {np.median(early_loss):.4f})" if len(early_loss) else ""))
    else:
        out.append("  not learning yet (replay buffer still filling)")

    # --- health ---
    warnings = health_warnings(log, evals, hard, window, patience)
    out.append("\nHEALTH: " + ("OK, no warnings" if not warnings else f"{len(warnings)} WARNING(S)"))
    out += [f"  ! {w}" for w in warnings]
    return "\n".join(out)


# ----------------------------------------------------------------------
# Learning curve
# ----------------------------------------------------------------------
# Reference palette from the dataviz skill: blue = our agent, orange = its
# evaluations; baselines are neutral grays told apart by line style + labels.
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
BASELINE_STYLES = {"random": ":", "heuristic (Lee)": "--", "heuristic (tuned)": "-."}


def plot(run_dir: Path, window: int = 50, baseline_path: Path = BASELINE) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")  # draw to a file, no window (works while training runs)
    import matplotlib.pyplot as plt

    log = read_rows(run_dir / "log.csv")
    evals = read_rows(run_dir / "eval.csv")
    if not log:
        return None
    baselines = load_baselines(baseline_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: lines per training game (faint dots) + moving average, and eval lines.
    steps, lines = column(log, "step"), column(log, "lines")
    ax1.scatter(steps, lines, s=3, color=BLUE, alpha=0.15, linewidths=0)
    if len(lines) >= window:
        avg = np.convolve(lines, np.ones(window) / window, mode="valid")
        ax1.plot(steps[window - 1:], avg, color=BLUE, lw=2, label=f"training games ({window}-episode average)")
    if evals:
        ax1.plot(column(evals, "step"), column(evals, "lines_mean"), "o-", color=ORANGE, lw=2, ms=5,
                 label="evaluation (greedy, selection seeds)")
    ax1.set_ylabel("lines per game")
    ax1.set_title("Lines per game", loc="left", color=INK)

    # Bottom: mean evaluation score.
    if evals:
        ax2.plot(column(evals, "step"), column(evals, "score_mean"), "o-", color=ORANGE, lw=2, ms=5,
                 label="evaluation mean score")
    ax2.set_ylabel("score")
    ax2.set_xlabel("training pieces (steps)")
    ax2.set_title("Evaluation score", loc="left", color=INK)

    # Baselines as horizontal lines, labeled directly at the right edge.
    from matplotlib.ticker import StrMethodFormatter

    for ax, key in ((ax1, "lines"), (ax2, "score")):
        # Headroom above the highest line/point so the legend never covers data.
        top = max([ax.get_ylim()[1]] + [b[key] for b in baselines.values()])
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]), top=top * 1.3)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        for name, style in BASELINE_STYLES.items():
            if name in baselines:
                y = baselines[name][key]
                ax.axhline(y, color=MUTED, ls=style, lw=1.2)
                ax.annotate(name, (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=8, color=MUTED)
        ax.grid(True, color="#e6e5e1", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)  # room for the baseline labels
    path = run_dir / "learning_curve.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="e.g. runs/full")
    parser.add_argument("--window", type=int, default=50, help="episodes per 'recent' window (default 50)")
    parser.add_argument("--patience", type=int, default=5,
                        help="warn if eval lines haven't improved in this many evaluations (default 5)")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    print(report(run_dir, args.window, args.patience))
    png = plot(run_dir, args.window)
    if png:
        print(f"\nlearning curve saved to {png}")


if __name__ == "__main__":
    main()
