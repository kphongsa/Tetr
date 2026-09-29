"""Compare a trained checkpoint against the step 2 baselines on the evaluation seeds.

Run from the repo root (after training):
    python -m scripts.compare --checkpoint runs/full/checkpoints/best.pt
    python -m scripts.compare --checkpoint runs/full/checkpoints/best.pt --games 20 --no-save   # quick look

Plays the learned agent greedily (epsilon = 0) on the SAME seeds and piece
cap as results/baseline_heuristic.json (seeds 10000..10099, cap 10,000),
then prints one table: random, heuristic (Lee), heuristic (tuned), learned.

The heuristic rows are read from the baseline file instead of replayed:
those agents are deterministic, so on the same seeds and cap they would
produce exactly the same games (and it saves ~15 minutes). The file's git
commit is recorded in the output, so this stays traceable.

Paired comparison: because both agents play the SAME 100 piece sequences,
we can compare them game by game (learned score - tuned score on seed s).
That removes most of the luck of the draw, so it detects smaller real
differences than comparing two averages would.

Writes results/step3_comparison.json (all numbers + per-game results) and
results/step3_comparison.md (the table).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.evaluate import evaluate, git_commit
from games.tetris.env import TetrisEnv
from scripts.train import BASELINE, INFO_KEYS, load_eval_agent

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def paired(learned_games: list[dict], baseline_games: list[dict], key: str) -> dict:
    """Game-by-game difference learned - baseline on the same seeds."""
    base = {g["seed"]: g for g in baseline_games}
    diffs = np.array([g[key] - base[g["seed"]][key] for g in learned_games], dtype=np.float64)
    n = len(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        # Standard error of the mean difference; |mean| > ~2 SE is unlikely to be luck.
        "se": float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "wins": int((diffs > 0).sum()),
        "ties": int((diffs == 0).sum()),
        "losses": int((diffs < 0).sum()),
        "games": n,
    }


def table(summaries: dict[str, dict]) -> str:
    """Markdown table in the same shape as CLAUDE.md's baseline table."""
    lines = ["| agent | mean lines | median lines | mean score | mean pieces | capped |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['lines']['mean']:,.1f} | {s['lines']['median']:,.1f} | "
                     f"{s['score']['mean']:,.0f} | {s['pieces']['mean']:,.0f} | {s['capped']}/{s['games']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=None, help="use only the first N eval seeds (default: all)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    baseline = json.loads(BASELINE.read_text())
    seeds = baseline["seeds"][: args.games] if args.games else baseline["seeds"]
    cap = baseline["max_pieces"]
    subset = len(seeds) < len(baseline["seeds"])

    agent, ck = load_eval_agent(args.checkpoint)
    progress = ck["progress"]
    print(f"checkpoint {args.checkpoint}: step {progress['step']}, episode {progress['episode']}, "
          f"best training-eval score {progress['best']}")
    print(f"playing {len(seeds)} games, seeds {seeds[0]}..{seeds[-1]}, cap {cap} pieces ...", flush=True)
    learned = evaluate(TetrisEnv(), lambda seed: agent, seeds, cap, INFO_KEYS)

    tuned_games = [g for g in baseline["games"] if g["seed"] in set(seeds)]
    summaries = {}
    if not subset:
        # Stored summaries cover all 100 seeds; only valid for the full set.
        summaries["random"] = baseline["other_summaries"]["random"]
        summaries["heuristic (Lee)"] = baseline["other_summaries"]["heuristic (Lee)"]
        summaries[baseline["agent"]] = baseline["summary"]
    else:
        # Subset: we only have per-game rows for the tuned heuristic.
        from core.evaluate import summarize

        s: dict = {k: summarize([g[k] for g in tuned_games]) for k in INFO_KEYS}
        s.update(games=len(tuned_games), capped=sum(g["truncated"] for g in tuned_games))
        summaries[baseline["agent"]] = s
    summaries["learned (TD, features)"] = learned["summary"]

    md = table(summaries)
    pairs = {k: paired(learned["games"], tuned_games, k) for k in ("score", "lines")}
    p = pairs["score"]
    verdict = (f"Learned vs {baseline['agent']}, paired per game: score {p['mean_diff']:+,.0f} "
               f"(SE {p['se']:,.0f}, {p['mean_diff'] / p['se']:+.1f} SE), "
               f"wins {p['wins']}/{p['games']} (ties {p['ties']})")
    print("\n" + md + "\n\n" + verdict)
    s = learned["summary"]
    print(f"\nlearned agent: {s['steps_per_second']:.0f} pieces/s, {s['seconds'] / 60:.1f} min")
    if s["capped"] > s["games"] / 2:
        print(f"WARNING: {s['capped']}/{s['games']} games hit the {cap}-piece cap; the cap is limiting the comparison.")

    if args.no_save or subset:
        if subset:
            print("\n(subset of seeds: not saved; run without --games for the official comparison)")
        return
    payload = {
        "description": "Step 3 learned agent vs step 2 baselines on the evaluation seeds.",
        "seeds": seeds,
        "max_pieces": cap,
        "run": ck["config"]["name"],
        "checkpoint": {"path": str(args.checkpoint), "progress": progress, "config": ck["config"]},
        "git": git_commit(),
        "baseline_git": baseline["git"],
        "summaries": summaries,
        "paired_vs_tuned": pairs,
        "learned_games": learned["games"],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "step3_comparison.json").write_text(json.dumps(payload, indent=2))
    (RESULTS / "step3_comparison.md").write_text(
        f"# Step 3: learned agent vs baselines\n\n"
        f"{len(seeds)} games, seeds {seeds[0]}..{seeds[-1]}, cap {cap:,} pieces, greedy play.\n"
        f"Checkpoint: `{args.checkpoint}` (training step {progress['step']:,}).\n\n{md}\n\n{verdict}\n")
    print("\nsaved results/step3_comparison.json and results/step3_comparison.md")


if __name__ == "__main__":
    main()
