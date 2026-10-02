"""build_site: step selection, best-step lookup, and a small end-to-end export."""

import json

from agents.heuristic_agent import HeuristicAgent
from core.evaluate import evaluate
from core.replay import make_replay, save_replay
from games.tetris.env import TetrisEnv
from scripts.build_site import best_step, build, pick_steps


def test_pick_steps():
    steps = list(range(100, 1100, 100))  # 10 evaluations
    assert pick_steps(steps, 20, []) == steps  # asking for more than exist -> all
    assert pick_steps(steps, 3, []) == [100, 500, 1000]  # first, middle-ish, last
    assert pick_steps(steps, 3, [700]) == [100, 500, 700, 1000]  # best always included
    assert pick_steps(steps, 1, []) == [1000]


def make_run(root, name):
    run = root / name
    (run / "replays").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"loop": {"best_metric": "score"}}))
    (run / "eval.csv").write_text("step,score_mean\n10,5.0\n20,9.0\n30,9.0\n")
    for step, seed in [(10, 1), (20, 1), (30, 1)]:
        res = evaluate(TetrisEnv(), lambda s: HeuristicAgent(), [seed], 20 + step, ("lines", "score", "pieces"),
                       record_actions=True)
        g = res["games"][0]
        save_replay(run / "replays" / f"step{step:09d}_seed{seed}.json",
                    make_replay("tetris", seed, g["actions"], step=step, lines=g["lines"], score=g["score"],
                                pieces=g["pieces"], terminated=g["terminated"]))
    return run


def test_build(tmp_path):
    run = make_run(tmp_path / "runs", "r")
    assert best_step(run) == 20  # ties: the first one wins
    out = tmp_path / "site" / "data"
    entries = build(tmp_path / "runs", ["r", "missing"], out, per_run=2)
    assert [e["step"] for e in entries] == [10, 20, 30]  # first + last + best
    assert [e["best"] for e in entries] == [False, True, False]
    index = json.loads((out / "index.json").read_text())["replays"]
    assert index == entries
    doc = json.loads((out / entries[0]["file"]).read_text())
    assert doc["format"] == "frames" and len(doc["frames"]) == entries[0]["pieces"] + 1
