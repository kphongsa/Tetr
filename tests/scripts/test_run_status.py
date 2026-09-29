"""Status script: health checks on hand-made logs, and a full report + PNG on a fake run."""

import csv
import json

import pytest

from scripts.run_status import (BASELINE, health_warnings, load_baselines, plot, report,
                                value_ceilings)

LEARNER = {"reward_scale": 0.01, "survival_bonus": 0.1, "gamma": 0.99}


def log_rows(losses, value_max=1.0):
    return [{"episode": i, "step": 10 * (i + 1), "loss": l, "value_mean": 0.5, "value_max": value_max}
            for i, l in enumerate(losses)]


def eval_rows(lines):
    return [{"step": 100 * i, "lines_mean": v} for i, v in enumerate(lines)]


def test_value_ceiling_numbers():
    hard, typical = value_ceilings(LEARNER, load_baselines())
    # (0.1 + 0.01 * 0.4 lines/piece * 200 points/line) / (1 - 0.99) = 90
    assert hard == pytest.approx(90.0)
    assert 40 < typical < hard  # the tuned heuristic earns ~44 points/piece -> ~54


def test_healthy_run_has_no_warnings():
    assert health_warnings(log_rows([0.05] * 100), eval_rows([10, 50, 100, 150]), 90.0) == []


def test_nan_loss_is_flagged():
    w = health_warnings(log_rows([0.05] * 10 + [float("nan")]), [], 90.0)
    assert len(w) == 1 and "NAN" in w[0]


def test_loss_jump_is_flagged():
    w = health_warnings(log_rows([0.05] * 50 + [1.0] * 10), [], 90.0, window=50)
    assert any("JUMPED" in x for x in w)


def test_values_above_ceiling_are_flagged():
    w = health_warnings(log_rows([0.05] * 10, value_max=150.0), [], 90.0)
    assert any("TOO HIGH" in x for x in w)


def test_eval_plateau_is_flagged_only_after_patience():
    assert any("NO IMPROVEMENT" in x
               for x in health_warnings([], eval_rows([10, 200, 150, 190, 180, 199, 120]), 90.0, patience=5))
    # A new best inside the last 5 evaluations: no warning.
    assert health_warnings([], eval_rows([10, 200, 150, 190, 180, 250]), 90.0, patience=5) == []


def write_fake_run(tmp_path, slow_episode=None):
    (tmp_path / "config.json").write_text(json.dumps(
        {"name": "fake", "loop": {"total_steps": 1000, "max_episode_steps": 2000}, "learner": LEARNER}))
    cols = ["episode", "step", "seed", "steps", "return", "terminated", "lines", "score", "pieces", "epsilon",
            "loss", "value_mean", "value_max", "target_mean", "steps_per_sec", "elapsed_sec"]
    with open(tmp_path / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for i in range(120):
            w.writerow({"episode": i, "step": 5 * (i + 1), "seed": i, "steps": 5, "return": 100, "terminated": 1,
                        "lines": i // 10, "score": 100, "pieces": 5, "epsilon": 0.5,
                        "loss": "" if i < 3 else 0.05, "value_mean": "" if i < 3 else 1.0,
                        "value_max": "" if i < 3 else 2.0, "target_mean": "" if i < 3 else 1.0,
                        "steps_per_sec": 0.01 if i == slow_episode else 300,
                        "elapsed_sec": 2.0 * (i + 1) + (500 if slow_episode is not None and i >= slow_episode else 0)})
    with open(tmp_path / "eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["step", "episode", "games", "capped", "lines_mean", "score_mean", "pieces_mean",
                               "ref_lines_mean", "ref_score_mean", "ref_pieces_mean"])
        w.writeheader()
        for i, v in enumerate([5, 20, 40]):
            w.writerow({"step": 200 * i, "episode": 40 * i, "games": 20, "capped": 0, "lines_mean": v,
                        "score_mean": 100 * v, "pieces_mean": 3 * v, "ref_lines_mean": 588.6,
                        "ref_score_mean": 65725, "ref_pieces_mean": 1510})


def test_report_and_plot_on_a_fake_run(tmp_path):
    write_fake_run(tmp_path)
    text = report(tmp_path)
    assert "pieces 600 / 1,000" in text and "HEALTH: OK" in text
    assert "heuristic (tuned), same 20 seeds" in text and "heuristic (Lee) (100 seeds)" in text
    assert plot(tmp_path).exists()


def test_baseline_file_has_all_three_baselines():
    assert set(load_baselines(BASELINE)) == {"random", "heuristic (Lee)", "heuristic (tuned)"}


def test_paused_game_is_noted_and_left_out_of_time_left(tmp_path):
    write_fake_run(tmp_path, slow_episode=110)  # 5 steps at 0.01/s = 500 s: "the laptop slept"
    text = report(tmp_path)
    assert "episode 110" in text and "slept" in text
    # Recent rate ignores the pause: 5 pieces per 2 s = 2.5/s, not dragged down by the 500 s.
    assert "2 recently" in text or "3 recently" in text
