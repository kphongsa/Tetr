"""Replay style summary: re-simulates a real game and the numbers are consistent with it."""

from agents.heuristic_agent import HeuristicAgent
from core.evaluate import evaluate
from core.replay import make_replay
from games.tetris.env import TetrisEnv
from scripts.replay_style import format_table, style


def test_style_of_a_heuristic_game():
    res = evaluate(TetrisEnv(), lambda s: HeuristicAgent(), [3], 300, ("lines", "score"), record_actions=True)
    game = res["games"][0]
    replay = make_replay("tetris", 3, game["actions"], step=0)
    s = style(replay)
    assert s["pieces"] == len(game["actions"]) and s["step"] == 0
    assert s["lines"] == game["lines"]
    assert 0 < s["avg_stack_height"] <= s["max_stack_height"] <= 22
    assert abs(s["share_singles"] + s["share_doubles"] + s["share_triples"] + s["share_tetrises"] - 1) < 1e-9
    assert style(replay, max_pieces=50)["pieces"] == 50
    assert "pcs/clear" in format_table([s])
