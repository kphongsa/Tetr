"""Replays: a game recorded as seed + action list (+ metadata), replayed by re-simulating.

CLAUDE.md rule 4. Because a game is fully determined by its seed and the
actions played, we don't need to store any boards: re-running the same
actions from the same seed reproduces the game exactly. A 1,000-piece game
is a few KB of JSON.

File format (JSON):
    {"format": "replay", "version": 1, "game": "tetris", "seed": 10000,
     "actions": [...], "metadata": {anything: training step, score, ...}}

Game-agnostic: replay_episode() only uses reset/step, so the step 4 viewer
(or anything else) can re-simulate any game that follows the env interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from core.evaluate import Env

REPLAY_VERSION = 1


def make_replay(game: str, seed: int, actions: Sequence[int], **metadata: Any) -> dict:
    return {
        "format": "replay",
        "version": REPLAY_VERSION,
        "game": game,
        "seed": int(seed),
        "actions": [int(a) for a in actions],
        "metadata": metadata,
    }


def save_replay(path: Path, replay: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact, one line: long games have thousands of actions, and an
    # indented file would put each one on its own line.
    path.write_text(json.dumps(replay))


def load_replay(path: Path) -> dict:
    replay = json.loads(Path(path).read_text())
    if replay.get("format") != "replay" or replay.get("version") != REPLAY_VERSION:
        raise ValueError(f"{path} is not a version-{REPLAY_VERSION} replay file")
    return replay


def replay_episode(env: Env, replay: dict, info_keys: Sequence[str] = ()) -> dict:
    """Re-simulate a replay. Returns the final info values plus how the game ended.

    Raises if the recorded actions don't fit the game (the game ends before
    the actions run out, or an action is illegal): that would mean the
    replay and the game code have drifted apart.
    """
    obs, info = env.reset(seed=replay["seed"])
    terminated = truncated = False
    for i, action in enumerate(replay["actions"]):
        if terminated or truncated:
            raise ValueError(f"game ended after {i} actions, replay has {len(replay['actions'])}")
        obs, _, terminated, truncated, info = env.step(action)
    return {"terminated": terminated, "truncated": truncated, **{k: info[k] for k in info_keys}}
