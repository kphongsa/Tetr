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

Step 4 adds two things on top:
    verify_replay()   re-simulate and check the result against the numbers
                      recorded in metadata (catches engine changes that would
                      silently make old replays play out differently)
    export_frames()   re-simulate and turn every step into a "frame" (a plain
                      dict) for viewers. WHAT goes into a frame is
                      game-specific, so the game supplies a FrameRecorder;
                      this module only drives the loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from core.evaluate import Env, git_commit

REPLAY_VERSION = 1


def make_replay(game: str, seed: int, actions: Sequence[int], **metadata: Any) -> dict:
    # Record which code produced the game, so if a replay ever stops
    # verifying we can diff the engine between that commit and now.
    # setdefault: callers that already pass git (the training loop) win.
    metadata.setdefault("git", git_commit()["commit"])
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
    final = _simulate(env, replay)
    return {"terminated": final["terminated"], "truncated": final["truncated"],
            **{k: final["info"][k] for k in info_keys}}


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
def verify_replay(env: Env, replay: dict) -> list[str]:
    """Re-simulate and compare against the recorded metadata. Returns problems ([] = OK).

    Which numbers get compared is decided generically: every metadata key
    that the game's final `info` also has (for Tetris: score, lines,
    pieces), plus whether the game ended by losing ("terminated").
    Metadata-only keys like the training step are ignored.
    """
    try:
        final = _simulate(env, replay)
    except ValueError as e:  # ran out of game, or an illegal action
        return [str(e)]
    meta = replay["metadata"]
    problems = [f"{k}: recorded {meta[k]!r}, re-simulated {final['info'][k]!r}"
                for k in sorted(meta) if k in final["info"] and meta[k] != final["info"][k]]
    if "terminated" in meta and meta["terminated"] != final["terminated"]:
        problems.append(f"terminated: recorded {meta['terminated']}, re-simulated {final['terminated']}")
    return problems


def _simulate(env: Env, replay: dict) -> dict:
    """Play the actions from the seed; return the final info and how the game ended."""
    obs, info = env.reset(seed=replay["seed"])
    terminated = truncated = False
    for i, action in enumerate(replay["actions"]):
        if terminated or truncated:
            raise ValueError(f"game ended after {i} actions, replay has {len(replay['actions'])}")
        obs, _, terminated, truncated, info = env.step(action)
    return {"info": info, "terminated": terminated, "truncated": truncated}


# ----------------------------------------------------------------------
# Frames export (for viewers: terminal, GIF, web)
# ----------------------------------------------------------------------
FRAMES_VERSION = 1


class FrameRecorder(Protocol):
    """Game-specific: turns the env's state into one JSON-ready dict per step.

    start() is called once after reset (frame 0 = the empty starting
    position), step() after every action. A recorder may keep state between
    calls (e.g. Tetris remembers which piece was about to be placed).
    header() returns constants every viewer needs (board size, colors, ...),
    written once per file instead of once per frame.
    """

    def header(self) -> dict: ...
    def start(self, env: Any, obs: Any, info: dict) -> dict: ...
    def step(self, env: Any, action: int, obs: Any, info: dict) -> dict: ...


def export_frames(env: Env, replay: dict, recorder: FrameRecorder) -> dict:
    """Re-simulate a replay and return a frames document (JSON-ready dict).

    The replay is verified on the way (same check as verify_replay): a
    viewer must never show a game that differs from the one recorded, so a
    mismatch raises instead of exporting.

    Layout:
        {"format": "frames", "version": 1, "game": ..., "seed": ...,
         "metadata": {...the replay's metadata...},
         "exported_git": "<commit that did the export>",
         "header": {...recorder.header()...},
         "frames": [frame 0, frame 1, ...]}   # len = actions + 1
    """
    obs, info = env.reset(seed=replay["seed"])
    frames = [recorder.start(env, obs, info)]
    terminated = truncated = False
    for i, action in enumerate(replay["actions"]):
        if terminated or truncated:
            raise ValueError(f"game ended after {i} actions, replay has {len(replay['actions'])}")
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(recorder.step(env, action, obs, info))
    meta = replay["metadata"]
    bad = [k for k in meta if k in info and meta[k] != info[k]]
    if bad or ("terminated" in meta and meta["terminated"] != terminated):
        raise ValueError(f"replay does not re-simulate to its recorded result (mismatch: {bad or ['terminated']})")
    return {
        "format": "frames",
        "version": FRAMES_VERSION,
        "game": replay["game"],
        "seed": replay["seed"],
        "metadata": meta,
        "exported_git": git_commit()["commit"],
        "header": recorder.header(),
        "frames": frames,
    }
