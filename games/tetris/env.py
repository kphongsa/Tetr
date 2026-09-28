"""Gymnasium-style environment: one step = place one piece.

This is the ONLY thing training code should touch. Interface (same method
signatures as the Gymnasium library, without depending on it):

    obs, info = env.reset(seed=None)
    obs, reward, terminated, truncated, info = env.step(action)

obs      {"board": uint8 (22, 10) array of 0/1, "piece": int 1..7}
         0/1 instead of piece ids: which piece left a block is irrelevant
         to play, and a smaller input is easier for a network to learn from.
reward   points scored by this placement (0, 100, 300, 500 or 800)
terminated  True when the game is over (next piece couldn't spawn)
truncated   True when max_pieces was reached (game was cut short, not lost)
info     action_mask   bool (40,) array, True = legal action right now
         legal_actions sorted list of legal action ids
         placements    list[Placement] (final cells of each legal action;
                       the rule-based agent in step 2 will score these)
         lines_cleared, score, lines, pieces, seed

Rule 2 of CLAUDE.md (variable legal actions): the set of legal actions
changes every step, so agents must read `action_mask` / `legal_actions`
from the latest info rather than assuming all 40 actions are valid.
"""

from __future__ import annotations

import numpy as np

from games.tetris.engine import TetrisEngine
from games.tetris.placements import NUM_ACTIONS, Placement, apply_placement, legal_placements
from games.tetris.render_text import render_game


class TetrisEnv:
    num_actions = NUM_ACTIONS

    def __init__(self, max_pieces: int | None = None) -> None:
        self.max_pieces = max_pieces
        self.engine: TetrisEngine | None = None
        self.seed: int | None = None
        self._placements: dict[int, Placement] = {}

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Start a new game.

        Every game gets a concrete integer seed, even if you pass None (we
        draw one from the OS). It's reported in info["seed"], so ANY game,
        including an unseeded one, can be recorded and replayed exactly.
        """
        if seed is None:
            # SeedSequence's .entropy is typed loosely (int | Sequence | None)
            # because it echoes whatever you passed in; with no argument it's
            # always a fresh int drawn from the OS. The assert tells mypy so.
            entropy = np.random.SeedSequence().entropy
            assert isinstance(entropy, int)
            seed = entropy % (2**63)
        self.seed = int(seed)
        self.engine = TetrisEngine(np.random.default_rng(self.seed))
        return self._obs(), self._info(lines_cleared=0)

    def step(self, action: int) -> tuple[dict, int, bool, bool, dict]:
        if self.engine is None:
            raise RuntimeError("call reset() before step()")
        if self.engine.game_over:
            raise RuntimeError("episode is over; call reset()")
        placement = self._placements.get(int(action))
        if placement is None:
            # Fail loudly: an illegal action is an agent bug (it ignored the
            # mask), and silently substituting a legal move would hide it.
            raise ValueError(f"illegal action {action}; legal: {sorted(self._placements)}")

        score_before = self.engine.score
        lines_cleared = apply_placement(self.engine, placement)
        reward = self.engine.score - score_before

        terminated = self.engine.game_over
        truncated = (
            not terminated
            and self.max_pieces is not None
            and self.engine.pieces_placed >= self.max_pieces
        )
        return self._obs(), reward, terminated, truncated, self._info(lines_cleared)

    def render(self, show_hidden: bool = True) -> str:
        # Hidden rows shown by default: that's where games are lost, so a
        # game-over board looks wrong without them.
        return render_game(self._game, show_hidden=show_hidden)

    # ------------------------------------------------------------------
    @property
    def _game(self) -> TetrisEngine:
        """The engine, guaranteed non-None (i.e. reset() has been called).

        self.engine is Optional because it doesn't exist before the first
        reset(). Going through this property gives a clear error instead of
        an AttributeError on None, and lets the type checker know it's safe.
        """
        if self.engine is None:
            raise RuntimeError("call reset() first")
        return self.engine

    def _obs(self) -> dict:
        return {
            "board": (self._game.board.grid != 0).astype(np.uint8),
            "piece": self._game.piece,
        }

    def _info(self, lines_cleared: int) -> dict:
        # Computing legal placements here (once per step) serves both the
        # mask for the agent and the lookup table step() uses next time.
        placements = legal_placements(self._game)
        self._placements = {p.action: p for p in placements}
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[list(self._placements)] = True
        return {
            "action_mask": mask,
            "legal_actions": [p.action for p in placements],
            "placements": placements,
            "lines_cleared": lines_cleared,
            "score": self._game.score,
            "lines": self._game.lines,
            "pieces": self._game.pieces_placed,
            "seed": self.seed,
        }
