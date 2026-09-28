"""Rule-based Tetris player: rate every afterstate with hand-picked weights, pick the best.

For each legal placement we build its afterstate (the board after the piece
locks and lines clear), turn it into 4 feature numbers, and score it as

    score = w_height * aggregate_height + w_lines * lines_cleared
          + w_holes * holes + w_bump * bumpiness

then play the highest-scoring placement. No learning and no randomness: the
same board and piece always give the same move.

Where the default weights come from
-----------------------------------
Yiyuan Lee, "Tetris AI - The (Near) Perfect Bot" (blog post, 2013). He tuned
these four weights with a genetic algorithm (a search that keeps and mixes the
best-scoring weight sets over many generations). Signs make sense: height,
holes and bumpiness are bad (negative), clearing lines is good (positive).
Caveat: his bot also looked at the NEXT piece; ours only sees the current
one, so expect somewhat weaker play than his reported numbers.

This agent is Tetris-specific (it imports games.tetris.features), which is
allowed: agents may depend on games, never the other way round. It still
only talks to the game through obs/info from the env.
"""

from __future__ import annotations

import numpy as np

from games.tetris.features import FEATURE_NAMES, afterstates, board_features

# Order must match FEATURE_NAMES: aggregate_height, lines_cleared, holes, bumpiness.
LEE_WEIGHTS: tuple[float, ...] = (-0.510066, 0.760666, -0.35663, -0.184483)


class HeuristicAgent:
    def __init__(self, weights=LEE_WEIGHTS) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"need {len(FEATURE_NAMES)} weights for {FEATURE_NAMES}")

    def score_placements(self, obs, info) -> list[tuple[int, float]]:
        """(action, score) for every legal placement, in the env's order (sorted by action).

        Split out from act() so scripts and tests can see WHY a move was chosen.
        """
        return [
            (a.placement.action, float(self.weights @ board_features(a.grid, a.lines_cleared)))
            for a in afterstates(obs["board"], info["placements"])
        ]

    def act(self, obs, info) -> int:
        scored = self.score_placements(obs, info)
        # Tie-breaking: placements arrive sorted by action id, and we only
        # replace the best on a STRICTLY higher score, so exact ties go to the
        # lowest action id. Deterministic, and independent of any rng.
        best_action, best_score = scored[0]
        for action, score in scored[1:]:
            if score > best_score:
                best_action, best_score = action, score
        return best_action
