# Glossary / learning notes

Plain-language definitions, added as terms come up. Newest at the bottom.

## Tetris / game-engine terms

- **Tetromino**: a shape made of 4 squares. Tetris has 7 of them: I, O, T, S, Z, J, L.
- **Engine**: the pure game rules (board, pieces, moves, scoring) with no drawing
  and no learning code. Everything else is built on top of it.
- **Bounding box**: the smallest square that contains a piece in every rotation
  (4×4 for I, 2×2 for O, 3×3 for the rest). A piece's position is the box's
  top-left corner; its cells are offsets inside the box.
- **Rotation system**: the rules for how a piece turns. Ours spins the shape
  inside its bounding box and nothing else.
- **Wall kick**: when a rotation would overlap a wall or block, some rotation
  systems try nudging the piece a square or two to make it fit. We have none yet.
- **SRS (Super Rotation System)**: the rotation system used by modern Tetris,
  with a specific table of wall kicks. Planned for step 5.
- **Randomizer**: decides the next piece. Ours is uniform (each piece has a 1/7
  chance). Modern Tetris uses a **7-bag**: shuffle all 7 pieces, deal them out,
  repeat, so you never wait long for any piece.
- **Collision**: a piece overlapping a wall, the floor, or a locked block. Every
  move is checked with the same test, `Board.fits`.
- **Lock**: the moment a falling piece stops and becomes part of the board.
- **Hidden rows / spawn area**: 2 rows above the visible board where new
  pieces appear. Game over happens when a new piece can't appear there.
- **Hard drop**: move the piece straight down as far as it can go, then lock it.
  **Soft drop**: move it down by one row.

## Reproducibility

- **Seed**: a number that starts a random number generator. The same seed gives
  the same sequence of "random" numbers, so the same game.
- **RNG (random number generator)**: here a `numpy.random.Generator`. We pass
  one explicitly into the game instead of using global randomness, so nothing
  else can change the game's random sequence.

## RL terms (from the step-1 plan)

- **Environment (env)**: the game wrapped in a standard interface an agent
  can play: `reset()` starts a game, `step(action)` plays one move.
- **Gymnasium**: the standard Python library interface for RL environments.
  We copy its method signatures without depending on the package.
- **Observation (obs)**: what the agent sees after each step (the board and
  current piece).
- **Action**: one choice the agent makes. For us, one placement of the current piece.
- **Reward**: the number the environment gives back after each action. Here,
  the points scored by that placement. RL agents learn to maximize the total.
- **Episode**: one full game, from `reset()` to game over.
- **terminated / truncated**: the two ways an episode ends. *Terminated* means
  the game really ended (game over). *Truncated* means we cut it short, e.g. a
  piece limit. The difference matters for learning: after truncation the
  game could have continued, so the agent shouldn't treat it as a loss.
- **Action mask**: an array of True/False, one per possible action, saying
  which are legal right now. Lets an agent with a fixed-size output (like a
  neural network) ignore illegal choices.
- **Placement action**: instead of pressing buttons, the agent picks where the
  piece ends up (rotation + column). Far fewer decisions per piece, so
  learning is much easier.
