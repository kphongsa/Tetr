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
- **Line clear**: a completely filled row disappears and everything above it
  moves down. Clearing 1/2/3/4 rows with one piece is a single/double/triple/
  **Tetris** (worth 100/300/500/800 here). Big clears are worth more per line,
  so a good player (or agent) saves up for Tetrises.
- **Stack**: the pile of locked blocks. **Well**: a one-column-wide gap kept
  open on purpose so a vertical I can drop in for a Tetris.
- **Active piece**: the piece currently falling and under control. It isn't
  written into the board array until it locks.
- **Spawn position**: where a new piece appears. Ours puts the bounding box's
  top-left at row 0, centred. That leaves room to rotate any piece right away.

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
- **Fixed action space**: numbering every *possible* action once (for us 40 =
  4 rotations × 10 columns), even though only some are legal at any moment.
  Neural networks need a fixed number of outputs, and the action mask marks
  which of those outputs are currently allowed.
- **Tuck / spin**: sliding a piece sideways under an overhang after it has
  dropped, or rotating it into a tight slot. Our placements can't do these
  (they only rotate, shift, then drop). Button-press actions in step 5 will.
- **Random baseline**: an agent that picks legal moves at random. It's the
  lowest bar: any strategy worth anything must beat it by a lot.

## Engineering terms

- **Benchmark**: a repeatable speed/quality measurement (same seeds every
  run), so changes can be compared fairly.
- **Profiling**: running code under a tool (`cProfile`) that measures where
  the time actually goes. Optimize the top item, not a guess.
- **Pieces per second**: our speed metric. RL needs millions of game steps,
  so this determines how long training takes.

## Step 2: rule-based baseline

- **Afterstate**: the board right after a placement locks and lines clear,
  before the next (random) piece appears. Picking a move = picking the
  afterstate you like best; step 3's network will learn to rate these.
- **Feature**: a single number summarizing something about a board (e.g. how
  many holes). Hand-picked features turn a 220-cell board into a few numbers
  a simple formula can reason about.
- **Aggregate height**: sum of all column heights. High = close to game over.
- **Hole**: an empty cell with a block above it in the same column. Can't be
  filled until the blocks above are cleared.
- **Bumpiness**: sum of height differences between neighbouring columns. A
  jagged surface fits pieces badly.
- **Copy vs peek**: two ways to look ahead without changing the real game.
  Copy = simulate on a copy of the board; peek = change the real board and
  undo. We copy: the board is tiny and there's nothing to undo wrongly.
