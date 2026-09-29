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
- **Heuristic (rule-based) agent**: a player whose decisions come from a
  hand-written formula, not learning. Ours scores each afterstate as a
  weighted sum of features and plays the best one.
- **Weights**: the multipliers in that sum. Their signs say "good" (+) or
  "bad" (−); their sizes say how much each feature matters relative to others.
- **Genetic algorithm**: a search method inspired by evolution. Keep the best
  weight sets, mix and mutate them, repeat. Used to find the published
  weights we start from.
- **Tie-breaking**: the rule for choosing between equally scored moves. Ours
  (lowest action id) is deterministic, so the same game replays identically.
- **tottime vs cumtime (cProfile)**: tottime = time spent inside a function
  itself; cumtime = including everything it calls. Look at cumtime to find
  which *part* of the program is slow, tottime to find the exact hot line.
- **NumPy call overhead**: every NumPy call costs ~1–5 µs even on a tiny
  array. On a 22×10 board that overhead, not the arithmetic, is most of the cost.

## Step 2c–2d: evaluation and tuning

- **Evaluation harness**: code that plays an agent on a fixed list of games
  and reports summary statistics. Same agent + same seeds = same numbers, so
  two agents can be compared fairly on identical games.
- **Truncation cap / step limit**: ending a game early after N steps (pieces)
  and marking it *truncated*, not lost. Keeps evaluations finite, but if most
  games hit the cap, strong agents all look the same.
- **Mean / median / standard deviation (std)**: the average; the middle value
  (less swayed by a few huge games); and how spread out the results are.
  Tetris results are very spread out (std ≈ mean).
- **Standard error**: roughly how far the measured mean could be from the
  "true" mean because we only played n games: std / √n. Differences smaller
  than ~2 standard errors may just be luck.
- **Provenance**: recording where a result came from (git commit, seeds,
  settings) so it can be reproduced and trusted later.
- **Return**: the total reward collected over one episode. It's what RL
  agents try to maximize.
- **Cross-entropy method (CEM)**: a search method. Keep a bell-curve "cloud"
  of candidate parameter vectors; each round sample some, score them, keep
  the best few (the *elite*), and move/shrink the cloud to fit the elite.
- **Elite**: the top fraction of candidates in a CEM round; the next round's
  cloud is fitted to them.
- **Fitness**: the single number a search method tries to maximize for a
  candidate (here: mean score over a few games).
- **Population**: how many candidates are sampled per round.
- **Search vs. RL**: search (like CEM) only sees each candidate's final total
  score and keeps what worked. RL learns from *which moves* in a game led to
  reward, which scales to huge models (neural networks) where search can't.
- **Gradient**: the direction in which changing the parameters improves a
  score fastest. Deep learning follows gradients; CEM doesn't use them.
- **Overfitting**: doing well only on the examples you trained/tuned on
  (e.g. particular piece sequences) and worse on new ones. Prevented by
  measuring on separate, never-seen examples.
- **Train / test split (tuning seeds vs. evaluation seeds)**: tune on one set
  of games, report results on a disjoint set. Only the second number is an
  honest estimate of how good the agent is.
- **Winner's curse**: the candidate that scored best on a few noisy games is
  probably partly lucky, so its score overstates its true quality. That's
  why we keep CEM's *average of the elite* rather than the single best one.
- **Normalization (unit length)**: rescaling a weight vector to length 1.
  For "pick the highest weighted sum", only the direction of the weights
  matters, so this removes a meaningless degree of freedom from the search.
- **CPU throttling**: a laptop slowing its processor after a few seconds of
  heavy load (heat/power limits). Short benchmarks then overstate how fast
  long runs will be.

## Step 3: first learning agent

- **Neural network**: a function built from layers of weighted sums with
  simple bends (like ReLU) in between. Its weights are *learned* from data
  rather than hand-set, and it can represent curved, interacting judgments.
- **MLP (multi-layer perceptron)**: the simplest neural network: a stack of
  fully-connected (Linear) layers. Ours: 4 features in → 64 → 64 → 1 number out.
- **Linear layer**: many weighted sums of the inputs at once (one per output),
  plus a constant (the *bias*). The heuristic agent was one Linear layer
  with one output.
- **ReLU**: max(0, x), applied between layers. Without it, stacked Linear
  layers collapse into a single weighted sum and gain nothing.
- **Parameters**: all the numbers inside the network (weights and biases)
  that training adjusts. Our MLP has ~4,500.
- **Value function V**: the network's estimate of "how much future reward
  will I collect from this position if I keep playing well". The agent picks
  the placement whose afterstate has the best value.
- **Q(s, a)**: the value of taking action a in state s = immediate reward +
  gamma × value of the afterstate. We compute it for every legal placement.
- **Epsilon-greedy exploration**: with probability epsilon make a random
  legal move, otherwise the best-rated one. Without exploring, an agent never
  tries moves it currently rates badly, so it can't discover they're good.
- **Input scaling / normalization**: dividing inputs by rough typical sizes
  so they're around 0..1. Networks learn poorly from inputs in the hundreds.
- **float32**: the 32-bit decimal-number type PyTorch uses by default: half
  the memory of NumPy's float64 and plenty precise for neural networks.
- **PyTorch**: the deep-learning library we use: tensors (NumPy-like arrays)
  plus automatic gradient computation and optimizers.
- **Temporal-difference (TD) learning**: learn a guess from a slightly better
  guess made one step later, instead of waiting for the game to end. After
  each move, nudge V(this afterstate) toward "reward of the next move +
  gamma × V(next afterstate)".
- **TD target**: that "slightly better guess": the number the network is
  trained to output for a given afterstate on this update.
- **Bootstrapping**: using the network's own estimate of the future (V of the
  next position) inside its training target. Done when the game continues,
  including when it was only *truncated*; not done at a real game over,
  where the future is worth nothing (or the game-over penalty).
- **Discount factor (gamma, γ)**: how much a reward one step later counts
  compared to now (0.99 → about a 100-step horizon). Keeps values finite.
- **Transition**: one move's record in memory: (afterstate chosen, did the
  game end, all afterstates available on the next piece).
- **Replay buffer**: a big ring of recent transitions; training samples random
  batches from it so consecutive, near-identical moves don't dominate.
- **Target network**: a frozen copy of the network used only to compute TD
  targets, refreshed every N updates, so the network isn't chasing a target
  that moves at every step.
- **Loss**: one number measuring how wrong predictions are on a batch;
  training lowers it. **Huber loss**: squared error for small errors, linear
  for big ones, so a few surprising samples can't cause huge updates.
- **Gradient step / backpropagation**: `loss.backward()` computes how the
  loss changes with every weight (backpropagation); the optimizer then moves
  each weight slightly in the direction that lowers the loss.
- **Optimizer (Adam)**: the rule for turning gradients into weight changes.
  Adam adapts each weight's step size to how noisy its gradient has been.
- **Learning rate**: the overall size of each weight change. Too high →
  unstable/diverges; too low → learns very slowly.
- **Batch size**: how many transitions are averaged per update. Bigger =
  smoother but slower updates.
- **learning_starts**: wait until the buffer has some variety before training.
- **Reward shaping**: adding reward the game doesn't give (our survival bonus
  and game-over penalty) to make learning easier. Risk: the agent optimizes
  the shaped reward, not the real goal. So we always *evaluate* on real score.
- **Reward scaling**: dividing rewards (800 → 8) so values and losses stay
  in a range networks handle well.
- **Why the TD loss can go UP while the agent improves**: the loss is in
  value units. As the agent survives longer, the true values grow (from ~0
  to ~10+), so the same *relative* error is a bigger absolute error, and the
  targets themselves keep shifting as the policy changes. Judge progress by
  lines/score, not by the loss alone. The loss is mainly a crash detector
  (explosions, NaN).
- **Irreducible error**: error no network can remove because the outcome is
  random given what it sees (e.g. whether the next piece can spawn).
- **TensorBoard**: a local web page that plots logged numbers live.
  `tensorboard --logdir runs`, then open http://localhost:6006.
- **Hyperparameter**: a setting chosen by us, not learned (learning rate,
  gamma, batch size...). All of ours live in the run config JSON.
- **CPU threads (torch_threads)**: how many CPU cores PyTorch uses for one
  calculation. For tiny networks more threads help only a little, because
  coordinating threads has its own overhead.
- **Checkpoint**: a file with everything needed to continue training later:
  network weights, target network, optimizer state, counters, random-number
  generator states, config. *latest* = most recent; *best* = the one that
  scored highest in evaluation (what we'd actually use to play).
- **Optimizer state**: Adam's running averages for every weight. Without
  them, the first updates after resuming would be sized wrongly.
- **RNG state**: the exact internal position of a random number generator.
  Saving and restoring it makes the "random" choices after resuming the same
  ones it would have made anyway.
- **Atomic write**: save to a temporary file, then rename it over the real
  one. A crash mid-save leaves the old file intact rather than half-written.
- **Resume**: continue a stopped run from its latest checkpoint. Our replay
  buffer isn't saved (too big), so it refills for `learning_starts` steps
  before learning continues. Resumed runs are therefore close to, but not
  bit-identical with, uninterrupted ones.
- **Greedy evaluation**: playing with epsilon = 0 (always the best-rated
  move) to measure what the agent has actually learned, without the noise of
  exploration moves.
- **Held-out evaluation seeds**: the fixed games (10000..10099) we measure on
  and never train on, so a good result can't come from memorizing them.
- **Replay file**: seed + actions + metadata as JSON. Re-simulating the
  actions from the seed reproduces the game exactly. We save the same two
  eval seeds at every evaluation, so you can watch how the agent plays the
  *same* piece sequence as training progresses.

## Step 3d: the real training run
- **Learning curve**: a plot of performance (lines, score) against training
  time. It's the main way to see whether the agent is still improving,
  has leveled off, or is getting worse.
- **Moving average**: the average of the last N values, recomputed at each
  point. Single training games are very noisy; a 50-game average shows the trend.
- **Value ceiling**: the largest value a correct network could ever predict.
  Here it's (best reward per piece) / (1 − gamma) = (0.1 + 0.01 × 0.4 × 200) /
  0.01 = 90. A prediction above it is certainly wrong.
- **Overestimation**: TD targets use a *max* over the network's own guesses,
  so random errors upward get picked more often than errors downward. The
  guesses can then feed on themselves and climb without limit, which is why
  the status script watches predicted values against the ceiling.
- **Divergence**: training blowing up: the loss or the values run off to huge
  numbers (or nan = "not a number"). Once that happens, nothing after it is
  meaningful. The run has to be restarted with safer settings.
- **Plateau**: performance stops improving for a long stretch. Near the end of
  a run that's normal; early, it can mean the network is too small, the
  inputs don't carry enough information, or learning has stalled.
- **Catastrophic forgetting**: a network getting *worse* at something it
  already did well, because newer training data pushed its weights elsewhere.
  It shows up as evaluation scores that rise and then fall. We keep best.pt
  for this reason.
