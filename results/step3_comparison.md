# Step 3: learned agent vs baselines

100 games, seeds 10000..10099, cap 10,000 pieces, greedy play.
Checkpoint: `runs\full\checkpoints\best.pt` (training step 197,404).

| agent | mean lines | median lines | mean score | mean pieces | capped |
|---|---:|---:|---:|---:|---:|
| random | 0.0 | 0.0 | 2 | 21 | 0/100 |
| heuristic (Lee) | 446.4 | 365.5 | 48,869 | 1,157 | 0/100 |
| heuristic (tuned) | 599.4 | 438.5 | 67,036 | 1,537 | 0/100 |
| learned (TD, features) | 455.8 | 361.5 | 50,749 | 1,182 | 0/100 |

Learned vs heuristic (tuned), paired per game: score -16,287 (SE 6,923, -2.4 SE), wins 47/100 (ties 0)
