# Step 3: learned agent vs baselines

100 games, seeds 10000..10099, cap 10,000 pieces, greedy play.
Checkpoint: `runs\lr_decay\checkpoints\best.pt` (training step 86,944).

| agent | mean lines | median lines | mean score | mean pieces | capped |
|---|---:|---:|---:|---:|---:|
| random | 0.0 | 0.0 | 2 | 21 | 0/100 |
| heuristic (Lee) | 446.4 | 365.5 | 48,869 | 1,157 | 0/100 |
| heuristic (tuned) | 599.4 | 438.5 | 67,036 | 1,537 | 0/100 |
| learned (TD, features) | 791.5 | 563.0 | 97,060 | 2,022 | 0/100 |

Learned vs heuristic (tuned), paired per game: score +30,024 (SE 10,890, +2.8 SE), wins 60/100 (ties 0)
