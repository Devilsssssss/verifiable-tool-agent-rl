# Return-v11 Candidate Results

This report evaluates the prespecified return-v11 equal-view SFT intervention
against the matched position-v10 adapters. Both final adapters were evaluated
with greedy decoding on the frozen 120-task return holdout and on the existing
120-task position confirmation set. All candidate rollouts were replayed with
unchanged scores.

## Frozen return holdout

| seed | matched v10 | return-v11 | gained | lost | paired bootstrap 95% interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 58/120 (48.3%) | 56/120 (46.7%) | 6 | 8 | [-7.5, 4.2] pp |
| 46 | 54/120 (45.0%) | 81/120 (67.5%) | 31 | 4 | [13.3, 30.8] pp |

The seed-45 result is a small decline. Seed 46 improves by 22.5 percentage
points, but the two training seeds disagree in direction, so this is not a
stable training conclusion.

The seed-46 gain is concentrated in the difficult expired-return cases: its
strict success is 29/30 for target position 1 but 0/30 for position 2. The
position-2 failure is therefore still unresolved. Seed 45 reaches 30/30 and
26/30 on eligible returns by position, but 0/60 on expired returns.

## Position confirmation regression

| seed | matched v10 | return-v11 | gained | lost | paired bootstrap 95% interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 105/120 (87.5%) | 96/120 (80.0%) | 2 | 11 | [-13.3, -1.7] pp |
| 46 | 101/120 (84.2%) | 96/120 (80.0%) | 1 | 6 | [-8.3, 0.0] pp |

Both seeds regress on the older five-family confirmation set. This set is a
regression check, not a fresh blind test, but the result prevents claiming a
general capability improvement from the age-distribution intervention.

## Failure analysis

The saved trajectories were summarized with
`scripts/analyze_failure_deltas.py`; the four JSON reports are the corresponding
`artifacts/return-v11-*-failure-deltas.summary.json` files. The main patterns
are:

- Seed 45 repeats `request_return` after the return window is closed: 58/120
  candidate failures contain both `return_window_closed` and `step_limit`.
  The candidate fails to inspect the target order on 16/64 failures.
- Seed 46 fixes the expired-return, position-1 cell (29/30 strict successes),
  but still gets 0/30 at expired-return, position 2. In that cell, 29/30
  episodes reach the correct terminal state but remain strict failures because
  a policy violation occurred; 7/30 also fail to inspect the target.
- On the old confirmation regression, seed 46 has 12 failures with a correct
  terminal state but a policy violation. This explains why terminal completion
  alone would overstate the improvement.

The failure report keeps task IDs, target age/position, violation signatures,
and candidate tool sequences for a small set of examples. It does not copy
model outputs into training data or choose a checkpoint.

## Interpretation

The intervention changed return target and distractor ages together while
holding the number of training views and optimizer inputs fixed. It did not
change the model architecture or introduce GRPO/RL. The evidence supports a
more limited conclusion: changing the supervised age distribution can move
performance on a particular synthetic return boundary, but the effect is
seed-sensitive and accompanied by regression on the prior confirmation set.

Do not report this experiment as a reproducible Agentic RL gain. The next
technical step, if the project continues, is to diagnose the position-2
expired-return failure and run a newly frozen third seed or a separately
controlled data intervention; do not select seed 46 as the headline model.

All runs use holdout digest
`c8e70024ca2b0ae69e8754a60e0ecbf32d11e1a66cb1bec3275c82a70d3282e3`,
`explicit-email-field-v1`, `strict-v2-target-inspection`, temperature 0, and
512 maximum new tokens.
