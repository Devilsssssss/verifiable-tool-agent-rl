# Return Boundary Holdout: v10 Baselines

This document records the three position-v10 baselines on the frozen 120-task
return holdout. The holdout digest is
`c8e70024ca2b0ae69e8754a60e0ecbf32d11e1a66cb1bec3275c82a70d3282e3`.
All runs use the same Qwen3-1.7B adapters, greedy decoding (`temperature=0`,
512 new tokens), `explicit-email-field-v1`, and
`strict-v2-target-inspection`. Every saved rollout was replayed successfully
and its original score was unchanged.

| adapter seed | strict success | eligible | expired | target inspection | policy compliance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | 58/120 (48.3%) | 50/60 (83.3%) | 8/60 (13.3%) | 97/120 (80.8%) | 62/120 (51.7%) |
| 46 | 54/120 (45.0%) | 46/60 (76.7%) | 8/60 (13.3%) | 100/120 (83.3%) | 62/120 (51.7%) |
| 47 | 46/120 (38.3%) | 18/60 (30.0%) | 28/60 (46.7%) | 78/120 (65.0%) | 108/120 (90.0%) |

The seed spread is itself evidence that one evaluation is insufficient. The
failure labels differ as well: seed45 repeatedly attempted `request_return`
until the step limit (58 failed tasks had both `return_window_closed` and
`step_limit` violations); seed46 had 58 failed tasks with a closed-window
violation, and 51 failed tasks nevertheless reached the correct terminal state.
Seed47 had 42 failures without target inspection, and 31 failures with the
correct terminal state. Counts may overlap. No run had wrong-email lookups or
format errors. Therefore the v11 result will be reported
for both prespecified seeds, with paired gained/lost tasks and family/position
cells, rather than selecting the better seed.

These numbers are a descriptive synthetic holdout result. The fresh entities,
wording, and ages change together, so they do not isolate delivery-age
generalization or establish real-world business performance.

The previous position confirmation set contains five task families, whereas
this holdout contains only returns. Their overall rates must not be treated as
a paired degradation. Restricting the old set to its 48 return tasks yields
33/48, 29/48, and 29/48 for seeds 45, 46, and 47 respectively; even those are
unpaired collections with different wording/entities/ages. The two sets
cannot establish an isolated age effect.

## Evidence and reproduction

For each seed `N`, `artifacts/return-holdout-seedN.summary.json` records the
decoding protocol and overall/family scores; `-replay.summary.json` records the
120 verified trajectories; `-analysis.summary.json` records position, age,
and failure-diagnostic counts. Raw trajectories and generated data stay out
of Git.

Once the frozen data, base model, and final adapters exist locally, run:

```bash
python scripts/evaluate_return_candidate.py --seed 45 --gpu 0
python scripts/evaluate_return_candidate.py --seed 46 --gpu 1
```

Each command verifies matched training controls and data hashes, evaluates
the frozen holdout and old confirmation set, replays all candidate rollouts,
and generates paired gained/lost-task comparisons. A partial evaluation is
never silently overwritten; a completed one is checked and reused. The
pipeline summary records completed stages and failures. GPU jobs are already
running for this round; these commands document reproduction, not a request
to launch duplicates.
