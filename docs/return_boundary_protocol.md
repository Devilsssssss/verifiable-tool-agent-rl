# Return Boundary Generalization Protocol

## Question

Does a model trained on a wider distribution of delivery ages make fewer return-policy and target-order mistakes than the current position-v10 model, while keeping the number of training views and optimizer inputs fixed?

## Frozen holdout

The holdout is generated before any candidate adapter is evaluated and is never used for training.

- 120 fresh return tasks: 60 `return_eligible` and 60 `return_expired`.
- Within each family, the target order is in position 1 for 30 tasks and position 2 for 30 tasks.
- Eligible target ages are 1, 29, and 30 days; expired target ages are 31, 32, and 60 days.
- Distractor ages are independently assigned from 8 and 45 days.
- Prompts, names, emails, and database IDs are fresh; IDs are offset by 50,000,000.
- The frozen file and audit summary record the seed, SHA-256 digest, family/position cells, and semantic audit result.

The public development data, the position-v10 training data, and the optional confirmation set are included in the collision audit. Any entity overlap or expert-plan failure stops the experiment.

## Baseline evaluation

Evaluate the existing position-v10 adapters for seeds 45, 46, and 47 with the same settings used for the prior confirmation evaluation:

- explicit email-field intervention;
- greedy decoding, temperature 0, maximum 512 new tokens;
- the existing strict-v2 evaluator and 120 holdout tasks;
- no retries or outcome-dependent prompt changes.

Record strict success by family and target position, target-order inspection, wrong-email use, policy violations, and terminal completion. The holdout result is descriptive: it is not used to select individual examples or tune a seed.

## Optional training intervention

Only after the three baseline runs are archived, create `return-v11` from the existing position-v10 training tasks. The intervention changes only return target/distractor delivery ages:

- eligible target ages: 2, 14, 28 days;
- expired target ages: 33, 46, 75 days;
- distractor ages: 8 and 45 days.

Task identities, prompts, family/position balance, lookup views, view counts, and training code remain unchanged. The resulting run is an equal-view data-distribution intervention, not a new RL algorithm. It must be evaluated on the already frozen holdout with the same decoding and strictness settings.

## Interpretation

Do not claim that GRPO or RL improves performance unless a separately controlled RL experiment shows a reproducible gain. If v10 fails on the frozen boundary set, report the failure mode and compare v11 only as a supervised data-distribution intervention. A gain must be reported together with family/position breakdowns and the exact frozen holdout digest.
