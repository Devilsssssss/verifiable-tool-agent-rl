# Return Boundary Training Protocol

## Purpose

The v10 seed replication did not satisfy the target-order inspection gate, and its
remaining strict failures concentrated in the eligible/expired return branch. This
is a supervised data-distribution comparison around the 30-day cutoff. It is not an
RL or GRPO experiment.

## Frozen intervention

The source is `data/generated/tasks_train.jsonl`, with 300 tasks and 60 tasks per
business family. The generated intervention keeps all 300 task slots and all 900
training views (normal, repair, and lookup). Only the 120 return tasks are changed:

- target position is balanced 30/30 within each return family;
- eligible target ages are 29 and 30 days;
- expired target ages are 31 and 32 days;
- the distractor order is fixed at 8 days;
- all non-return tasks and all prompts remain unchanged.

The independent held-out splits used by the semantic audit are the generated
validation/test tasks and the existing confirmation tasks. The resulting files and
hashes are recorded in `artifacts/return-boundary-v1-training-audit.summary.json`.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/create_return_boundary_training.py `
  --heldout data/generated/tasks_validation.jsonl `
    data/generated/tasks_test.jsonl data/confirmation/tasks_test.jsonl
```

The generator refuses to overwrite a different frozen output. The summary must show
`semantic_audit.status=pass`, 300 training tasks, 900 views, and the four age cells
at 30 examples each.

## Training and decision gate

Train the boundary candidate with the same Qwen3-1.7B LoRA settings and seed pair used
for the v10 comparison, then evaluate it on the already frozen return holdout and the
old position confirmation set. Report both seeds and every family/position cell.
Do not start GRPO until the boundary comparison is complete and the target-order
inspection gate is resolved. A gain on this supervised intervention must not be
described as evidence that RL helped.

The completed seed45/46 results are summarized in `docs/return_boundary_v1_results.md`.
Both pipelines completed holdout evaluation, replay, stratified analysis, old-confirmation
regression, and paired comparison. The target-order gate remains unresolved, so this
protocol does not authorize a GRPO run.

After each adapter has produced `training_manifest.json`, the reproducible evaluation
pipeline is:

```powershell
python scripts/evaluate_return_boundary_candidate.py --seed 45 --gpu 0
python scripts/evaluate_return_boundary_candidate.py --seed 46 --gpu 1
```

The pipeline checks the frozen data hashes and v10-matched training controls, evaluates
the 120-task return holdout and the old position confirmation set, replays every
trajectory, and writes paired/family summaries under `artifacts/`.
