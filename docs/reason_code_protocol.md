# Strict-v3 Reason-Code Protocol

The historical evaluator remains `strict-v2-target-inspection`. This document
defines an opt-in `strict-v3-reason-code` protocol that closes a narrow scoring
blind spot for `escalate_unchanged` tasks.

## What changes

Generated escalation goals carry one structured code:

| task condition | required code |
| --- | --- |
| shipped order cannot be cancelled | `shipped_order_cannot_cancel` |
| delivered order is beyond the 30-day return window | `return_window_expired` |

The model still supplies a natural-language `reason` for readability, but the
environment validates the structured `reason_code` against the inspected goal
order. A missing or mismatched code records
`escalation_reason_code_mismatch`, so the episode cannot be strict-v3 success.
Target-order inspection remains mandatory.

## Compatibility

No historical strict-v2 score is changed. `RetailToolEnvironment.reset()`
defaults to strict-v2, and old traces preserve their original escalation
argument shape. To use the new protocol in a controlled local test:

```python
from toolrl.environment import REASON_CODE_SCORING_PROTOCOL
from toolrl.expert import reset_environment

reset_environment(environment, task, scoring_protocol=REASON_CODE_SCORING_PROTOCOL)
```

`scripts/rescore_rollouts.py` accepts `--scoring-protocol
strict-v3-reason-code` for an explicitly labeled offline replay. Do not mix
strict-v2 and strict-v3 summaries in one comparison, and do not overwrite the
historical artifacts.

## Scope

This is an evaluator-integrity improvement, not a model-training result. The
existing v10/v11 numbers and all README claims remain strict-v2 results. A
future v3 evaluation should include positive examples, a correct target with a
wrong reason code, and a distractor-only inspection before escalation.
