import importlib.util
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from toolrl.audit import audit_tasks
from toolrl.generator import entity_tokens, generate_tasks
from toolrl.position_shift import create_balanced_position_tasks, target_order_position
from toolrl.return_holdout import (
    RETURN_FAMILIES,
    TRAIN_DAYS,
    build_return_holdout,
    vary_return_days,
)


def test_return_analysis_rejects_duplicate_and_mixed_protocol_rows() -> None:
    module_path = Path(__file__).parents[1] / "scripts" / "analyze_return_holdout.py"
    spec = importlib.util.spec_from_file_location("analyze_return_holdout", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    validate_rollouts = module._validate_rollouts

    tasks = [task for task in generate_tasks({"test": 10}, seed=97)
             if task.family in RETURN_FAMILIES]
    # The helper is intentionally tested independently of model output parsing.
    tasks = tasks[:2]
    task_index = {task.task_id: task for task in tasks}
    row = {
        "task_id": tasks[0].task_id,
        "success": False,
        "policy_compliant": False,
        "trace": [],
        "family": tasks[0].family,
        "target_order_inspected": False,
        "terminal_goal_satisfied": False,
        "run_name": "run-a",
        "scoring_protocol": "strict-v2-target-inspection",
        "input_intervention": "explicit-email-field-v1",
    }
    with pytest.raises(ValueError, match="exactly once"):
        validate_rollouts([row, deepcopy(row)], task_index)
    row2 = deepcopy(row)
    row2["task_id"] = tasks[1].task_id
    row2["family"] = tasks[1].family
    row2["run_name"] = "run-b"
    with pytest.raises(ValueError, match="one complete run_name"):
        validate_rollouts([row, row2], task_index)
    row2["run_name"] = "run-a"
    validate_rollouts([row, row2], task_index)
    with pytest.raises(ValueError, match="identical task IDs"):
        validate_rollouts([row], task_index)
    del row2["target_order_inspected"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_rollouts([row, row2], task_index)


def test_boundary_holdout_preserves_expert_correctness_and_entity_isolation() -> None:
    source = [task for task in generate_tasks({"test": 60}, seed=91)
              if task.family in RETURN_FAMILIES]
    before = deepcopy(source)
    holdout = build_return_holdout(source, seed=92, id_offset=50_000_000)
    assert source == before
    assert not entity_tokens(source) & entity_tokens(holdout)
    counts = Counter((task.family, target_order_position(task)) for task in holdout)
    assert all(counts[(family, position)] == 6
               for family in RETURN_FAMILIES for position in (1, 2))
    ages = Counter(order["days_since_delivery"] for task in holdout
                   for order in task.initial_state["orders"] if order["id"] == task.goal["order_id"])
    assert ages == {1: 4, 29: 4, 30: 4, 31: 4, 32: 4, 60: 4}
    report = audit_tasks(source + holdout)
    assert report["expert_successes"] == 48
    assert report["noop_successes"] == report["direct_shortcut_successes"] == 0


def test_training_age_intervention_leaves_nonreturns_and_task_semantics_intact() -> None:
    source = create_balanced_position_tasks(
        generate_tasks({"train": 60}), seed=93, output_split="train"
    )
    varied = vary_return_days(source, seed=94, days=TRAIN_DAYS)
    for before, after in zip(source, varied, strict=True):
        assert before.goal == after.goal
        assert before.expert_plan == after.expert_plan
        assert before.prompt == after.prompt
        assert target_order_position(before) == target_order_position(after)
        if before.family not in RETURN_FAMILIES:
            assert before == after
    assert audit_tasks(varied)["expert_successes"] == 60


def test_age_intervention_rejects_wrong_side_of_policy_boundary() -> None:
    source = create_balanced_position_tasks(
        generate_tasks({"train": 60}), seed=95, output_split="train"
    )
    with pytest.raises(ValueError, match="delivery ages disagree"):
        vary_return_days(source, seed=96, days={**TRAIN_DAYS, "return_expired": (30,)})
