from __future__ import annotations

import pytest

from toolrl.environment import RetailToolEnvironment
from toolrl.expert import replay_expert, reset_environment
from toolrl.generator import FAMILIES, generate_tasks


@pytest.mark.parametrize("family", FAMILIES)
def test_expert_plan_completes_each_family(family: str) -> None:
    tasks = generate_tasks({"train": len(FAMILIES)}, seed=7)
    task = next(task for task in tasks if task.family == family)

    replay = replay_expert(task)

    assert replay["reward_report"]["success"] is True
    assert replay["reward_report"]["policy_compliant"] is True
    assert replay["reward_report"]["sparse"] == 1.0
    assert replay["reward_report"]["shaped"] > 1.0


def test_direct_state_change_is_rejected_before_inspection() -> None:
    task = generate_tasks({"train": 1})[0]
    environment = RetailToolEnvironment()
    reset_environment(environment, task)

    output = environment.cancel_order(int(task.goal["order_id"]))

    assert '"ok": false' in output
    assert environment.reward_report["success"] is False
    assert environment.reward_report["policy_compliant"] is False
    assert environment.state_snapshot["orders"][0]["status"] == "processing"


def test_noop_receives_negative_shaped_reward() -> None:
    task = generate_tasks({"train": 1})[0]
    environment = RetailToolEnvironment()
    reset_environment(environment, task)

    assert environment.get_reward() == -0.2
    assert environment.reward_report["success"] is False


def test_step_limit_is_enforced() -> None:
    task = generate_tasks({"train": 1})[0]
    environment = RetailToolEnvironment()
    reset_environment(environment, task)
    environment._max_steps = 1

    environment.lookup_user(task.initial_state["users"][0]["email"])
    output = environment.lookup_user(task.initial_state["users"][0]["email"])

    assert "tool_call_limit_exceeded" in output
    assert environment.reward_report["policy_compliant"] is False


def test_shaped_reward_reports_correct_prefix_progress() -> None:
    task = generate_tasks({"train": 1})[0]
    environment = RetailToolEnvironment()
    reset_environment(environment, task)
    environment.lookup_user(task.initial_state["users"][0]["email"])
    environment.list_orders(int(task.initial_state["users"][0]["id"]))

    assert environment.reward_report["progress"] == 2 / task.optimal_steps
    assert environment.get_reward() > -0.2

