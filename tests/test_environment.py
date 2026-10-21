from __future__ import annotations

import pytest

from toolrl.environment import REASON_CODE_SCORING_PROTOCOL, RetailToolEnvironment
from toolrl.expert import repair_training_row, replay_expert, reset_environment
from toolrl.generator import FAMILIES, generate_tasks


@pytest.mark.parametrize("family", FAMILIES)
def test_expert_plan_completes_each_family(family: str) -> None:
    tasks = generate_tasks({"train": len(FAMILIES)}, seed=7)
    task = next(task for task in tasks if task.family == family)

    replay = replay_expert(task)

    assert replay["reward_report"]["success"] is True
    assert replay["reward_report"]["target_order_inspected"] is True
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


def test_escalating_after_inspecting_only_distractor_is_not_success() -> None:
    task = next(
        task
        for task in generate_tasks({"train": 5}, seed=8)
        if task.family == "cancel_shipped"
    )
    environment = RetailToolEnvironment()
    reset_environment(environment, task)
    user_id = int(task.initial_state["users"][0]["id"])
    target_id = int(task.goal["order_id"])
    distractor_id = next(
        int(row["id"]) for row in task.initial_state["orders"] if int(row["id"]) != target_id
    )

    environment.lookup_user(task.initial_state["users"][0]["email"])
    environment.list_orders(user_id)
    environment.get_order(distractor_id)
    environment.escalate_to_human("The requested order cannot be changed.")

    report = environment.reward_report
    assert report["terminal_goal_satisfied"] is True
    assert report["target_order_inspected"] is False
    assert report["success"] is False


def test_reason_code_protocol_rejects_wrong_escalation_reason() -> None:
    task = next(
        task
        for task in generate_tasks({"train": 5}, seed=8)
        if task.family == "cancel_shipped"
    )
    environment = RetailToolEnvironment()
    reset_environment(environment, task, scoring_protocol=REASON_CODE_SCORING_PROTOCOL)
    user_id = int(task.initial_state["users"][0]["id"])
    target_id = int(task.goal["order_id"])
    environment.lookup_user(task.initial_state["users"][0]["email"])
    environment.list_orders(user_id)
    environment.get_order(target_id)

    output = environment.escalate_to_human(
        "The order has shipped and cannot be cancelled.",
        reason_code="return_window_expired",
    )

    assert "escalation_reason_code_mismatch" in output
    assert environment.reward_report["success"] is False
    assert environment.reward_report["policy_compliant"] is False


def test_reason_code_protocol_accepts_matching_escalation_reason() -> None:
    task = next(
        task
        for task in generate_tasks({"train": 5}, seed=8)
        if task.family == "return_expired"
    )
    environment = RetailToolEnvironment()
    reset_environment(environment, task, scoring_protocol=REASON_CODE_SCORING_PROTOCOL)
    user_id = int(task.initial_state["users"][0]["id"])
    target_id = int(task.goal["order_id"])
    environment.lookup_user(task.initial_state["users"][0]["email"])
    environment.list_orders(user_id)
    environment.get_order(target_id)
    environment.escalate_to_human(
        "The return window has expired.", reason_code="return_window_expired"
    )

    assert environment.reward_report["success"] is True
    assert environment.reward_report["terminal_goal_satisfied"] is True


def test_reason_code_protocol_still_requires_inspecting_the_target_order() -> None:
    task = next(
        task
        for task in generate_tasks({"train": 5}, seed=8)
        if task.family == "return_expired"
    )
    environment = RetailToolEnvironment()
    reset_environment(environment, task, scoring_protocol=REASON_CODE_SCORING_PROTOCOL)
    user_id = int(task.initial_state["users"][0]["id"])
    target_id = int(task.goal["order_id"])
    distractor_id = next(
        int(row["id"])
        for row in task.initial_state["orders"]
        if int(row["id"]) != target_id
    )
    environment.lookup_user(task.initial_state["users"][0]["email"])
    environment.list_orders(user_id)
    environment.get_order(distractor_id)
    environment.escalate_to_human(
        "The return window has expired.", reason_code="return_window_expired"
    )

    assert environment.reward_report["terminal_goal_satisfied"] is True
    assert environment.reward_report["target_order_inspected"] is False
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


def test_strict_shaped_reward_separates_terminal_and_policy_diagnostics() -> None:
    task = next(task for task in generate_tasks({"train": 10}) if task.family == "return_expired")
    environment = RetailToolEnvironment()
    reset_environment(
        environment,
        task,
        reward_mode="strict_shaped",
        scoring_protocol="strict-v3-reason-code",
    )
    for step in task.expert_plan:
        getattr(environment, step["tool"])(**step["arguments"])

    report = environment.reward_report
    assert report["success"] is True
    assert report["terminal_goal_satisfied"] is True
    assert report["target_order_inspected"] is True
    assert report["policy_compliant"] is True
    assert report["strict_shaped"] > 1.5
    assert environment.get_reward() == report["strict_shaped"]


@pytest.mark.parametrize("family", FAMILIES)
def test_repair_training_row_contains_feedback_and_valid_recovery(family: str) -> None:
    task = next(task for task in generate_tasks({"train": 5}, seed=42) if task.family == family)
    row = repair_training_row(task)
    prompt_contents = [
        message.get("content") for message in row["messages"][: row["completion_start"]]
    ]
    assert any("unexpected keyword argument" in str(content) for content in prompt_contents)
    assert row["messages"][row["completion_start"] - 1]["role"] == "tool"
    assert row["messages"][row["completion_start"]]["role"] == "assistant"
    repaired_call = row["messages"][row["completion_start"]]["tool_calls"][0]["function"]
    expected = next(
        step
        for step in task.expert_plan
        if step["tool"]
        in {
            "cancel_order",
            "request_return",
            "replace_order_item",
            "issue_refund",
            "escalate_to_human",
        }
    )
    assert repaired_call["name"] == expected["tool"]
    assert repaired_call["arguments"] == expected["arguments"]


@pytest.mark.parametrize("family", ["cancel_processing", "cancel_shipped", "return_eligible"])
def test_terminal_recovery_does_not_relax_strict_success(family: str) -> None:
    task = next(task for task in generate_tasks({"train": 5}) if task.family == family)
    environment = RetailToolEnvironment()
    reset_environment(environment, task)
    environment._record_external_failure("lookup_user", {"order_id": 0}, "missing email")
    for step in task.expert_plan:
        getattr(environment, step["tool"])(**step["arguments"])

    report = environment.reward_report
    assert report["terminal_goal_satisfied"] is True
    assert report["format_recovery_success"] is True
    assert report["success"] is False
    assert report["policy_compliant"] is False
    assert report["sparse"] == 0.0


def test_business_policy_violation_disqualifies_format_recovery() -> None:
    task = generate_tasks({"train": 1})[0]
    environment = RetailToolEnvironment()
    reset_environment(environment, task)
    environment.cancel_order(int(task.goal["order_id"]))
    environment._record_external_failure("lookup_user", {"order_id": 0}, "missing email")
    for step in task.expert_plan:
        getattr(environment, step["tool"])(**step["arguments"])

    report = environment.reward_report
    assert report["terminal_goal_satisfied"] is True
    assert report["format_recovery_success"] is False
    assert report["success"] is False
