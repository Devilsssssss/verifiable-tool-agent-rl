from __future__ import annotations

import inspect
import json
from typing import Any

from toolrl.environment import REASON_CODE_SCORING_PROTOCOL, RetailToolEnvironment
from toolrl.generator import generate_tasks
from toolrl.prompts import REASON_CODE_INPUT_CONTRACT
from toolrl.rollout import environment_tools, run_episode


def test_tools_can_be_enumerated_before_environment_reset() -> None:
    tools = environment_tools(RetailToolEnvironment())

    assert {tool.__name__ for tool in tools} == {
        "cancel_order",
        "check_inventory",
        "escalate_to_human",
        "get_order",
        "issue_refund",
        "list_orders",
        "lookup_user",
        "replace_order_item",
        "request_return",
        "search_products",
    }


def test_tool_adapter_drops_null_schema_slots() -> None:
    environment = RetailToolEnvironment()
    task = generate_tasks({"train": 1})[0]
    environment.reset(
        initial_state_json=json.dumps(task.initial_state),
        goal_json=json.dumps(task.goal),
        max_steps=task.max_steps,
        optimal_steps=task.optimal_steps,
    )
    lookup = next(tool for tool in environment_tools(environment) if tool.__name__ == "lookup_user")

    result = lookup(email=task.initial_state["users"][0]["email"], user_id=None)

    assert '"ok": true' in result


def test_reason_code_input_contract_requires_reason_code() -> None:
    task = next(task for task in generate_tasks({"train": 5}, seed=8) if task.family == "return_expired")
    environment = RetailToolEnvironment()
    environment.reset(
        initial_state_json=json.dumps(task.initial_state),
        goal_json=json.dumps(task.goal),
        max_steps=task.max_steps,
        optimal_steps=task.optimal_steps,
        scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
    )
    escalate = next(
        tool
        for tool in environment_tools(
            environment,
            input_contract=REASON_CODE_INPUT_CONTRACT,
        )
        if tool.__name__ == "escalate_to_human"
    )
    parameter = inspect.signature(escalate).parameters["reason_code"]
    assert parameter.default is inspect.Parameter.empty
    assert "shipped_order_cannot_cancel" in str(parameter.annotation)
    assert "return_window_expired" in str(parameter.annotation)


def test_strict_v3_scoring_does_not_change_legacy_tool_schema() -> None:
    environment = RetailToolEnvironment()
    escalate = next(
        tool
        for tool in environment_tools(environment)
        if tool.__name__ == "escalate_to_human"
    )

    assert inspect.signature(escalate).parameters["reason_code"].default is None


def test_trl_style_reflection_is_safe_before_environment_reset() -> None:
    methods = inspect.getmembers(RetailToolEnvironment(), predicate=inspect.ismethod)

    assert "reset" in {name for name, _ in methods}
    assert "get_reward" in {name for name, _ in methods}


def test_rollout_executes_structured_tool_calls() -> None:
    task = generate_tasks({"train": 1})[0]
    plan = iter(task.expert_plan)

    def policy(messages: list[dict[str, Any]], tools: list) -> dict[str, Any]:
        del messages, tools
        try:
            step = next(plan)
        except StopIteration:
            return {"role": "assistant", "content": "done"}
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": step["tool"],
                        "arguments": step["arguments"],
                    },
                }
            ],
        }

    record = run_episode(task, policy)

    assert record["success"] is True
    assert record["policy_compliant"] is True
    assert record["steps"] == task.optimal_steps


def test_rollout_supports_reason_code_scoring_protocol() -> None:
    task = generate_tasks({"train": 1})[0]
    plan = iter(task.expert_plan)

    def policy(messages: list[dict[str, Any]], tools: list) -> dict[str, Any]:
        del messages, tools
        try:
            step = next(plan)
        except StopIteration:
            return {"role": "assistant", "content": "done"}
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": step["tool"], "arguments": step["arguments"]},
                }
            ],
        }

    record = run_episode(
        task,
        policy,
        reward_mode="strict_shaped",
        scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
    )

    assert record["success"] is True
    assert record["reward"] > 1.5


def test_rollout_penalizes_unknown_tool() -> None:
    task = generate_tasks({"train": 1})[0]

    def policy(messages: list[dict[str, Any]], tools: list) -> dict[str, Any]:
        del messages, tools
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "invented_tool", "arguments": {}},
                }
            ],
        }

    record = run_episode(task, policy, max_turns=1)

    assert record["success"] is False
    assert record["policy_compliant"] is False
    assert record["violations"] == ["invalid_tool_call"]
