from __future__ import annotations

import inspect
import json
from typing import Any

from toolrl.environment import REASON_CODE_SCORING_PROTOCOL, RetailToolEnvironment
from toolrl.generator import generate_tasks
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
