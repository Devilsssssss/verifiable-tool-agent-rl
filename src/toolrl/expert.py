from __future__ import annotations

import json
from typing import Any

from .environment import SCORING_PROTOCOL, RetailToolEnvironment
from .prompts import conversation_prompt
from .schemas import AgentTask


def reset_environment(
    environment: RetailToolEnvironment,
    task: AgentTask,
    *,
    reward_mode: str = "shaped",
    scoring_protocol: str = SCORING_PROTOCOL,
) -> None:
    environment.reset(
        initial_state_json=json.dumps(task.initial_state, sort_keys=True),
        goal_json=json.dumps(task.goal, sort_keys=True),
        max_steps=task.max_steps,
        optimal_steps=task.optimal_steps,
        reward_mode=reward_mode,
        scoring_protocol=scoring_protocol,
        task_id=task.task_id,
    )


def replay_expert(task: AgentTask, *, reward_mode: str = "shaped") -> dict[str, Any]:
    environment = RetailToolEnvironment()
    reset_environment(environment, task, reward_mode=reward_mode)
    messages: list[dict[str, Any]] = list(conversation_prompt(task.prompt))
    for index, step in enumerate(task.expert_plan, start=1):
        tool_name = str(step["tool"])
        arguments = dict(step["arguments"])
        call_id = f"call_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
            }
        )
        output = getattr(environment, tool_name)(**arguments)
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "tool_call_id": call_id,
                "content": output,
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": "The request has been completed according to the retail policy.",
        }
    )
    return {
        "task_id": task.task_id,
        "family": task.family,
        "messages": messages,
        "reward": environment.get_reward(),
        "reward_report": environment.reward_report,
        "trace": environment.trace,
    }


def training_row(task: AgentTask, *, reward_mode: str = "shaped") -> dict[str, Any]:
    replay = replay_expert(task, reward_mode=reward_mode)
    if not replay["reward_report"]["success"]:
        raise ValueError(f"expert plan failed for {task.task_id}")
    return {
        "task_id": task.task_id,
        "family": task.family,
        "split": task.split,
        "messages": replay["messages"],
    }


def repair_training_row(task: AgentTask) -> dict[str, Any]:
    """Create one verified tool-argument repair example for SFT."""
    environment = RetailToolEnvironment()
    reset_environment(environment, task)
    messages: list[dict[str, Any]] = list(conversation_prompt(task.prompt))
    inserted = False
    repair_boundary = 0
    call_index = 0
    for step in task.expert_plan:
        tool_name = str(step["tool"])
        arguments = dict(step["arguments"])
        call_index += 1
        if not inserted and tool_name in {
            "cancel_order",
            "request_return",
            "replace_order_item",
            "issue_refund",
            "escalate_to_human",
        }:
            extra_key = {
                "cancel_order": "user_id",
                "request_return": "reason",
                "replace_order_item": "user_id",
                "issue_refund": "user_id",
                "escalate_to_human": "order_id",
            }[tool_name]
            corrupted = dict(arguments)
            corrupted[extra_key] = 0 if extra_key.endswith("_id") else "not part of this tool"
            bad_call_id = f"repair_{call_index}_bad"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": bad_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": corrupted},
                        }
                    ],
                }
            )
            try:
                output = getattr(environment, tool_name)(**corrupted)
            except TypeError as error:
                output = json.dumps({"ok": False, "error": str(error)}, sort_keys=True)
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": bad_call_id,
                    "content": output,
                }
            )
            inserted = True
            repair_boundary = len(messages)
            call_index += 1
        call_id = f"call_{call_index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
            }
        )
        output = getattr(environment, tool_name)(**arguments)
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "tool_call_id": call_id,
                "content": output,
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": "The request has been completed according to the retail policy.",
        }
    )
    if not inserted or not environment.reward_report["success"]:
        raise ValueError(f"repair example failed for {task.task_id}")
    return {
        "task_id": f"{task.task_id}-repair",
        "family": task.family,
        "split": task.split,
        "messages": messages,
        "completion_start": repair_boundary,
    }


def grpo_row(
    task: AgentTask,
    *,
    reward_mode: str,
    scoring_protocol: str = SCORING_PROTOCOL,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "prompt": conversation_prompt(task.prompt),
        "initial_state_json": json.dumps(task.initial_state, sort_keys=True),
        "goal_json": json.dumps(task.goal, sort_keys=True),
        "max_steps": task.max_steps,
        "optimal_steps": task.optimal_steps,
        "reward_mode": reward_mode,
        "scoring_protocol": scoring_protocol,
    }

