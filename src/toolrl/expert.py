from __future__ import annotations

import json
from typing import Any

from .environment import RetailToolEnvironment
from .prompts import conversation_prompt
from .schemas import AgentTask


def reset_environment(
    environment: RetailToolEnvironment,
    task: AgentTask,
    *,
    reward_mode: str = "shaped",
) -> None:
    environment.reset(
        initial_state_json=json.dumps(task.initial_state, sort_keys=True),
        goal_json=json.dumps(task.goal, sort_keys=True),
        max_steps=task.max_steps,
        optimal_steps=task.optimal_steps,
        reward_mode=reward_mode,
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


def grpo_row(task: AgentTask, *, reward_mode: str) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "prompt": conversation_prompt(task.prompt),
        "initial_state_json": json.dumps(task.initial_state, sort_keys=True),
        "goal_json": json.dumps(task.goal, sort_keys=True),
        "max_steps": task.max_steps,
        "optimal_steps": task.optimal_steps,
        "reward_mode": reward_mode,
    }

