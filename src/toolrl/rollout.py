from __future__ import annotations

import inspect
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any

from .environment import RetailToolEnvironment
from .expert import reset_environment
from .prompts import conversation_prompt
from .schemas import AgentTask

ResponsePolicy = Callable[[list[dict[str, Any]], list[Callable[..., str]]], dict[str, Any]]


def environment_tools(environment: RetailToolEnvironment) -> list[Callable[..., str]]:
    return [
        _tool_adapter(environment, name)
        for name, method in inspect.getmembers(type(environment), predicate=inspect.isfunction)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    ]


def _tool_adapter(environment: RetailToolEnvironment, name: str) -> Callable[..., str]:
    method = getattr(environment, name)

    @wraps(method)
    def call(**arguments: Any) -> str:
        # Chat-model schemas often materialize unused optional slots as null.
        # Remove only nulls; non-null unknown arguments remain strict failures.
        return method(**{key: value for key, value in arguments.items() if value is not None})

    call.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
    return call


def run_episode(
    task: AgentTask,
    policy: ResponsePolicy,
    *,
    reward_mode: str = "shaped",
    max_turns: int | None = None,
) -> dict[str, Any]:
    environment = RetailToolEnvironment()
    reset_environment(environment, task, reward_mode=reward_mode)
    messages: list[dict[str, Any]] = list(conversation_prompt(task.prompt))
    turns = max_turns or task.max_steps + 2
    generated_tokens = 0
    stopped_reason = "turn_limit"

    for turn_index in range(turns):
        tools = environment_tools(environment)
        response = deepcopy(policy(deepcopy(messages), tools))
        response.setdefault("role", "assistant")
        generated_tokens += int(response.pop("_generated_tokens", 0))
        messages.append(response)
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            stopped_reason = "assistant_final"
            break
        tool_map = {tool.__name__: tool for tool in tools}
        for call_index, tool_call in enumerate(tool_calls):
            call_id = str(tool_call.get("id") or f"turn_{turn_index}_call_{call_index}")
            function = tool_call.get("function") or {}
            name = str(function.get("name", "unknown"))
            arguments = function.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
                output = environment._record_external_failure(
                    name, arguments, "tool arguments must be a JSON object"
                )
            elif name not in tool_map:
                output = environment._record_external_failure(name, arguments, "unknown tool")
            else:
                try:
                    output = tool_map[name](**arguments)
                except (TypeError, ValueError) as exc:
                    output = environment._record_external_failure(name, arguments, str(exc))
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call_id,
                    "content": output,
                }
            )

    report = environment.reward_report
    return {
        "task_id": task.task_id,
        "family": task.family,
        **report,
        "reward": environment.get_reward(),
        "stopped_reason": stopped_reason,
        "generated_tokens": generated_tokens,
        "recovered_from_failure": bool(
            report["success"] and any(not step["success"] for step in environment.trace)
        ),
        "trace": environment.trace,
        "messages": messages,
    }
