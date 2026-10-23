from __future__ import annotations

import inspect
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any, Literal

from .environment import SCORING_PROTOCOL, RetailToolEnvironment
from .expert import reset_environment
from .prompts import (
    INPUT_CONTRACTS,
    LEGACY_INPUT_CONTRACT,
    REASON_CODE_INPUT_CONTRACT,
    conversation_prompt,
)
from .schemas import AgentTask

ResponsePolicy = Callable[[list[dict[str, Any]], list[Callable[..., str]]], dict[str, Any]]


def environment_tools(
    environment: RetailToolEnvironment,
    *,
    input_contract: str = LEGACY_INPUT_CONTRACT,
) -> list[Callable[..., str]]:
    if input_contract not in INPUT_CONTRACTS:
        raise ValueError(f"unsupported input contract: {input_contract}")
    return [
        _tool_adapter(
            environment,
            name,
            require_reason_code=input_contract == REASON_CODE_INPUT_CONTRACT,
        )
        for name, method in inspect.getmembers(type(environment), predicate=inspect.isfunction)
        if name not in {"reset", "get_reward"} and not name.startswith("_")
    ]


def _tool_adapter(
    environment: RetailToolEnvironment,
    name: str,
    *,
    require_reason_code: bool = False,
) -> Callable[..., str]:
    method = getattr(environment, name)

    @wraps(method)
    def call(**arguments: Any) -> str:
        # Chat-model schemas often materialize unused optional slots as null.
        # Remove only nulls; non-null unknown arguments remain strict failures.
        return method(**{key: value for key, value in arguments.items() if value is not None})

    signature = inspect.signature(method)
    if require_reason_code and name == "escalate_to_human":
        parameters = [
            parameter.replace(
                default=inspect.Parameter.empty,
                annotation=Literal[
                    "shipped_order_cannot_cancel",
                    "return_window_expired",
                ],
            )
            if parameter.name == "reason_code"
            else parameter
            for parameter in signature.parameters.values()
        ]
        signature = signature.replace(parameters=parameters)
    call.__signature__ = signature  # type: ignore[attr-defined]
    return call


def run_episode(
    task: AgentTask,
    policy: ResponsePolicy,
    *,
    reward_mode: str = "shaped",
    scoring_protocol: str = SCORING_PROTOCOL,
    input_contract: str = LEGACY_INPUT_CONTRACT,
    max_turns: int | None = None,
) -> dict[str, Any]:
    environment = RetailToolEnvironment()
    reset_environment(
        environment,
        task,
        reward_mode=reward_mode,
        scoring_protocol=scoring_protocol,
    )
    messages: list[dict[str, Any]] = list(
        conversation_prompt(task.prompt, input_contract=input_contract)
    )
    turns = max_turns or task.max_steps + 2
    generated_tokens = 0
    stopped_reason = "turn_limit"

    for turn_index in range(turns):
        tools = environment_tools(environment, input_contract=input_contract)
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
