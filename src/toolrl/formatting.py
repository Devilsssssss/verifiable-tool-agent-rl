from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

INPUT_FORMAT = "canonical-v4"


def sort_template_json(tokenizer: Any) -> Any:
    """Match canonical serialization inside GRPO's native callable tool loop."""
    if not isinstance(tokenizer.chat_template, str):
        raise TypeError("this protocol requires a single Qwen chat template")
    tokenizer.chat_template = re.sub(
        r"\|\s*tojson(?!\s*\()", "| tojson(sort_keys=True)", tokenizer.chat_template
    )
    return tokenizer


def canonical_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep heterogeneous argument objects out of Arrow's nested struct union.

    Qwen's template accepts arguments as a JSON string and emits the object
    directly. Preserve all supplied arguments, including deliberate errors;
    this is serialization, not tool-call correction.
    """
    result = deepcopy(messages)
    for message in result:
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except json.JSONDecodeError:
                    # Keep malformed generated calls visible in recovery history.
                    continue
                if not isinstance(decoded, dict):
                    continue
                arguments = decoded
            function["arguments"] = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return result


def canonical_tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    from transformers.utils import get_json_schema

    schemas = [tool if isinstance(tool, dict) else get_json_schema(tool) for tool in tools]
    return json.loads(json.dumps(schemas, sort_keys=True, ensure_ascii=False))


def sft_conversation(row: dict[str, Any], schemas: list[dict[str, Any]]) -> dict[str, Any]:
    if "messages" in row:
        boundary = int(row.get("completion_start", 2))
        if not 0 < boundary < len(row["messages"]):
            raise ValueError(f"invalid completion boundary for {row['task_id']}")
        prompt, completion = row["messages"][:boundary], row["messages"][boundary:]
    else:
        prompt, completion = row["prompt"], row["completion"]
    return {
        "task_id": row["task_id"],
        "family": row["family"],
        "prompt": canonical_messages(prompt),
        "completion": canonical_messages(completion),
        "tools": json.dumps(schemas, sort_keys=True, ensure_ascii=False),
    }
