import json

import pytest

from toolrl.formatting import canonical_messages, sft_conversation


def test_malformed_call_history_remains_visible_for_recovery() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "lookup_user", "arguments": "{broken json"}}],
        },
        {"role": "tool", "content": "tool arguments must be a JSON object"},
    ]
    assert canonical_messages(messages) == messages


def test_serialization_preserves_error_arguments_without_mutating_trace() -> None:
    arguments = {"order_id": 12, "extra": {"wrong": True}, "nullable": None}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "cancel_order", "arguments": arguments}}],
        }
    ]
    formatted = canonical_messages(messages)
    encoded = formatted[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(encoded) == arguments
    assert messages[0]["tool_calls"][0]["function"]["arguments"] is arguments
    assert canonical_messages(formatted) == formatted


def test_completion_boundary_keeps_invalid_call_out_of_targets() -> None:
    row = {
        "task_id": "repair",
        "family": "cancel_processing",
        "completion_start": 4,
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "cancel"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "cancel_order", "arguments": {"order_id": 12, "bad": 1}}}
                ],
            },
            {"role": "tool", "content": "unexpected keyword"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "cancel_order", "arguments": {"order_id": 12}}}
                ],
            },
        ],
    }
    result = sft_conversation(row, [])
    assert "bad" in result["prompt"][2]["tool_calls"][0]["function"]["arguments"]
    assert "bad" not in result["completion"][0]["tool_calls"][0]["function"]["arguments"]
    row["completion_start"] = len(row["messages"])
    with pytest.raises(ValueError, match="boundary"):
        sft_conversation(row, [])
