from toolrl.transformers_policy import normalize_tool_response


def test_normalize_wrapped_response_schema_call() -> None:
    response = {
        "role": "assistant",
        "content": (
            '<|im_start|>\n{"name": "lookup_user", "arguments": '
            '{"email": "test@example.test", "order_id": null}}\n<|im_start|>'
        ),
    }

    normalized = normalize_tool_response(response)

    assert normalized["content"] is None
    assert normalized["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_user",
                "arguments": {"email": "test@example.test"},
            },
        }
    ]


def test_normalize_leaves_unstructured_text_unchanged() -> None:
    response = {"role": "assistant", "content": "The request is complete."}

    assert normalize_tool_response(response) == response


def test_normalize_preserves_native_tool_calls() -> None:
    response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_user",
                    "arguments": {"email": "a@b.test", "user_id": None},
                },
            }
        ],
    }

    normalized = normalize_tool_response(response)
    assert normalized["tool_calls"][0]["function"]["arguments"] == {"email": "a@b.test"}
