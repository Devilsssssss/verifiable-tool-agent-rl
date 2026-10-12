from copy import deepcopy
from types import SimpleNamespace

from toolrl.transformers_policy import TransformersToolPolicy, normalize_tool_response


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


def test_raw_capture_does_not_correct_email_or_change_policy_output(monkeypatch):
    import sys

    raw = '<tool_call>{"name":"lookup_user","arguments":{"email":"For account a@test"}}</tool_call>'
    parsed = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "lookup_user", "arguments": {"email": "For account a@test"}}}
        ],
    }

    class Tokens:
        shape = (1, 2)

        def to(self, device):
            return self

        def __getitem__(self, index):
            self.values = [1, 2] if index == 0 else [3, 4, 5]
            return self

        def tolist(self):
            return self.values

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=Context))
    monkeypatch.setitem(
        sys.modules,
        "trl.chat_template_utils",
        SimpleNamespace(parse_response=lambda *args, **kwargs: deepcopy(parsed)),
    )
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        apply_chat_template=lambda *args, **kwargs: {"input_ids": Tokens()},
        decode=lambda ids, **kwargs: raw,
    )
    model = SimpleNamespace(device="cpu", generate=lambda **kwargs: Tokens())

    def create(capture):
        policy = TransformersToolPolicy.__new__(TransformersToolPolicy)
        policy.model, policy.tokenizer = model, tokenizer
        policy.max_new_tokens, policy.temperature, policy.top_p = 10, 0, 0.9
        policy.input_format = "legacy-v3"
        policy.capture_generation, policy.last_generation = capture, None
        return policy

    recorded, plain = create(True), create(False)
    assert recorded([], []) == plain([], []) == {**parsed, "_generated_tokens": 3}
    assert recorded.last_generation["raw_completion"] == raw
    assert recorded.last_generation["parsed_response"] == parsed
    assert recorded.last_generation["normalized_response"] == parsed
    assert plain.last_generation is None
