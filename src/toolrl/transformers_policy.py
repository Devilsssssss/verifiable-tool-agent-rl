from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

os.environ.setdefault("USE_TF", "0")


def normalize_tool_response(response: dict[str, Any]) -> dict[str, Any]:
    """Recover strict JSON tool calls emitted as plain assistant content.

    Some models generate the response-schema JSON correctly but wrap it in chat
    control tokens that ``parse_response`` does not recognize. Only the exact
    ``{"name": ..., "arguments": {...}}`` shape is accepted here; malformed
    and unknown calls still flow through the environment's normal penalties.
    """
    native_calls = response.get("tool_calls")
    if isinstance(native_calls, list) and native_calls:
        cleaned_calls = []
        changed = False
        for call in native_calls:
            if not isinstance(call, dict):
                cleaned_calls.append(call)
                continue
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("arguments"), dict):
                cleaned_calls.append(call)
                continue
            arguments = function["arguments"]
            clean_arguments = {key: value for key, value in arguments.items() if value is not None}
            changed = changed or len(clean_arguments) != len(arguments)
            cleaned_calls.append(
                {**call, "function": {**function, "arguments": clean_arguments}}
            )
        if changed:
            return {**response, "tool_calls": cleaned_calls}
        return response
    content = response.get("content")
    if not isinstance(content, str):
        return response

    decoder = json.JSONDecoder()
    for position, character in enumerate(content):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[position:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name")
        arguments = candidate.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        clean_arguments = {key: value for key, value in arguments.items() if value is not None}
        return {
            **response,
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": name, "arguments": clean_arguments},
                }
            ],
        }
    return response


class TransformersToolPolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        from trl.chat_template_utils import add_response_schema, get_training_chat_template

        self.model = model
        self.tokenizer = add_response_schema(tokenizer)
        patched = get_training_chat_template(self.tokenizer)
        if patched is not None:
            self.tokenizer.chat_template = patched
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[..., str]],
    ) -> dict[str, Any]:
        import torch
        from trl.chat_template_utils import parse_response

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        completion_ids = output_ids[0, prompt_length:].tolist()
        response = parse_response(
            self.tokenizer,
            completion_ids,
            prefix=encoded["input_ids"][0].tolist(),
        )
        response = normalize_tool_response(response)
        response["_generated_tokens"] = len(completion_ids)
        return response
