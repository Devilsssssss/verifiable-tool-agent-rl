from __future__ import annotations

import pytest

from toolrl.prompts import REASON_CODE_INPUT_CONTRACT, conversation_prompt


def test_strict_v2_prompt_remains_unchanged() -> None:
    messages = conversation_prompt("return the item")

    assert len(messages) == 2
    assert "reason_code" not in messages[0]["content"]


def test_reason_code_prompt_exposes_contract() -> None:
    messages = conversation_prompt(
        "return the item",
        input_contract=REASON_CODE_INPUT_CONTRACT,
    )

    content = messages[0]["content"]
    assert "reason_code" in content
    assert "return_window_expired" in content
    assert "Do not retry" in content


def test_prompt_rejects_unknown_input_contract() -> None:
    with pytest.raises(ValueError, match="unsupported input contract"):
        conversation_prompt("return the item", input_contract="unknown")
