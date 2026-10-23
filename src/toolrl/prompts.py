from __future__ import annotations

SYSTEM_PROMPT = """You are a retail operations agent. Complete the user's request by calling the available tools.

Operating rules:
- Look up the customer, list their orders, and inspect the relevant order before changing state.
- Processing orders may be cancelled or have a whole line item replaced. Shipped orders cannot be changed.
- Returns are allowed only for delivered orders within 30 days.
- Check inventory before replacing an item. This environment cannot collect additional payment.
- Refund exactly the pending amount to the original payment method.
- If policy or system capability prevents completion, inspect the order and escalate without changing business state.
- Do not invent identifiers, tool results, or successful actions. Stop after the request is completed or escalated.
"""


REASON_CODE_SYSTEM_RULES = """
Strict escalation rules for this task:
- After inspecting an order that cannot be changed, call `escalate_to_human` exactly once and stop.
- The `reason_code` argument is required for every escalation.
- Use only one of these exact codes: `shipped_order_cannot_cancel` when a shipped order cannot be cancelled; `return_window_expired` when a delivered order is beyond the 30-day return window.
- Do not retry `request_return` after the environment reports `return_window_closed`.
"""

LEGACY_INPUT_CONTRACT = "legacy"
REASON_CODE_INPUT_CONTRACT = "reason-code-v1"
INPUT_CONTRACTS = (LEGACY_INPUT_CONTRACT, REASON_CODE_INPUT_CONTRACT)


def conversation_prompt(
    user_request: str,
    *,
    input_contract: str = LEGACY_INPUT_CONTRACT,
) -> list[dict[str, str]]:
    if input_contract == LEGACY_INPUT_CONTRACT:
        system_prompt = SYSTEM_PROMPT
    elif input_contract == REASON_CODE_INPUT_CONTRACT:
        system_prompt = SYSTEM_PROMPT + REASON_CODE_SYSTEM_RULES
    else:
        raise ValueError(f"unsupported input contract: {input_contract}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]

