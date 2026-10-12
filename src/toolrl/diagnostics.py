"""Development diagnostics; these never replace unaided strict task evaluation."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from .rollout import ResponsePolicy
from .schemas import AgentTask

FRAMES = (
    "For account {email}: {request}",
    "My login email is {email}. {request}",
    "Customer email: {email}\nRequest: {request}",
    "{request} Contact email: {email}.",
)


def factorial_cases(task: AgentTask) -> list[dict[str, Any]]:
    """Fully cross frames and synthetic email structures on training states only."""
    if task.split != "train":
        raise ValueError("factorial diagnostics require training tasks")
    digest = hashlib.sha256(f"diagnostic:{task.task_id}".encode()).hexdigest()
    number = int(digest[:8], 16) % 10000
    emails = {
        "seen": task.initial_state["users"][0]["email"],
        "short_numeric": f"customer.{number}@example.test",
        "long_numeric": f"customer.{number + 100000000}@example.test",
        "hash": f"client.{digest[:10]}@example.test",
        "unseen_prefix": f"subscriber.{number:04d}@example.test",
    }
    products = task.initial_state["products"]
    old, new = products[0]["name"], products[1]["name"]
    if task.family.startswith("cancel_"):
        request = f"Please cancel the purchase of {old} and refund my payment."
    elif task.family.startswith("return_"):
        request = f"Please return {old} and refund my payment."
    else:
        request = f"Please replace {old} with {new} and refund the difference."
    return [
        {
            "frame": index,
            "email_structure": structure,
            "expected_email": email,
            "prompt": frame.format(email=email, request=request),
        }
        for index, frame in enumerate(FRAMES)
        for structure, email in emails.items()
    ]


def correct_lookup(response: dict[str, Any], email: str) -> bool:
    calls = response.get("tool_calls") or []
    if len(calls) != 1 or not isinstance(calls[0], dict):
        return False
    function = calls[0].get("function") or {}
    return function.get("name") == "lookup_user" and function.get("arguments") == {"email": email}


class OracleFirstLookup:
    """Inject exactly one public lookup; later calls remain entirely model generated."""

    def __init__(self, policy: ResponsePolicy, email: str) -> None:
        self.policy = policy
        self.email = email
        self.used = False

    def __call__(self, messages, tools) -> dict[str, Any]:
        if self.used:
            return self.policy(messages, tools)
        self.used = True
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "oracle_lookup",
                    "function": {"name": "lookup_user", "arguments": {"email": self.email}},
                }
            ],
        }


class ExplicitEmailPrompt:
    """Input-format diagnostic: expose only the email already written by the user."""

    def __init__(self, policy: ResponsePolicy) -> None:
        self.policy = policy

    def __call__(self, messages, tools) -> dict[str, Any]:
        changed = deepcopy(messages)
        request = changed[1]["content"]
        emails = re.findall(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", request)
        if len(emails) != 1:
            raise ValueError("explicit-field diagnostic requires one email visible in user text")
        changed[1]["content"] = f"Customer email: {emails[0]}\nRequest: {request}"
        return self.policy(changed, tools)
