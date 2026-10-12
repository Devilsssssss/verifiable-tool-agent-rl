from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

from .diagnostics import FRAMES
from .schemas import AgentTask


def training_variants(task: AgentTask) -> list[AgentTask]:
    """Training-only 4x5 crossing; probe-only subscriber prefix is never used."""
    if task.split != "train":
        raise ValueError("factorial augmentation requires training tasks")
    digest = hashlib.sha256(f"factorial-v9:45:{task.task_id}".encode()).hexdigest()
    word = "".join(chr(ord("a") + int(char, 16)) for char in digest[:10])
    number = int(digest[10:18], 16) % 10000
    original = task.initial_state["users"][0]["email"]
    emails = (
        original,
        f"customer.{number:04d}@example.test",
        f"member.{number + 800000000}@example.test",
        f"client.{digest[:10]}@example.test",
        f"{word}.{number:04d}@example.test",
    )
    products = task.initial_state["products"]
    old, new = products[0]["name"], products[1]["name"]
    if task.family.startswith("cancel_"):
        request = f"Please cancel the purchase of {old} and refund my payment."
    elif task.family.startswith("return_"):
        request = f"Please return {old} and refund my payment."
    else:
        request = f"Please replace {old} with {new} and refund the difference."

    def rewrite(value, email):
        if isinstance(value, str):
            return email if value == original else value
        if isinstance(value, dict):
            return {key: rewrite(item, email) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rewrite(item, email) for item in value]
        return value

    return [
        replace(
            task,
            task_id=f"{task.task_id}-factorial-{frame_index}-{structure}",
            prompt=frame.format(email=email, request=request),
            initial_state=rewrite(deepcopy(task.initial_state), email),
            expert_plan=tuple(rewrite(task.expert_plan, email)),
            template_id=f"factorial-v9-frame-{frame_index}-email-{structure}",
        )
        for frame_index, frame in enumerate(FRAMES)
        for structure, email in enumerate(emails)
    ]
