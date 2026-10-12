from __future__ import annotations

import hashlib

from .prompts import conversation_prompt
from .schemas import AgentTask

# Development-driven training augmentation. These are not evaluation templates.
EMAIL_FRAMES = (
    "Regarding {email}, {request}",
    "For customer {email}, {request}",
    "For my account at {email}: {request}",
    "The registered email address is {email}. {request}",
    "Account holder email: {email}; {request}",
    "Contact me at {email}. {request}",
    "{request} My registered email is {email}.",
    "{request}\nCustomer contact: {email}",
    "Email = {email}\n{request}",
    "I have an account ({email}). {request}",
    "Please help the customer whose email is {email}: {request}",
    "{request} The account identifier is the email {email}.",
)


def lookup_training_row(task: AgentTask, *, seed: int = 45, profile: str = "v6") -> dict:
    """Supervise just the first lookup, using entities from a training task."""
    if task.split != "train":
        raise ValueError("lookup curriculum only accepts training tasks")
    if profile not in {"v6", "v7"}:
        raise ValueError("unknown lookup curriculum profile")
    email = task.initial_state["users"][0]["email"]
    products = task.initial_state["products"]
    old_name, new_name = products[0]["name"], products[1]["name"]
    if task.family.startswith("cancel_"):
        request = f"Please cancel the purchase of {old_name} and refund my payment."
    elif task.family.startswith("return_"):
        request = f"Please return {old_name} and refund my payment."
    else:
        request = f"Please replace {old_name} with {new_name} and refund the difference."
    digest = hashlib.sha256(f"lookup:{seed}:{task.task_id}".encode()).hexdigest()
    frames = EMAIL_FRAMES
    if profile == "v7":
        # Targeted backfill after development validation. The generic phrase
        # overlaps validation and must not be called unseen wording any longer.
        frames = (
            "For account {email}, please help with this request: {request}",
            "For account {email}: {request}",
            "For account {email}\nRequest: {request}",
            "For account {email} - {request}",
        )
    frame = int(digest, 16) % len(frames)
    prompt = frames[frame].format(email=email, request=request)
    messages = conversation_prompt(prompt)
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "lookup_user", "arguments": {"email": email}},
                }
            ],
        }
    )
    return {
        "task_id": f"{task.task_id}-lookup",
        "family": task.family,
        "split": "train",
        "messages": messages,
        "lookup_frame": frame,
        "lookup_profile": profile,
    }
