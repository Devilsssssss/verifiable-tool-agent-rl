from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Any

from .schemas import AgentTask

# These are training-only templates, not copies of validation/test requests.
TRAINING_REQUESTS = {
    "cancel_processing": (
        "My login email is {email}. I would like to cancel {old_name} and receive my money back.",
        "Customer email: {email}\nRequest: cancel the purchase of {old_name} and refund the payment.",
        "Please undo the {old_name} purchase and refund me. My email address is {email}.",
    ),
    "cancel_shipped": (
        "My login email is {email}. I would like to cancel {old_name} and receive my money back.",
        "Customer email: {email}\nRequest: cancel the purchase of {old_name} and refund the payment.",
        "Please undo the {old_name} purchase and refund me. My email address is {email}.",
    ),
    "exchange_cheaper": (
        "My login email is {email}. Change {old_name} to {new_name} and refund the price difference.",
        "Customer email: {email}\nRequest: replace {old_name} with {new_name}; refund the difference.",
        "I prefer {new_name} instead of {old_name}. Please refund any difference. Contact: {email}.",
    ),
    "return_eligible": (
        "My login email is {email}. Please accept a return of {old_name} and refund my payment.",
        "Customer email: {email}\nRequest: return {old_name} and get the payment refunded.",
        "I no longer want {old_name}. Please process a return with a refund. Contact: {email}.",
    ),
    "return_expired": (
        "My login email is {email}. Please accept a return of {old_name} and refund my payment.",
        "Customer email: {email}\nRequest: return {old_name} and get the payment refunded.",
        "I no longer want {old_name}. Please process a return with a refund. Contact: {email}.",
    ),
}


def diversify_training_task(task: AgentTask, *, seed: int = 45) -> AgentTask:
    """Vary visible training entities and wording without touching held-out tasks.

    IDs, prices, state, goal and action budget stay fixed. One transformed task
    replaces one source task; this does not manufacture additional independence.
    """
    if task.split != "train":
        raise ValueError("augmentation is restricted to training tasks")
    digest = hashlib.sha256(f"{seed}:{task.task_id}".encode()).hexdigest()
    rng = random.Random(int(digest, 16))
    code = digest[:10]
    names = ("mina", "alex", "chen", "river", "kai", "pat")
    stem = rng.choice(names)
    email = rng.choice((f"{stem}.{code}", f"{stem}-{code}", f"{code}.{stem}"))
    email += "@" + rng.choice(("mailbox.test", "shopmail.test", "customer.test"))
    old_products = task.initial_state["products"]
    brand = rng.choice(("Orbit", "Lumen", "Quartz", "Willow", "Atlas", "Nova"))
    replacements = {
        task.initial_state["users"][0]["email"]: email,
        old_products[0]["name"]: f"{brand} Headphones {code}",
        old_products[1]["name"]: f"{brand} Earbuds {code}",
        old_products[2]["name"]: f"{brand} Charger {code}",
    }

    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: transform(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [transform(item) for item in value]
        if isinstance(value, str):
            return replacements.get(value, value)
        return value

    template_index = rng.randrange(len(TRAINING_REQUESTS[task.family]))
    prompt = TRAINING_REQUESTS[task.family][template_index].format(
        email=email,
        old_name=replacements[old_products[0]["name"]],
        new_name=replacements[old_products[1]["name"]],
    )
    return replace(
        task,
        task_id=f"{task.task_id}-diverse",
        prompt=prompt,
        initial_state=transform(task.initial_state),
        expert_plan=tuple(transform(task.expert_plan)),
        template_id=f"train-{task.family}-diverse-{template_index}",
    )
