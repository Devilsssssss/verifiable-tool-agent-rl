from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .position_shift import create_balanced_position_tasks, target_order_position
from .schemas import AgentTask

RETURN_FAMILIES = ("return_eligible", "return_expired")
RETURN_REQUEST = (
    "Customer contact: {email}. I would like to send back {old_name}; "
    "please process the request under your return policy."
)
TRAIN_DAYS = {"return_eligible": (2, 14, 28), "return_expired": (33, 46, 75)}
HOLDOUT_DAYS = {"return_eligible": (1, 29, 30), "return_expired": (31, 32, 60)}
DISTRACTOR_DAYS = (8, 45)


def vary_return_days(
    tasks: list[AgentTask], *, seed: int, days: dict[str, tuple[int, ...]],
    distractor_days: tuple[int, ...] = DISTRACTOR_DAYS,
) -> list[AgentTask]:
    """Balance target/distractor delivery ages separately in each family/position cell."""
    result = list(tasks)
    rng = random.Random(seed)
    if not distractor_days or any(day < 0 for day in distractor_days):
        raise ValueError("distractor delivery ages must be nonnegative and nonempty")
    for family in RETURN_FAMILIES:
        eligible = family == "return_eligible"
        if not days[family] or any((day <= 30) != eligible or day < 0 for day in days[family]):
            raise ValueError(f"delivery ages disagree with family: {family}")
        cells = [(target, distractor) for target in days[family] for distractor in distractor_days]
        for position in (1, 2):
            indices = [
                index for index, task in enumerate(tasks)
                if task.family == family and target_order_position(task) == position
            ]
            if not indices or len(indices) % len(cells):
                raise ValueError("each return family/position count must be a multiple of age cells")
            assignments = cells * (len(indices) // len(cells))
            rng.shuffle(assignments)
            for index, (target_days, distractor_age) in zip(indices, assignments, strict=True):
                task = tasks[index]
                state = deepcopy(task.initial_state)
                for order in state["orders"]:
                    if order["id"] == task.goal["order_id"]:
                        order["days_since_delivery"] = target_days
                    else:
                        order["days_since_delivery"] = distractor_age
                result[index] = replace(task, initial_state=state)
    return result


def build_return_holdout(
    tasks: list[AgentTask], *, seed: int, id_offset: int, output_split: str = "return-holdout"
) -> list[AgentTask]:
    """Create fresh, position-balanced eligible/expired return tasks.

    Source tasks provide only the business-state templates. All visible entities and
    dependent identifiers are replaced, and the target position is randomized after
    replacement. This is an evaluation-only transformation; no model output is used.
    """
    if not tasks:
        raise ValueError("source tasks must be nonempty")
    if id_offset <= 0:
        raise ValueError("id_offset must be positive")
    if any(task.family not in RETURN_FAMILIES for task in tasks):
        raise ValueError("source tasks must contain only return families")
    counts = {family: sum(task.family == family for task in tasks) for family in RETURN_FAMILIES}
    if counts[RETURN_FAMILIES[0]] != counts[RETURN_FAMILIES[1]] or counts[RETURN_FAMILIES[0]] % 2:
        raise ValueError("return families must have equal, even counts")

    transformed = []
    for index, task in enumerate(tasks):
        digest = hashlib.sha256(f"return-holdout:{seed}:{task.task_id}".encode()).hexdigest()
        code = digest[:10]
        rng = random.Random(int(digest, 16))
        products = task.initial_state["products"]
        brand = rng.choice(("Cedar", "Harbor", "Lumen", "Orbit", "Quartz", "Willow"))
        email = f"return.{code}@heldout.test"
        strings = {
            task.initial_state["users"][0]["email"]: email,
            products[0]["name"]: f"{brand} Headphones {code}",
            products[1]["name"]: f"{brand} Earbuds {code}",
            products[2]["name"]: f"{brand} Charger {code}",
        }
        identifiers = {
            row["id"]: row["id"] + id_offset
            for table in ("users", "products", "orders")
            for row in task.initial_state[table]
        }
        prompt = RETURN_REQUEST.format(
            email=email,
            old_name=strings[products[0]["name"]],
        )
        transformed.append(
            replace(
                task,
                task_id=f"return-holdout-{task.family}-{index:04d}",
                split=output_split,
                prompt=prompt,
                initial_state=_transform(task.initial_state, strings, identifiers),
                expert_plan=tuple(_transform(task.expert_plan, strings, identifiers)),
                goal=_transform(task.goal, strings, identifiers),
                template_id=f"return-holdout-{task.family}-v1",
            )
        )
    balanced = create_balanced_position_tasks(transformed, seed=seed + 1, output_split=output_split)
    return vary_return_days(balanced, seed=seed + 2, days=HOLDOUT_DAYS)


def _transform(value: Any, strings: dict[str, str], identifiers: dict[int, int]) -> Any:
    if isinstance(value, dict):
        return {key: _transform(item, strings, identifiers) for key, item in value.items()}
    if isinstance(value, list):
        return [_transform(item, strings, identifiers) for item in value]
    if isinstance(value, tuple):
        return tuple(_transform(item, strings, identifiers) for item in value)
    if isinstance(value, str):
        return strings.get(value, value)
    if isinstance(value, int):
        return identifiers.get(value, value)
    return deepcopy(value)
