from __future__ import annotations

import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .schemas import AgentTask


def target_order_position(task: AgentTask) -> int:
    """Return the target order's one-based position in ``list_orders`` output."""
    target_id = _target_order_id(task)
    order_ids = sorted(int(row["id"]) for row in task.initial_state["orders"])
    if target_id not in order_ids:
        raise ValueError(f"target order is absent from initial state: {task.task_id}")
    return order_ids.index(target_id) + 1


def set_target_order_position(
    task: AgentTask, position: int, *, output_split: str = "position-probe"
) -> AgentTask:
    """Place the target order at a requested rank without changing task semantics."""
    if position not in {1, 2}:
        raise ValueError("position must be 1 or 2")

    target_id = _target_order_id(task)
    order_rows = task.initial_state["orders"]
    order_ids = {int(row["id"]) for row in order_rows}
    distractors = sorted(order_ids - {target_id})
    if len(distractors) != 1:
        raise ValueError(f"expected one distractor order: {task.task_id}")
    distractor_id = distractors[0]
    current_position = target_order_position(task)

    if current_position == position:
        state = deepcopy(task.initial_state)
        goal = deepcopy(task.goal)
        expert_plan = tuple(deepcopy(step) for step in task.expert_plan)
    else:
        state = deepcopy(task.initial_state)
        for row in state["orders"]:
            row["id"] = _swap_id(int(row["id"]), target_id, distractor_id)
        for row in state["order_items"]:
            row["order_id"] = _swap_id(int(row["order_id"]), target_id, distractor_id)
        goal = _swap_order_refs(task.goal, target_id, distractor_id)
        expert_plan = tuple(
            _swap_order_refs(step, target_id, distractor_id) for step in task.expert_plan
        )

    return replace(
        task,
        task_id=f"{task.task_id}-target-position-{position}",
        split=output_split,
        initial_state=state,
        goal=goal,
        expert_plan=expert_plan,
        template_id=f"{task.template_id}-target-position-{position}",
    )


def create_balanced_position_probe(
    tasks: list[AgentTask], *, seed: int
) -> list[AgentTask]:
    """Assign half of every task family to each target-order position."""
    return create_balanced_position_tasks(tasks, seed=seed, output_split="position-probe")


def create_balanced_position_tasks(
    tasks: list[AgentTask], *, seed: int, output_split: str
) -> list[AgentTask]:
    """Assign half of every task family to each position for a named split."""
    if not tasks:
        raise ValueError("task collection is empty")
    if not output_split:
        raise ValueError("output_split must be nonempty")

    by_family: dict[str, list[int]] = defaultdict(list)
    for index, task in enumerate(tasks):
        by_family[task.family].append(index)

    positions: dict[int, int] = {}
    rng = random.Random(seed)
    for family, indices in sorted(by_family.items()):
        if len(indices) % 2:
            raise ValueError(f"family must contain an even number of tasks: {family}")
        shuffled = list(indices)
        rng.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        positions.update({index: 1 for index in shuffled[:midpoint]})
        positions.update({index: 2 for index in shuffled[midpoint:]})

    probe = [
        set_target_order_position(task, positions[index], output_split=output_split)
        for index, task in enumerate(tasks)
    ]
    counts = Counter((task.family, target_order_position(task)) for task in probe)
    for family, indices in by_family.items():
        expected = len(indices) // 2
        if counts[(family, 1)] != expected or counts[(family, 2)] != expected:
            raise AssertionError(f"position balancing failed for {family}")
    return probe


def shift_target_order_position(task: AgentTask) -> AgentTask:
    """Backward-compatible helper that moves the target order to position two."""
    return set_target_order_position(task, 2)


def _target_order_id(task: AgentTask) -> int:
    order_id = task.goal.get("order_id")
    if not isinstance(order_id, int):
        raise TypeError(f"task goal has no order_id: {task.task_id}")
    return order_id


def _swap_id(value: int, left: int, right: int) -> int:
    if value == left:
        return right
    if value == right:
        return left
    return value


def _swap_order_refs(value: Any, left: int, right: int, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            name: _swap_order_refs(item, left, right, key=name)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_swap_order_refs(item, left, right) for item in value]
    if isinstance(value, tuple):
        return tuple(_swap_order_refs(item, left, right) for item in value)
    if key == "order_id" and isinstance(value, int):
        return _swap_id(value, left, right)
    return value
