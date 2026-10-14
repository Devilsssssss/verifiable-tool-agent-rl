from collections import Counter

import pytest

from toolrl.expert import replay_expert
from toolrl.generator import FAMILIES, generate_tasks
from toolrl.position_shift import (
    create_balanced_position_probe,
    set_target_order_position,
    target_order_position,
)


@pytest.mark.parametrize("family_index", range(len(FAMILIES)))
def test_position_shift_preserves_every_family_and_expert(family_index: int) -> None:
    task = generate_tasks({"test": len(FAMILIES)}, seed=42)[family_index]
    original = task.to_dict()
    shifted = set_target_order_position(task, 2)

    assert task.to_dict() == original
    assert shifted.split == "position-probe"
    assert shifted.prompt == task.prompt
    assert shifted.goal["order_id"] != task.goal["order_id"]
    assert shifted.goal["kind"] == task.goal["kind"]
    assert target_order_position(shifted) == 2
    assert replay_expert(shifted)["reward_report"]["success"] is True


def test_position_one_is_a_deep_copy_with_updated_identity() -> None:
    task = generate_tasks({"test": len(FAMILIES)}, seed=43)[0]
    positioned = set_target_order_position(task, 1)

    assert target_order_position(positioned) == 1
    assert positioned.goal == task.goal
    assert positioned.initial_state == task.initial_state
    assert positioned.initial_state is not task.initial_state
    assert positioned.task_id.endswith("target-position-1")


def test_balanced_probe_stratifies_each_family_and_is_reproducible() -> None:
    tasks = generate_tasks({"test": 20}, seed=44)
    first = create_balanced_position_probe(tasks, seed=20261014)
    second = create_balanced_position_probe(tasks, seed=20261014)
    counts = Counter((task.family, target_order_position(task)) for task in first)

    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    assert all(counts[(family, position)] == 2 for family in FAMILIES for position in (1, 2))


def test_balanced_probe_rejects_odd_family_counts() -> None:
    tasks = generate_tasks({"test": len(FAMILIES)}, seed=45)
    with pytest.raises(ValueError, match="even number"):
        create_balanced_position_probe(tasks, seed=20261014)
