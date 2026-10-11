from __future__ import annotations

from toolrl.audit import audit_tasks
from toolrl.generator import FAMILIES, entity_tokens, generate_tasks


def test_generation_is_deterministic() -> None:
    counts = {"train": 10, "validation": 5, "test": 5}

    first = generate_tasks(counts, seed=42)
    second = generate_tasks(counts, seed=42)

    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]


def test_splits_have_disjoint_entities_and_all_families() -> None:
    tasks = generate_tasks({"train": 10, "validation": 10, "test": 10})
    grouped = {
        split: [task for task in tasks if task.split == split]
        for split in ("train", "validation", "test")
    }

    assert entity_tokens(grouped["train"]).isdisjoint(entity_tokens(grouped["validation"]))
    assert entity_tokens(grouped["train"]).isdisjoint(entity_tokens(grouped["test"]))
    assert entity_tokens(grouped["validation"]).isdisjoint(entity_tokens(grouped["test"]))
    assert {task.family for task in grouped["test"]} == set(FAMILIES)

    def ids(split: str, table: str) -> set[int]:
        return {
            int(row["id"])
            for task in grouped[split]
            for row in task.initial_state[table]
        }

    for table in ("users", "products", "orders"):
        assert ids("train", table).isdisjoint(ids("validation", table))
        assert ids("train", table).isdisjoint(ids("test", table))
        assert ids("validation", table).isdisjoint(ids("test", table))


def test_full_audit_rejects_shortcuts_and_accepts_experts() -> None:
    tasks = generate_tasks({"train": 5, "validation": 5, "test": 5})

    report = audit_tasks(tasks)

    assert report["status"] == "pass"
    assert report["expert_successes"] == 15
    assert report["noop_successes"] == 0
    assert report["direct_shortcut_successes"] == 0
    assert report["entity_overlap_across_splits"] == 0
