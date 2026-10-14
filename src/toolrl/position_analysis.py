from __future__ import annotations

from collections import defaultdict
from typing import Any

from .position_shift import target_order_position
from .schemas import AgentTask


def analyze_position_runs(
    tasks: list[AgentTask],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize paired model outcomes by target-order position and family."""
    task_index = _unique_index((task.task_id, task) for task in tasks)
    run_indexes = [
        _unique_index((str(row["task_id"]), row) for row in rows)
        for rows in (baseline, candidate)
    ]
    expected = set(task_index)
    if not expected or any(set(index) != expected for index in run_indexes):
        raise ValueError("tasks and both runs must contain identical nonempty task IDs")

    strata: dict[str, list[str]] = defaultdict(list)
    for task_id, task in task_index.items():
        position = target_order_position(task)
        strata["overall"].append(task_id)
        strata[f"position_{position}"].append(task_id)
        strata[f"{task.family}/position_{position}"].append(task_id)

    report = {
        "strata": {
            name: _paired_summary(task_ids, run_indexes[0], run_indexes[1])
            for name, task_ids in sorted(strata.items())
        },
        "position_effect": {
            "baseline_position_2_minus_1": _position_difference(
                strata, run_indexes[0]
            ),
            "candidate_position_2_minus_1": _position_difference(
                strata, run_indexes[1]
            ),
        },
        "scope": (
            "Descriptive stratified results from one deterministic decoding run. "
            "They measure task-level position sensitivity, not training-seed variability."
        ),
    }
    if all("trace" in row for index in run_indexes for row in index.values()):
        report["target_order_selection"] = {
            "baseline": _selection_summary(strata, task_index, run_indexes[0]),
            "candidate": _selection_summary(strata, task_index, run_indexes[1]),
        }
    return report


def _unique_index(items: Any) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for key, value in items:
        if key in index:
            raise ValueError(f"duplicate task ID: {key}")
        index[key] = value
    return index


def _paired_summary(
    task_ids: list[str],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs = [
        (bool(baseline[task_id]["success"]), bool(candidate[task_id]["success"]))
        for task_id in task_ids
    ]
    baseline_successes = sum(before for before, _ in pairs)
    candidate_successes = sum(after for _, after in pairs)
    count = len(pairs)
    return {
        "tasks": count,
        "baseline_successes": baseline_successes,
        "baseline_success_rate": baseline_successes / count,
        "candidate_successes": candidate_successes,
        "candidate_success_rate": candidate_successes / count,
        "gained_tasks": sum(not before and after for before, after in pairs),
        "lost_tasks": sum(before and not after for before, after in pairs),
        "success_rate_difference": (candidate_successes - baseline_successes) / count,
    }


def _position_difference(
    strata: dict[str, list[str]], run: dict[str, dict[str, Any]]
) -> float:
    rates = []
    for position in (1, 2):
        task_ids = strata[f"position_{position}"]
        rates.append(sum(bool(run[task_id]["success"]) for task_id in task_ids) / len(task_ids))
    return rates[1] - rates[0]


def _selection_summary(
    strata: dict[str, list[str]],
    tasks: dict[str, AgentTask],
    run: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for name in ("overall", "position_1", "position_2"):
        task_ids = strata[name]
        selected = sum(_selected_target(tasks[task_id], run[task_id]) for task_id in task_ids)
        result[name] = {
            "tasks": len(task_ids),
            "target_order_selections": selected,
            "target_order_selection_rate": selected / len(task_ids),
        }
    return result


def _selected_target(task: AgentTask, record: dict[str, Any]) -> bool:
    target_id = task.goal["order_id"]
    inspected = [
        step["arguments"].get("order_id")
        for step in record["trace"]
        if step["tool"] == "get_order" and step["success"]
    ]
    return target_id in inspected
