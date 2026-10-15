from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.io import load_tasks, read_jsonl
from toolrl.position_shift import target_order_position


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in result:
            raise ValueError(f"duplicate task ID in {label}: {task_id}")
        result[task_id] = row
    if not result:
        raise ValueError(f"{label} is empty")
    return result


def _target_age(task: Any) -> int | None:
    target_id = task.goal["order_id"]
    for order in task.initial_state["orders"]:
        if order["id"] == target_id:
            value = order["days_since_delivery"]
            return None if value is None else int(value)
    raise ValueError(f"target order not found: {task.task_id}")


def _signature(row: dict[str, Any]) -> str:
    violations = tuple(sorted(set(row.get("violations", []))))
    return "+".join(violations) if violations else "none"


def _group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    candidate_failures = [item for item in items if not item["candidate_success"]]
    return {
        "tasks": count,
        "baseline_successes": sum(item["baseline_success"] for item in items),
        "candidate_successes": sum(item["candidate_success"] for item in items),
        "gained_tasks": sum(item["gained"] for item in items),
        "lost_tasks": sum(item["lost"] for item in items),
        "candidate_failed_with_correct_terminal_state": sum(
            item["candidate_terminal"] for item in candidate_failures
        ),
        "candidate_failed_without_target_inspection": sum(
            not item["candidate_target_inspected"] for item in candidate_failures
        ),
        "candidate_failure_signatures": dict(
            sorted(Counter(item["candidate_signature"] for item in candidate_failures).items())
        ),
    }


def analyze_failure_deltas(
    tasks: list[Any], baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    task_index = {task.task_id: task for task in tasks}
    if not task_index:
        raise ValueError("tasks is empty")
    baseline_index = _index(baseline, "baseline")
    candidate_index = _index(candidate, "candidate")
    expected = set(task_index)
    if set(baseline_index) != expected or set(candidate_index) != expected:
        raise ValueError("tasks and both runs must contain identical task IDs")

    rows: list[dict[str, Any]] = []
    for task_id, task in task_index.items():
        before = baseline_index[task_id]
        after = candidate_index[task_id]
        baseline_success = bool(before["success"])
        candidate_success = bool(after["success"])
        candidate_terminal = bool(after.get("terminal_goal_satisfied", False))
        target_inspected = bool(after.get("target_order_inspected", False))
        rows.append(
            {
                "task_id": task_id,
                "family": task.family,
                "position": target_order_position(task),
                "target_age": _target_age(task),
                "baseline_success": baseline_success,
                "candidate_success": candidate_success,
                "gained": not baseline_success and candidate_success,
                "lost": baseline_success and not candidate_success,
                "candidate_terminal": candidate_terminal,
                "candidate_target_inspected": target_inspected,
                "candidate_signature": _signature(after),
                "candidate_tools": [step["tool"] for step in after.get("trace", [])],
            }
        )

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata["overall"].append(row)
        strata[f"{row['family']}/position_{row['position']}"].append(row)
        strata[f"{row['family']}/target_age_{row['target_age']}"].append(row)

    candidates = {
        "gained": [row for row in rows if row["gained"]],
        "lost": [row for row in rows if row["lost"]],
        "candidate_failed_with_correct_terminal_state": [
            row for row in rows if not row["candidate_success"] and row["candidate_terminal"]
        ],
        "candidate_failed_without_target_inspection": [
            row for row in rows if not row["candidate_success"] and not row["candidate_target_inspected"]
        ],
    }
    examples = {
        name: sorted(items, key=lambda row: row["task_id"])[:8]
        for name, items in candidates.items()
    }
    return {
        "overall": _group_summary(rows),
        "by_stratum": {
            name: _group_summary(items) for name, items in sorted(strata.items())
        },
        "failure_examples": examples,
        "scope": (
            "Descriptive task-level failure analysis. It summarizes saved model outputs and "
            "does not select training data, checkpoints, or hyperparameters."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze task-level failure deltas between two runs")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_failure_deltas(
        load_tasks(args.tasks), read_jsonl(args.baseline), read_jsonl(args.candidate)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
