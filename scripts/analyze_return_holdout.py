from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import SCORING_PROTOCOL
from toolrl.io import load_tasks, read_jsonl
from toolrl.metrics import summarize_reports
from toolrl.position_shift import target_order_position
from toolrl.return_holdout import RETURN_FAMILIES


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a frozen return-policy holdout run")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    task_index = {task.task_id: task for task in tasks}
    if len(task_index) != len(tasks):
        raise ValueError("task file contains duplicate task IDs")
    if any(task.family not in RETURN_FAMILIES for task in tasks):
        raise ValueError("return holdout must contain only return tasks")
    rows = read_jsonl(args.rollouts)
    _validate_rollouts(rows, task_index)
    source = json.loads(args.rollouts.with_suffix(".summary.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
    if source.get("tasks_sha256") != digest:
        raise ValueError("rollout summary tasks_sha256 does not match the supplied task file")
    for key in ("run_name", "scoring_protocol", "input_intervention"):
        if source.get(key) != rows[0][key]:
            raise ValueError(f"rollout summary and rows disagree on {key}")
    if source.get("overall", {}).get("tasks") != len(rows):
        raise ValueError("rollout summary and rows disagree on task count")

    reports = []
    wrong_email_tasks = 0
    for row in rows:
        task = task_index[str(row["task_id"])]
        expected_email = task.initial_state["users"][0]["email"]
        wrong_email = any(
            step["tool"] == "lookup_user"
            and step["arguments"].get("email") != expected_email
            for step in row["trace"]
        )
        wrong_email_tasks += wrong_email
        target_age = next(
            order["days_since_delivery"]
            for order in task.initial_state["orders"]
            if order["id"] == task.goal["order_id"]
        )
        reports.append(
            {
                **row,
                "family": task.family,
                "target_position": target_order_position(task),
                "target_age": target_age,
                "wrong_email": wrong_email,
            }
        )

    position_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    age_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reports:
        position_groups[f"{row['family']}/position_{row['target_position']}"].append(row)
        age_groups[f"{row['family']}/target_age_{row['target_age']}"].append(row)

    metrics = summarize_reports(reports)
    if any(source.get(key) != metrics[key] for key in ("overall", "by_family")):
        raise ValueError("rollout summary metrics do not match saved rows")
    failures = [row for row in reports if not row["success"]]

    summary = {
        "run_name": rows[0].get("run_name") if rows else None,
        "rollouts": len(rows),
        "tasks_sha256": digest,
        "rollouts_path": str(args.rollouts),
        "scoring_protocol": rows[0].get("scoring_protocol") if rows else None,
        "input_intervention": rows[0].get("input_intervention", "original") if rows else None,
        "wrong_email_tasks": wrong_email_tasks,
        **metrics,
        "failure_diagnostics": {
            "failed_tasks": len(failures),
            "failed_without_target_inspection": sum(
                not row["target_order_inspected"] for row in failures
            ),
            "failed_with_correct_terminal_state": sum(
                row["terminal_goal_satisfied"] for row in failures
            ),
            "failed_tasks_by_violation": dict(sorted(Counter(
                violation for row in failures for violation in set(row["violations"])
            ).items())),
        },
        "by_family_position": {
            key: _compact_metrics(value) for key, value in sorted(position_groups.items())
        },
        "by_family_target_age": {
            key: _compact_metrics(value) for key, value in sorted(age_groups.items())
        },
        "scope": (
            "Descriptive analysis of the frozen return holdout; see "
            "docs/return_boundary_protocol.md for experiment timing and interpretation limits."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _validate_rollouts(rows: list[dict[str, Any]], task_index: dict[str, Any]) -> None:
    if not rows:
        raise ValueError("rollouts must not be empty")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or "task_id" not in row:
            raise ValueError("every rollout must be an object with task_id")
        task_id = str(row["task_id"])
        ids.append(task_id)
        missing = {
            "success", "policy_compliant", "trace", "target_order_inspected",
            "terminal_goal_satisfied", "family",
        } - row.keys()
        if missing:
            raise ValueError(f"rollout {task_id} is missing fields: {sorted(missing)}")
        if not isinstance(row["trace"], list):
            raise TypeError(f"rollout {task_id} trace must be a list")
        if task_id in task_index and row["family"] != task_index[task_id].family:
            raise ValueError(f"rollout {task_id} family disagrees with task file")
    if len(ids) != len(set(ids)):
        raise ValueError("rollouts must contain each task ID exactly once")
    if set(task_index) != set(ids):
        missing = sorted(set(task_index) - set(ids))
        extra = sorted(set(ids) - set(task_index))
        raise ValueError(
            f"tasks and rollouts must contain identical task IDs; missing={missing}, extra={extra}"
        )

    for field in ("run_name", "scoring_protocol", "input_intervention"):
        values = {row.get(field, "<missing>") for row in rows}
        if len(values) != 1 or "<missing>" in values:
            raise ValueError(f"rollouts must use one complete {field} value")
    if rows[0]["scoring_protocol"] != SCORING_PROTOCOL:
        raise ValueError(f"return analysis requires scoring protocol {SCORING_PROTOCOL}")


def _compact_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "tasks": count,
        "task_successes": sum(bool(row["success"]) for row in rows),
        "task_success_rate": sum(bool(row["success"]) for row in rows) / count,
        "policy_compliance_rate": sum(bool(row["policy_compliant"]) for row in rows) / count,
        "target_inspection_rate": sum(bool(row["target_order_inspected"]) for row in rows)
        / count,
        "terminal_goal_completion_rate": sum(
            bool(row["terminal_goal_satisfied"]) for row in rows
        )
        / count,
        "wrong_email_tasks": sum(bool(row["wrong_email"]) for row in rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
