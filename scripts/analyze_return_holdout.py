from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.io import load_tasks, read_jsonl
from toolrl.metrics import summarize_reports
from toolrl.position_shift import target_order_position


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a frozen return-policy holdout run")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    task_index = {task.task_id: task for task in tasks}
    rows = read_jsonl(args.rollouts)
    if set(task_index) != {str(row["task_id"]) for row in rows}:
        raise ValueError("tasks and rollouts must contain identical task IDs")

    reports = []
    wrong_email_tasks = 0
    for row in rows:
        task = task_index[str(row["task_id"])]
        expected_email = task.initial_state["users"][0]["email"]
        wrong_email = any(
            step["tool"] == "lookup_user"
            and step["arguments"].get("email") != expected_email
            for step in row.get("trace", [])
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

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reports:
        groups["overall"].append(row)
        groups[f"{row['family']}/position_{row['target_position']}"].append(row)
        groups[f"{row['family']}/target_age_{row['target_age']}"] .append(row)

    summary = {
        "run_name": rows[0].get("run_name") if rows else None,
        "rollouts": len(rows),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "rollouts_path": str(args.rollouts),
        "scoring_protocol": rows[0].get("scoring_protocol") if rows else None,
        "input_intervention": rows[0].get("input_intervention", "original") if rows else None,
        "wrong_email_tasks": wrong_email_tasks,
        "overall": summarize_reports(reports)["overall"],
        "by_family": summarize_reports(reports)["by_family"],
        "by_family_position": {
            key: _compact_metrics(value) for key, value in sorted(groups.items()) if key != "overall"
        },
        "scope": (
            "Descriptive analysis of the frozen return holdout. No outcome was used to alter "
            "the holdout, prompts, decoding or training data."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _compact_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "tasks": count,
        "task_successes": sum(bool(row["success"]) for row in rows),
        "task_success_rate": sum(bool(row["success"]) for row in rows) / count,
        "policy_compliance_rate": sum(bool(row["policy_compliant"]) for row in rows) / count,
        "target_inspection_rate": sum(bool(row.get("target_order_inspected", False)) for row in rows)
        / count,
        "terminal_goal_completion_rate": sum(
            bool(row.get("terminal_goal_satisfied", False)) for row in rows
        )
        / count,
        "wrong_email_tasks": sum(bool(row["wrong_email"]) for row in rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
