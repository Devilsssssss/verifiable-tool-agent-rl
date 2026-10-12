from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import RetailToolEnvironment
from toolrl.expert import reset_environment
from toolrl.io import load_tasks, read_jsonl
from toolrl.metrics import summarize_reports


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay saved rollouts and audit final database state"
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {task.task_id: task for task in load_tasks(args.tasks)}
    records = read_jsonl(args.rollouts)
    assisted = [bool(record.get("oracle_first_lookup", False)) for record in records]
    if any(assisted) and not all(assisted):
        raise ValueError("cannot pool oracle-assisted and unaided rollouts")
    interventions = {record.get("input_intervention", "original") for record in records}
    if len(interventions) != 1:
        raise ValueError("cannot pool different input interventions")
    seen = set()
    reports = []
    wrong_email_tasks = 0
    for record in records:
        task = tasks[record["task_id"]]
        if task.task_id in seen:
            raise ValueError(f"duplicate rollout: {task.task_id}")
        seen.add(task.task_id)
        environment = RetailToolEnvironment()
        reset_environment(environment, task)
        for step in record["trace"]:
            if step["violation"] == "invalid_tool_call":
                environment._record_external_failure(
                    step["tool"], step["arguments"], step["result"]["error"]
                )
            else:
                getattr(environment, step["tool"])(**step["arguments"])
        if environment.trace != record["trace"]:
            raise ValueError(f"trace replay mismatch: {task.task_id}")
        report = environment.reward_report
        for key in ("success", "policy_compliant", "steps", "valid_tool_rate", "shaped"):
            if report[key] != record[key]:
                raise ValueError(f"original score changed for {task.task_id}: {key}")
        expected_email = task.initial_state["users"][0]["email"]
        wrong_email_tasks += any(
            step["tool"] == "lookup_user" and step["arguments"].get("email") != expected_email
            for step in record["trace"]
        )
        reports.append({"family": task.family, **report})
    summary = {
        "scoring_context": "oracle-assisted diagnostic" if any(assisted) else "unaided",
        "input_intervention": next(iter(interventions)),
        "rollouts": str(args.rollouts),
        "tasks": str(args.tasks),
        "replay_verified": len(reports),
        "original_scores_unchanged": True,
        "wrong_email_tasks": wrong_email_tasks,
        "definition": {
            "task_success": "Original strict success: correct terminal state and no violations.",
            "terminal_goal_completion": "Correct terminal state, independent of prior violations.",
            "format_recovery_rate": (
                "Correct terminal state with only invalid_tool_call violations, divided by "
                "episodes containing invalid_tool_call. Other policy violations remain disqualifying."
            ),
        },
        **summarize_reports(reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
