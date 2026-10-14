from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import SCORING_PROTOCOL, RetailToolEnvironment
from toolrl.expert import reset_environment
from toolrl.io import load_tasks, read_jsonl, write_jsonl
from toolrl.metrics import summarize_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore saved traces with the current scorer")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_summary_path = args.rollouts.with_suffix(".summary.json")
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    tasks = {task.task_id: task for task in load_tasks(args.tasks)}
    records = read_jsonl(args.rollouts)
    rescored = []
    changed_successes = 0
    for record in records:
        task = tasks[record["task_id"]]
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
        changed_successes += bool(record["success"]) != bool(report["success"])
        rescored.append(
            {
                **record,
                **report,
                "reward": environment.get_reward(),
                "scoring_protocol": SCORING_PROTOCOL,
            }
        )

    write_jsonl(args.output, rescored)
    summary = {
        **source_summary,
        "run_name": f'{source_summary["run_name"]}-strict-v2',
        "scoring_protocol": SCORING_PROTOCOL,
        "rescored_from": str(args.rollouts),
        "source_scoring_protocol": source_summary["scoring_protocol"],
        "changed_success_labels": changed_successes,
        **summarize_reports(rescored),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
