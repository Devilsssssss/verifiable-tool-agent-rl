from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import REASON_CODE_SCORING_PROTOCOL, RetailToolEnvironment
from toolrl.expert import grpo_row, reset_environment
from toolrl.io import load_tasks, write_jsonl


def strict_reason_code_task(task):
    if task.family not in {"cancel_shipped", "return_expired"}:
        return task
    reason_code = (
        "shipped_order_cannot_cancel"
        if task.family == "cancel_shipped"
        else "return_window_expired"
    )
    goal = dict(task.goal)
    goal["reason_code"] = reason_code
    plan = []
    for step in task.expert_plan:
        updated = {"tool": step["tool"], "arguments": dict(step["arguments"])}
        if updated["tool"] == "escalate_to_human":
            updated["arguments"]["reason_code"] = reason_code
        plan.append(updated)
    return replace(task, goal=goal, expert_plan=tuple(plan))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze audited boundary-SFT tasks as the final GRPO training set"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/return-boundary-v1/tasks_train.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/grpo-final-v1/train.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "artifacts/grpo-final-v1-training-audit.summary.json",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.source)
    allowed_splits = {"train", "return-boundary-train"}
    if not tasks or any(task.split not in allowed_splits for task in tasks):
        raise ValueError("final GRPO training data contains a non-training split")

    rows = []
    for raw_task in tasks:
        task = strict_reason_code_task(raw_task)
        environment = RetailToolEnvironment()
        reset_environment(
            environment,
            task,
            reward_mode="strict_shaped",
            scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
        )
        for step in task.expert_plan:
            getattr(environment, step["tool"])(**step["arguments"])
        report = environment.reward_report
        if not report["success"]:
            raise ValueError(f"expert plan failed strict-v3 audit for {task.task_id}: {report}")
        rows.append(
            grpo_row(
                task,
                reward_mode="strict_shaped",
                scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
            )
        )

    if args.output.exists():
        existing = args.output.read_text(encoding="utf-8")
        expected = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        if existing != expected:
            raise ValueError("refusing to overwrite a different frozen GRPO training file")
    write_jsonl(args.output, rows)
    summary = {
        "status": "frozen-before-grpo-training",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "tasks": len(tasks),
        "families": sorted({task.family for task in tasks}),
        "reward_mode": "strict_shaped",
        "scoring_protocol": REASON_CODE_SCORING_PROTOCOL,
        "expert_strict_successes": len(rows),
        "scope": (
            "Final two-seed GRPO attempt initialized from the audited return-boundary-v1 "
            "SFT adapter; holdout must be frozen separately before training."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
