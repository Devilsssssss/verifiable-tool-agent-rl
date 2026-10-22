from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POSITION_RE = re.compile(r"target-position-(?P<position>[12])$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _position(task_id: str) -> str:
    match = POSITION_RE.search(task_id)
    return match.group("position") if match else "unknown"


def _task_index(path: Path) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in _read_jsonl(path)}


def _failure_categories(record: dict[str, Any], task: dict[str, Any]) -> list[str]:
    violations = {step.get("violation") for step in record.get("trace", [])}
    categories: list[str] = []
    if task["family"] == "return_expired" and "return_window_closed" in violations:
        categories.append("expired_retry_request_return")
    if "escalation_reason_code_mismatch" in violations:
        categories.append("wrong_or_missing_reason_code")
    if "step_limit" in violations:
        categories.append("step_limit_after_failure")
    if not record.get("target_order_inspected", False):
        categories.append("missing_target_order_inspection")
    if task["family"] == "return_eligible" and not record.get("success", False):
        categories.append("eligible_return_regression")
    return categories


def _reason_code_values(record: dict[str, Any]) -> Counter[str]:
    values: Counter[str] = Counter()
    for step in record.get("trace", []):
        if step.get("tool") != "escalate_to_human":
            continue
        value = step.get("arguments", {}).get("reason_code")
        values[str(value) if value is not None else "<missing>"] += 1
    return values


def analyze(tasks_path: Path, run_paths: list[Path]) -> dict[str, Any]:
    tasks = _task_index(tasks_path)
    runs: list[dict[str, Any]] = []
    aggregate_categories: Counter[str] = Counter()
    aggregate_violations: Counter[str] = Counter()
    aggregate_reason_codes: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_position: dict[str, Counter[str]] = defaultdict(Counter)

    for run_path in run_paths:
        records = _read_jsonl(run_path)
        categories: Counter[str] = Counter()
        violations: Counter[str] = Counter()
        reason_codes: Counter[str] = Counter()
        family_stats: dict[str, Counter[str]] = defaultdict(Counter)
        position_stats: dict[str, Counter[str]] = defaultdict(Counter)
        failed = 0
        for record in records:
            task = tasks[record["task_id"]]
            family = task["family"]
            position = _position(task["task_id"])
            success = bool(record.get("success", False))
            family_stats[family]["tasks"] += 1
            position_stats[position]["tasks"] += 1
            if success:
                family_stats[family]["successes"] += 1
                position_stats[position]["successes"] += 1
                continue
            failed += 1
            family_stats[family]["failures"] += 1
            position_stats[position]["failures"] += 1
            for category in _failure_categories(record, task):
                categories[category] += 1
                aggregate_categories[category] += 1
            for step in record.get("trace", []):
                violation = step.get("violation")
                if violation:
                    violations[violation] += 1
                    aggregate_violations[violation] += 1
            for value, count in _reason_code_values(record).items():
                reason_codes[value] += count
                aggregate_reason_codes[value] += count

        for family, stats in family_stats.items():
            for key, value in stats.items():
                by_family[family][key] += value
        for position, stats in position_stats.items():
            for key, value in stats.items():
                by_position[position][key] += value
        runs.append(
            {
                "run": str(run_path),
                "tasks": len(records),
                "failures": failed,
                "successes": len(records) - failed,
                "failure_rate": failed / len(records) if records else None,
                "failure_categories": dict(categories),
                "violations": dict(violations),
                "reason_code_arguments": dict(reason_codes),
                "by_family": {key: dict(value) for key, value in family_stats.items()},
                "by_position": {key: dict(value) for key, value in position_stats.items()},
            }
        )

    return {
        "version": "grpo-failure-analysis-v1",
        "tasks": str(tasks_path),
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "runs": runs,
        "aggregate": {
            "failure_categories": dict(aggregate_categories),
            "violations": dict(aggregate_violations),
            "reason_code_arguments": dict(aggregate_reason_codes),
            "by_family": {key: dict(value) for key, value in by_family.items()},
            "by_position": {key: dict(value) for key, value in by_position.items()},
        },
        "interpretation": {
            "primary_bottleneck": "return_expired",
            "mechanism": (
                "The boundary-v1 SFT initialization was trained without machine-readable "
                "reason_code arguments, while strict-v3 requires an exact reason_code that "
                "is not exposed in the natural-language prompt."
            ),
            "next_experiment": (
                "Expose the reason-code contract in the system/tool prompt and run a matched "
                "two-seed SFT repair before attempting another RL experiment."
            ),
            "do_not_do": (
                "Do not relax strict-v3 scoring or treat repeated request_return calls as "
                "progress; both would hide the actual policy failure."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify final GRPO rollout failures")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.tasks, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
