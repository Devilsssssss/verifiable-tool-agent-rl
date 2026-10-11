from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import RetailToolEnvironment
from toolrl.expert import replay_expert, reset_environment
from toolrl.io import load_tasks
from toolrl.metrics import summarize_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="run deterministic environment baselines")
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "data" / "generated" / "tasks_test.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "environment_baselines_summary.json"
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    expert_reports = []
    noop_reports = []
    for task in tasks:
        expert = replay_expert(task)
        expert_reports.append({"family": task.family, **expert["reward_report"]})
        environment = RetailToolEnvironment()
        reset_environment(environment, task)
        noop_reports.append({"family": task.family, **environment.reward_report})
    report = {
        "tasks": len(tasks),
        "expert": summarize_reports(expert_reports),
        "noop": summarize_reports(noop_reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

