from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.generator import FAMILIES, generate_tasks
from toolrl.io import write_jsonl
from toolrl.rollout import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU demonstration with a scripted expert, not a model"
    )
    parser.add_argument("--family", choices=FAMILIES, default="cancel_processing")
    parser.add_argument("--inject-format-error", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    task = next(t for t in generate_tasks({"train": 5}) if t.family == args.family)
    plan = list(task.expert_plan)
    if args.inject_format_error:
        index = next(
            i
            for i, step in enumerate(plan)
            if step["tool"]
            in {"cancel_order", "replace_order_item", "request_return", "escalate_to_human"}
        )
        step = plan[index]
        plan.insert(
            index,
            {"tool": step["tool"], "arguments": {**step["arguments"], "unexpected_argument": 1}},
        )
    iterator = iter(plan)

    def scripted_expert(messages, tools):
        step = next(iterator, None)
        if step is None:
            return {"role": "assistant", "content": "Finished."}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": step["tool"], "arguments": step["arguments"]}}],
        }

    result = run_episode(task, scripted_expert)
    print("Policy: scripted expert (environment demonstration; not model evaluation)")
    print(f"Request: {task.prompt}")
    for index, step in enumerate(result["trace"], 1):
        status = "OK" if step["success"] else f"ERROR: {step['result'].get('error')}"
        print(f"{index}. {step['tool']}({json.dumps(step['arguments'])}) -> {status}")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "success",
                    "terminal_goal_satisfied",
                    "format_recovery_success",
                    "steps",
                )
            },
            indent=2,
        )
    )
    if args.output:
        write_jsonl(args.output, [result])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
