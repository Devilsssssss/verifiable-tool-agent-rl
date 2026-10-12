from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.curriculum import lookup_training_row
from toolrl.environment import RetailToolEnvironment
from toolrl.expert import reset_environment
from toolrl.io import load_tasks, read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Add training-only first-lookup supervision")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/diverse")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts/lookup-curriculum.summary.json"
    )
    args = parser.parse_args()
    source = args.data_dir / "tasks_train.jsonl"
    tasks = load_tasks(source)
    rows = [lookup_training_row(task, seed=args.seed) for task in tasks]
    for task, row in zip(tasks, rows, strict=True):
        env = RetailToolEnvironment()
        reset_environment(env, task)
        call = row["messages"][-1]["tool_calls"][0]["function"]
        result = json.loads(env.lookup_user(**call["arguments"]))
        if result.get("user", {}).get("id") != task.initial_state["users"][0]["id"]:
            raise ValueError(f"lookup curriculum fails replay: {task.task_id}")
    repairs = read_jsonl(args.data_dir / "sft_repair_train.jsonl")
    write_jsonl(args.data_dir / "sft_lookup_train.jsonl", rows)
    extra_path = args.data_dir / "sft_repair_lookup_train.jsonl"
    write_jsonl(extra_path, repairs + rows)
    summary = {
        "seed": args.seed,
        "independent_training_tasks": len(tasks),
        "normal_views": len(tasks),
        "repair_views": len(repairs),
        "lookup_views": len(rows),
        "total_views": len(tasks) + len(repairs) + len(rows),
        "lookup_replay_verified": len(rows),
        "frame_counts": dict(Counter(row["lookup_frame"] for row in rows)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "extra_sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
        "development_driven": True,
        "held_out_modified": False,
        "scope": "First-call supervision; does not add independent business tasks.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
