from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.audit import audit_tasks
from toolrl.generator import generate_tasks
from toolrl.io import load_tasks, write_jsonl
from toolrl.position_shift import target_order_position
from toolrl.return_holdout import RETURN_FAMILIES, build_return_holdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a fresh return-policy holdout")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--id-offset", type=int, default=50_000_000)
    parser.add_argument("--output", type=Path, default=ROOT / "data/return-holdout/tasks_test.jsonl")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts/return-holdout-audit.summary.json"
    )
    parser.add_argument("--development-dir", type=Path, default=ROOT / "data/generated")
    parser.add_argument(
        "--position-training", type=Path, default=ROOT / "data/generated/tasks_train.jsonl"
    )
    parser.add_argument("--heldout", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    generated = generate_tasks({"test": 300}, seed=args.seed)
    source = [task for task in generated if task.family in RETURN_FAMILIES]
    tasks = build_return_holdout(source, seed=args.seed, id_offset=args.id_offset)
    existing = [
        task
        for split in ("train", "validation", "test")
        for task in load_tasks(args.development_dir / f"tasks_{split}.jsonl")
    ]
    existing = _merge_unique(existing, load_tasks(args.position_training))
    for path in args.heldout:
        existing = _merge_unique(existing, load_tasks(path))
    audit = audit_tasks(existing + tasks)
    if args.output.exists() and load_tasks(args.output) != tasks:
        raise ValueError("refusing to overwrite a different frozen return holdout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, (task.to_dict() for task in tasks))
    positions = Counter(target_order_position(task) for task in tasks)
    family_positions = {
        family: Counter(target_order_position(task) for task in tasks if task.family == family)
        for family in RETURN_FAMILIES
    }
    summary = {
        "status": "frozen-before-model-evaluation",
        "seed": args.seed,
        "id_offset": args.id_offset,
        "source_generated_tasks": len(generated),
        "tasks": len(tasks),
        "family_counts": dict(Counter(task.family for task in tasks)),
        "position_counts": dict(sorted(positions.items())),
        "family_position_counts": {
            family: dict(sorted(counts.items())) for family, counts in family_positions.items()
        },
        "age_position_cells": dict(sorted(Counter(
            f"{task.family}/position_{target_order_position(task)}/target_{order['days_since_delivery']}"
            f"/distractor_{next(o['days_since_delivery'] for o in task.initial_state['orders'] if o['id'] != task.goal['order_id'])}"
            for task in tasks
            for order in task.initial_state['orders'] if order['id'] == task.goal['order_id']
        ).items())),
        "audit": audit,
        "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "selection_policy": "Freeze before evaluating any candidate adapter; do not tune on outcomes.",
        "scope": "Fresh return-policy templates, entities and IDs; target position balanced within each family.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _merge_unique(existing, additions):
    """Merge audit inputs while rejecting conflicting duplicate task IDs."""
    by_id = {task.task_id: task for task in existing}
    merged = list(existing)
    for task in additions:
        previous = by_id.get(task.task_id)
        if previous is None:
            by_id[task.task_id] = task
            merged.append(task)
        elif previous != task:
            raise ValueError(f"conflicting duplicate task_id detected: {task.task_id}")
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
