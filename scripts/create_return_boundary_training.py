from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.audit import audit_tasks
from toolrl.expert import repair_training_row, training_row
from toolrl.io import load_tasks, read_jsonl, write_jsonl
from toolrl.position_shift import create_balanced_position_tasks, target_order_position
from toolrl.return_holdout import RETURN_FAMILIES, vary_return_days

BOUNDARY_DAYS = {
    "return_eligible": (29, 30),
    "return_expired": (31, 32),
}
BOUNDARY_DISTRACTOR_DAYS = (8,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an equal-budget return-policy boundary SFT intervention"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/return-boundary-v1")
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "artifacts/return-boundary-v1-training-audit.summary.json",
    )
    args = parser.parse_args()

    source_tasks_path = args.data_dir / "tasks_train.jsonl"
    source_tasks = load_tasks(source_tasks_path)

    return_indices = [
        index for index, task in enumerate(source_tasks) if task.family in RETURN_FAMILIES
    ]
    return_tasks = [source_tasks[index] for index in return_indices]
    family_counts = Counter(task.family for task in return_tasks)
    if set(family_counts) != set(RETURN_FAMILIES) or any(
        family_counts[family] != 60 for family in RETURN_FAMILIES
    ):
        raise ValueError(f"expected 60 tasks per return family, got {dict(family_counts)}")

    balanced = create_balanced_position_tasks(
        return_tasks, seed=args.seed, output_split="return-boundary-train"
    )
    boundary_tasks = vary_return_days(
        balanced,
        seed=args.seed + 1,
        days=BOUNDARY_DAYS,
        distractor_days=BOUNDARY_DISTRACTOR_DAYS,
    )
    tasks = list(source_tasks)
    for index, task in zip(return_indices, boundary_tasks, strict=True):
        tasks[index] = task

    heldout = [task for path in args.heldout for task in load_tasks(path)]
    audit = audit_tasks(tasks + heldout)
    normal = [training_row(task) for task in tasks]
    repairs = [repair_training_row(task) for task in tasks]
    lookup = [_lookup_training_row(task) for task in tasks]
    source_normal = [training_row(task) for task in source_tasks]
    source_repairs = [repair_training_row(task) for task in source_tasks]
    for index, task in enumerate(tasks):
        if task.family not in RETURN_FAMILIES and (
            normal[index] != source_normal[index] or repairs[index] != source_repairs[index]
        ):
            raise ValueError(f"non-return training view changed at index {index}")

    outputs = {
        "tasks": args.output_dir / "tasks_train.jsonl",
        "normal": args.output_dir / "sft_train.jsonl",
        "repairs": args.output_dir / "sft_repair_train.jsonl",
        "lookup": args.output_dir / "sft_lookup_train.jsonl",
        "extra": args.output_dir / "sft_repair_lookup_train.jsonl",
    }
    rows = {
        "tasks": [task.to_dict() for task in tasks],
        "normal": normal,
        "repairs": repairs,
        "lookup": lookup,
        "extra": repairs + lookup,
    }
    for key, path in outputs.items():
        if path.exists() and read_jsonl(path) != rows[key]:
            raise ValueError(f"refusing to overwrite different frozen training data: {key}")
        write_jsonl(path, rows[key])

    boundary_family_position: dict[str, Counter[int]] = defaultdict(Counter)
    boundary_age_cells: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    for task in boundary_tasks:
        position = target_order_position(task)
        boundary_family_position[task.family][position] += 1
        target = next(
            row["days_since_delivery"]
            for row in task.initial_state["orders"]
            if row["id"] == task.goal["order_id"]
        )
        distractor = next(
            row["days_since_delivery"]
            for row in task.initial_state["orders"]
            if row["id"] != task.goal["order_id"]
        )
        boundary_age_cells[task.family][(target, distractor)] += 1

    summary = {
        "status": "frozen-before-training",
        "seed": args.seed,
        "source": str(source_tasks_path),
        "source_sha256": {
            "tasks": _sha256(source_tasks_path),
        },
        "independent_training_tasks": len(tasks),
        "return_boundary_tasks": len(boundary_tasks),
        "normal_views": len(normal),
        "repair_views": len(repairs),
        "lookup_views": len(lookup),
        "total_views": len(normal) + len(repairs) + len(lookup),
        "return_family_counts": dict(sorted(family_counts.items())),
        "return_family_position_counts": {
            family: {str(position): count for position, count in sorted(counts.items())}
            for family, counts in sorted(boundary_family_position.items())
        },
        "return_age_cells": {
            family: {
                f"target={target},distractor={distractor}": count
                for (target, distractor), count in sorted(cells.items())
            }
            for family, cells in sorted(boundary_age_cells.items())
        },
        "target_age_schedule": BOUNDARY_DAYS,
        "distractor_age_schedule": BOUNDARY_DISTRACTOR_DAYS,
        "semantic_audit": audit,
        "lookup_views_regenerated_from_same_tasks": True,
        "nonreturn_views_unchanged": True,
        "output_sha256": {name: _sha256(path) for name, path in outputs.items()},
        "intervention": (
            "Only return-task target position and target/distractor delivery ages change; "
            "all 300 tasks, 900 views, prompts, and lookup views remain in budget."
        ),
        "scope": (
            "Independent supervised boundary comparison around the 30-day return cutoff; "
            "not an RL or GRPO experiment. Do not start GRPO until this comparison and "
            "the target-order gate are resolved."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lookup_training_row(task) -> dict:
    """Keep the first lookup call as a separately supervised training view."""
    row = training_row(task)
    row["task_id"] = f"{task.task_id}-lookup"
    row["messages"] = row["messages"][:3]
    row["lookup_frame"] = len(row["messages"])
    return row


if __name__ == "__main__":
    raise SystemExit(main())
