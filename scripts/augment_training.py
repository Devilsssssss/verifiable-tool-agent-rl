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
from toolrl.augmentation import diversify_training_task
from toolrl.expert import grpo_row, repair_training_row, training_row
from toolrl.io import load_tasks, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diversify training requests, leaving evaluation fixed"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/diverse")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--profile", choices=("v5", "v8"), default="v5")
    parser.add_argument("--confirmation", type=Path, help="Also audit overlap; no model evaluation")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts/diverse-training-audit.summary.json"
    )
    args = parser.parse_args()
    source = args.data_dir / "tasks_train.jsonl"
    tasks = [
        diversify_training_task(task, seed=args.seed, profile=args.profile)
        for task in load_tasks(source)
    ]
    # Include the unchanged development splits when checking entity leakage.
    held_out = [
        task
        for split in ("validation", "test")
        for task in load_tasks(args.data_dir / f"tasks_{split}.jsonl")
    ]
    if args.confirmation:
        held_out.extend(load_tasks(args.confirmation))
    report = audit_tasks(tasks + held_out)
    write_jsonl(args.output_dir / "tasks_train.jsonl", (task.to_dict() for task in tasks))
    write_jsonl(args.output_dir / "sft_train.jsonl", (training_row(task) for task in tasks))
    write_jsonl(
        args.output_dir / "sft_repair_train.jsonl", (repair_training_row(task) for task in tasks)
    )
    for mode in ("sparse", "shaped"):
        write_jsonl(
            args.output_dir / f"grpo_train_{mode}.jsonl",
            (grpo_row(task, reward_mode=mode) for task in tasks),
        )
    report.update(
        {
            "augmentation_seed": args.seed,
            "augmentation_profile": args.profile,
            "email_domain_counts": dict(
                Counter(task.initial_state["users"][0]["email"].split("@")[1] for task in tasks)
            ),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_tasks": len(tasks),
            "independent_training_tasks": len(tasks),
            "training_views": 2 * len(tasks),
            "training_template_counts": dict(Counter(task.template_id for task in tasks)),
            "held_out_modified": False,
            "confirmation_audited": str(args.confirmation) if args.confirmation else None,
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
