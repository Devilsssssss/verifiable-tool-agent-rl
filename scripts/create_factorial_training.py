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
from toolrl.expert import repair_training_row, training_row
from toolrl.factorial_training import training_variants
from toolrl.io import load_tasks, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded, balanced training-only factorial views")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    sources = load_tasks(args.source)
    if any(task.split != "train" for task in sources):
        raise ValueError("all augmentation sources must be training tasks")
    counts = Counter(task.family for task in sources)
    if len(sources) != 300 or len(counts) != 5 or set(counts.values()) != {60}:
        raise ValueError("this bounded protocol requires 300 tasks, 60 in each of five families")
    family_counts = Counter()
    lookup_counts = Counter()
    tasks = []
    lookup_tasks = []
    for task in sources:
        variants = training_variants(task)
        # Cycle within each family, so email structures do not encode business family.
        cell = family_counts[task.family] % len(variants)
        tasks.append(variants[cell])
        family_counts[task.family] += 1
        if lookup_counts[task.family] < 4:
            lookup_tasks.extend(variants)
            lookup_counts[task.family] += 1
    if len(family_counts) != 5 or any(count != 4 for count in lookup_counts.values()):
        raise ValueError("need at least four training tasks in each of the five families")
    heldout = [task for path in args.heldout for task in load_tasks(path)]
    if any(task.split == "train" for task in heldout):
        raise ValueError("heldout paths must not contain training tasks")
    # Deduplicate the subset also present in normal trajectories before task auditing.
    unique = {task.task_id: task for task in tasks + lookup_tasks}
    report = audit_tasks(list(unique.values()) + heldout)
    normal = [training_row(task) for task in tasks]
    repair = [repair_training_row(task) for task in tasks]
    lookup = []
    for task in lookup_tasks:
        # Replay every variant through the same expert before cutting to the first answer.
        row = training_row(task)
        row["messages"] = row["messages"][:3]
        row["task_id"] += "-lookup"
        lookup.append(row)
    write_jsonl(args.output_dir / "tasks_train.jsonl", (task.to_dict() for task in tasks))
    write_jsonl(args.output_dir / "sft_train.jsonl", normal)
    write_jsonl(args.output_dir / "sft_repair_lookup_train.jsonl", repair + lookup)
    report.update(
        {
            "profile": "factorial-v9",
            "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
            "independent_training_tasks": len(sources),
            "normal_views": len(normal),
            "repair_views": len(repair),
            "lookup_views": len(lookup),
            "training_views": len(normal) + len(repair) + len(lookup),
            "normal_cells": dict(Counter(f"{task.family}/{task.template_id}" for task in tasks)),
            "lookup_cells": dict(
                Counter(f"{task.family}/{task.template_id}" for task in lookup_tasks)
            ),
            "heldout_modified": False,
            "scope": "Development-driven SFT; 1000 views versus v8's 900, not equal-budget to v8.",
            "heldout_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.heldout
            },
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("normal_cells", "lookup_cells")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
