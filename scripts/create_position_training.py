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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an equal-budget position-balanced variant of v8 SFT data"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/diverse-v8")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/position-v10")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--heldout", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "artifacts/position-v10-training-audit.summary.json",
    )
    args = parser.parse_args()

    source_tasks_path = args.data_dir / "tasks_train.jsonl"
    source_lookup_path = args.data_dir / "sft_lookup_train.jsonl"
    source_tasks = load_tasks(source_tasks_path)
    source_lookup = read_jsonl(source_lookup_path)
    if len(source_lookup) != len(source_tasks):
        raise ValueError("lookup view count must match independent training task count")

    tasks = create_balanced_position_tasks(
        source_tasks, seed=args.seed, output_split="train"
    )
    heldout = [task for path in args.heldout for task in load_tasks(path)]
    audit = audit_tasks(tasks + heldout)
    normal = [training_row(task) for task in tasks]
    repairs = [repair_training_row(task) for task in tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tasks": args.output_dir / "tasks_train.jsonl",
        "normal": args.output_dir / "sft_train.jsonl",
        "repairs": args.output_dir / "sft_repair_train.jsonl",
        "lookup": args.output_dir / "sft_lookup_train.jsonl",
        "extra": args.output_dir / "sft_repair_lookup_train.jsonl",
    }
    write_jsonl(outputs["tasks"], (task.to_dict() for task in tasks))
    write_jsonl(outputs["normal"], normal)
    write_jsonl(outputs["repairs"], repairs)
    write_jsonl(outputs["lookup"], source_lookup)
    write_jsonl(outputs["extra"], repairs + source_lookup)

    position_counts = Counter(target_order_position(task) for task in tasks)
    family_positions: dict[str, Counter[int]] = defaultdict(Counter)
    for task in tasks:
        family_positions[task.family][target_order_position(task)] += 1
    lookup_copy_identical = outputs["lookup"].read_bytes() == source_lookup_path.read_bytes()
    if not lookup_copy_identical:
        raise AssertionError("lookup views changed during position balancing")

    summary = {
        "seed": args.seed,
        "independent_training_tasks": len(tasks),
        "normal_views": len(normal),
        "repair_views": len(repairs),
        "lookup_views": len(source_lookup),
        "total_views": len(normal) + len(repairs) + len(source_lookup),
        "position_counts": {str(key): value for key, value in sorted(position_counts.items())},
        "family_position_counts": {
            family: {str(key): value for key, value in sorted(counts.items())}
            for family, counts in sorted(family_positions.items())
        },
        "semantic_audit": audit,
        "lookup_views_byte_identical_to_v8": lookup_copy_identical,
        "source_sha256": {
            "tasks": _sha256(source_tasks_path),
            "lookup": _sha256(source_lookup_path),
        },
        "output_sha256": {name: _sha256(path) for name, path in outputs.items()},
        "scope": (
            "Development-driven SFT intervention. Only order position and dependent IDs change "
            "in normal/repair views; v8 lookup views are copied byte-for-byte."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
