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
from toolrl.io import load_tasks, write_jsonl
from toolrl.position_shift import create_balanced_position_probe, target_order_position


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a balanced order-position generalization probe"
    )
    parser.add_argument("--input", type=Path, default=ROOT / "data/confirmation/tasks_test.jsonl")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/position-shifted-tasks.jsonl"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts/position-shifted-tasks.summary.json"
    )
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()

    tasks = load_tasks(args.input)
    probe = create_balanced_position_probe(tasks, seed=args.seed)
    audit = audit_tasks(probe)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, (task.to_dict() for task in probe))

    position_counts = Counter(target_order_position(task) for task in probe)
    family_positions: dict[str, Counter[int]] = defaultdict(Counter)
    for task in probe:
        family_positions[task.family][target_order_position(task)] += 1

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "tasks": len(probe),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "position_counts": {str(key): value for key, value in sorted(position_counts.items())},
        "family_position_counts": {
            family: {str(key): value for key, value in sorted(counts.items())}
            for family, counts in sorted(family_positions.items())
        },
        "semantic_audit": audit,
        "transformation": (
            "stratified random assignment of target order to list position 1 or 2; "
            "swap order IDs and every dependent reference together"
        ),
        "scope": (
            "Held-out position generalization probe; do not use outcomes for training, "
            "prompt changes, or checkpoint selection."
        ),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
