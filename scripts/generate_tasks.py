from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from toolrl.expert import grpo_row, training_row
from toolrl.generator import generate_tasks
from toolrl.io import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="generate verifiable retail-agent tasks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--validation", type=int, default=60)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    if any(count <= 0 for count in counts.values()):
        raise ValueError("every split count must be positive")
    tasks = generate_tasks(counts, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in counts:
        split_tasks = [task for task in tasks if task.split == split]
        write_jsonl(args.output_dir / f"tasks_{split}.jsonl", (task.to_dict() for task in split_tasks))
        write_jsonl(
            args.output_dir / f"grpo_{split}_sparse.jsonl",
            (grpo_row(task, reward_mode="sparse") for task in split_tasks),
        )
        write_jsonl(
            args.output_dir / f"grpo_{split}_shaped.jsonl",
            (grpo_row(task, reward_mode="shaped") for task in split_tasks),
        )
        if split != "test":
            write_jsonl(
                args.output_dir / f"sft_{split}.jsonl",
                (training_row(task) for task in split_tasks),
            )
    print(json.dumps({"seed": args.seed, "split_counts": counts, "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

