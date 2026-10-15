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
from toolrl.io import load_tasks, read_jsonl, write_jsonl
from toolrl.return_holdout import RETURN_FAMILIES, TRAIN_DAYS, vary_return_days


def main() -> int:
    parser = argparse.ArgumentParser(description="Create equal-view return-age SFT intervention")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/position-v10")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/return-v11")
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts/return-v11-training-audit.summary.json")
    args = parser.parse_args()
    source = load_tasks(args.data_dir / "tasks_train.jsonl")
    tasks = vary_return_days(source, seed=args.seed, days=TRAIN_DAYS)
    heldout = [task for path in args.heldout for task in load_tasks(path)]
    audit = audit_tasks(tasks + heldout)
    normal = [training_row(task) for task in tasks]
    repairs = [repair_training_row(task) for task in tasks]
    lookup = read_jsonl(args.data_dir / "sft_lookup_train.jsonl")
    source_normal = read_jsonl(args.data_dir / "sft_train.jsonl")
    source_repairs = read_jsonl(args.data_dir / "sft_repair_train.jsonl")
    for index, task in enumerate(tasks):
        if (task.family not in RETURN_FAMILIES
                and (normal[index] != source_normal[index] or repairs[index] != source_repairs[index])):
            raise ValueError("non-return training view changed")
    outputs = {
        "tasks": args.output_dir / "tasks_train.jsonl",
        "normal": args.output_dir / "sft_train.jsonl",
        "repairs": args.output_dir / "sft_repair_train.jsonl",
        "lookup": args.output_dir / "sft_lookup_train.jsonl",
        "extra": args.output_dir / "sft_repair_lookup_train.jsonl",
    }
    for key, rows in (("tasks", [t.to_dict() for t in tasks]), ("normal", normal),
                      ("repairs", repairs), ("lookup", lookup), ("extra", repairs + lookup)):
        if outputs[key].exists() and read_jsonl(outputs[key]) != rows:
            raise ValueError(f"refusing to overwrite different frozen training data: {key}")
        write_jsonl(outputs[key], rows)
    lookup_identical = (outputs["lookup"].read_bytes()
                        == (args.data_dir / "sft_lookup_train.jsonl").read_bytes())
    if not lookup_identical:
        raise ValueError("lookup training views are not byte-identical to v10")
    summary = {
        "status": "frozen-before-training", "seed": args.seed,
        "independent_training_tasks": len(tasks),
        "total_views": len(normal) + len(repairs) + len(lookup),
        "family_counts": dict(Counter(task.family for task in tasks)),
        "lookup_views_byte_identical_to_v10": lookup_identical,
        "nonreturn_views_unchanged": True,
        "semantic_audit": audit,
        "output_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in outputs.items()},
        "intervention": "Only return target/distractor delivery ages change; 900 views and positions preserved.",
        "scope": "Equal views and optimizer steps, not equal token counts; two age factors change jointly.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
