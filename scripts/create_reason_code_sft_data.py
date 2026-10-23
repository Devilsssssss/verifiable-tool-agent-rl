from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import REASON_CODE_SCORING_PROTOCOL
from toolrl.expert import repair_training_row, replay_expert, training_row
from toolrl.io import load_tasks, write_jsonl
from toolrl.prompts import REASON_CODE_INPUT_CONTRACT
from toolrl.reason_codes import strict_reason_code_task


def _lookup_training_row(task) -> dict:
    row = training_row(
        task,
        reward_mode="shaped",
        scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
        input_contract=REASON_CODE_INPUT_CONTRACT,
    )
    row["task_id"] = f"{task.task_id}-lookup"
    row["messages"] = row["messages"][:3]
    row["lookup_frame"] = len(row["messages"])
    return row


def _prepare(source: Path, output_dir: Path, split: str) -> dict[str, object]:
    source_tasks = load_tasks(source)
    tasks = [strict_reason_code_task(task) for task in source_tasks]
    reports = [
        replay_expert(
            task,
            reward_mode="shaped",
            scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
            input_contract=REASON_CODE_INPUT_CONTRACT,
        )
        for task in tasks
    ]
    if not all(report["reward_report"]["success"] for report in reports):
        failed = [report["task_id"] for report in reports if not report["reward_report"]["success"]]
        raise ValueError(f"strict-v3 expert audit failed: {failed[:5]}")
    normal = [
        training_row(
            task,
            reward_mode="shaped",
            scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
            input_contract=REASON_CODE_INPUT_CONTRACT,
        )
        for task in tasks
    ]
    repairs = [
        repair_training_row(
            task,
            scoring_protocol=REASON_CODE_SCORING_PROTOCOL,
            input_contract=REASON_CODE_INPUT_CONTRACT,
        )
        for task in tasks
    ]
    outputs = {
        "tasks": output_dir / f"tasks_{split}.jsonl",
        "normal": output_dir / f"sft_{split}.jsonl",
        "repairs": output_dir / f"sft_repair_{split}.jsonl",
    }
    rows = {
        "tasks": [task.to_dict() for task in tasks],
        "normal": normal,
        "repairs": repairs,
    }
    if split == "train":
        lookup = [_lookup_training_row(task) for task in tasks]
        outputs.update(
            {
                "lookup": output_dir / "sft_lookup_train.jsonl",
                "extra": output_dir / "sft_repair_lookup_train.jsonl",
            }
        )
        rows.update({"lookup": lookup, "extra": repairs + lookup})
    for name, path in outputs.items():
        write_jsonl(path, rows[name])
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tasks": len(tasks),
        "strict_v3_expert_successes": len(reports),
        "normal_views": len(normal),
        "repair_views": len(repairs),
        "lookup_views": len(rows.get("lookup", [])),
        "total_training_views": (
            len(normal) + len(repairs) + len(rows.get("lookup", []))
            if split == "train"
            else None
        ),
        "output_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in outputs.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create SFT data with an explicit strict-v3 reason-code contract"
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        default=ROOT / "data/return-boundary-v1/tasks_train.jsonl",
    )
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=ROOT / "data/generated/tasks_validation.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/reason-code-sft-v1",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "artifacts/reason-code-sft-v1-training-audit.summary.json",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": _prepare(args.train_source, args.output_dir, "train"),
        "validation": _prepare(args.validation_source, args.output_dir, "validation"),
    }
    summary = {
        "status": "audited-before-sft-training",
        "output_dir": str(args.output_dir),
        "scoring_protocol": REASON_CODE_SCORING_PROTOCOL,
        "input_contract": REASON_CODE_INPUT_CONTRACT,
        "prompt_contract": (
            "strict-v3 system rules expose the required reason_code enum and forbid retrying "
            "request_return after return_window_closed"
        ),
        "splits": splits,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
