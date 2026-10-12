from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from toolrl.formatting import INPUT_FORMAT
from toolrl.io import load_tasks, write_jsonl
from toolrl.metrics import summarize_reports
from toolrl.rollout import run_episode
from toolrl.transformers_policy import TransformersToolPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="evaluate a model in the retail tool environment")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/generated/tasks_test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-format", choices=(INPUT_FORMAT, "legacy-v3"), default=INPUT_FORMAT)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    policy = TransformersToolPolicy(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        input_format=args.input_format,
    )
    records = []
    partial_path = args.output.with_suffix(".partial.jsonl")
    for index, task in enumerate(tasks, start=1):
        record = run_episode(task, policy)
        record["run_name"] = args.run_name
        records.append(record)
        # A failed/interrupted run retains clearly labelled partial trajectories.
        # Only a finished run writes the requested output and its summary.
        write_jsonl(partial_path, records)
        print(
            json.dumps(
                {
                    "index": index,
                    "task_id": task.task_id,
                    "success": record["success"],
                    "compliant": record["policy_compliant"],
                    "steps": record["steps"],
                }
            ),
            flush=True,
        )
    partial_path.replace(args.output)
    summary = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "adapter": str(args.adapter) if args.adapter else None,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "input_format": args.input_format,
        "tasks_path": str(args.tasks),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "limit": args.limit,
        "scoring_protocol": "strict-v1-with-terminal-diagnostics",
        **summarize_reports(records),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
