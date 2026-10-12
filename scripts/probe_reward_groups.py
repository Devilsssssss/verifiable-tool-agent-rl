from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.io import load_tasks, write_jsonl
from toolrl.metrics import summarize_reports
from toolrl.rollout import run_episode
from toolrl.transformers_policy import TransformersToolPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="Training-only reward variation preflight")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.98)
    args = parser.parse_args()
    if args.limit < 1 or args.generations < 2 or args.temperature <= 0:
        raise ValueError("require positive limit/temperature and at least two generations")
    tasks = load_tasks(args.tasks)[: args.limit]
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("reward preflight only accepts nonempty training tasks")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    policy = TransformersToolPolicy(
        model, tokenizer, max_new_tokens=512, temperature=args.temperature, top_p=args.top_p
    )
    records, groups = [], []
    partial_path = args.output.with_suffix(".partial.jsonl")
    for task in tasks:
        group = []
        for generation in range(args.generations):
            record = run_episode(task, policy)
            record["generation"] = generation
            records.append(record)
            group.append(record)
            write_jsonl(partial_path, records)
        scores = {mode: [row[mode] for row in group] for mode in ("sparse", "shaped")}
        groups.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "rewards": scores,
                "reward_stdev": {
                    mode: statistics.pstdev(values) for mode, values in scores.items()
                },
            }
        )
        print(json.dumps(groups[-1]), flush=True)
    partial_path.replace(args.output)
    summary = {
        "adapter": str(args.adapter),
        "tasks": str(args.tasks),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "groups": len(groups),
        "generations": args.generations,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens_per_turn": 512,
        "scope": (
            "Transformers inference preflight, not native GRPO group measurements. "
            "Task action budgets stay fixed; native GRPO logs remain authoritative."
        ),
        "reward_modes": {
            mode: {
                "nonconstant_groups": sum(group["reward_stdev"][mode] > 0 for group in groups),
                "mean_group_stdev": statistics.mean(
                    group["reward_stdev"][mode] for group in groups
                ),
                "all_zero_groups": sum(not any(group["rewards"][mode]) for group in groups),
            }
            for mode in ("sparse", "shaped")
        },
        "group_details": groups,
        **summarize_reports(records),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
