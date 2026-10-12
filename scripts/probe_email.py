from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.augmentation import TRAINING_REQUESTS
from toolrl.environment import RetailToolEnvironment
from toolrl.io import load_tasks, read_jsonl, write_jsonl
from toolrl.prompts import conversation_prompt
from toolrl.rollout import environment_tools
from toolrl.transformers_policy import TransformersToolPolicy


def lookup_controls(task, row: dict) -> dict[str, tuple[str, str]]:
    """Cross email spelling and request wording; never score business success."""
    if row.get("split") != "train" or row.get("family") != task.family:
        raise ValueError("lookup controls require a same-family training row")
    messages = row["messages"]
    function = messages[-1]["tool_calls"][0]["function"]
    if function["name"] != "lookup_user":
        raise ValueError("lookup control answer must be lookup_user")
    train_email = function["arguments"]["email"]
    train_prompt = messages[1]["content"]
    email = task.initial_state["users"][0]["email"]
    if train_email not in train_prompt or email not in task.prompt:
        raise ValueError("lookup control email missing from prompt")
    return {
        "original-wording-training-email": (task.prompt.replace(email, train_email), train_email),
        "lookup-wording-unseen-email": (train_prompt.replace(train_email, email), email),
        "lookup-exact-training-row": (train_prompt, train_email),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="First-call wording probe, not task evaluation")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--lookup-data", type=Path, help="Optional training-only lookup controls")
    args = parser.parse_args()
    lookup_rows = read_jsonl(args.lookup_data) if args.lookup_data else []
    if any(row.get("split") != "train" for row in lookup_rows):
        raise ValueError("lookup controls only accept training rows")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    policy = TransformersToolPolicy(model, tokenizer, max_new_tokens=512)
    records = []
    tools = environment_tools(RetailToolEnvironment())
    for task in load_tasks(args.tasks)[: args.limit]:
        email = task.initial_state["users"][0]["email"]
        products = task.initial_state["products"]
        prompts = {"original": (task.prompt, email)}
        for index, template in enumerate(TRAINING_REQUESTS[task.family]):
            prompts[f"training-wording-{index}"] = (
                template.format(
                    email=email, old_name=products[0]["name"], new_name=products[1]["name"]
                ),
                email,
            )
        if args.lookup_data:
            rows = [row for row in lookup_rows if row["family"] == task.family]
            if not rows:
                raise ValueError(f"missing lookup control family: {task.family}")
            prompts.update(lookup_controls(task, rows[len(records) % len(rows)]))
        for condition, (prompt, expected_email) in prompts.items():
            response = policy(conversation_prompt(prompt), tools)
            calls = response.get("tool_calls") or []
            function = calls[0].get("function", {}) if len(calls) == 1 else {}
            arguments = function.get("arguments", {})
            correct = (
                function.get("name") == "lookup_user"
                and isinstance(arguments, dict)
                and arguments == {"email": expected_email}
            )
            records.append(
                {
                    "task_id": task.task_id,
                    "condition": condition,
                    "prompt": prompt,
                    "expected_email": expected_email,
                    "response": response,
                    "correct": correct,
                }
            )
            print(
                json.dumps({"task_id": task.task_id, "condition": condition, "correct": correct}),
                flush=True,
            )
    write_jsonl(args.output, records)
    conditions = sorted({row["condition"] for row in records})
    summary = {
        "adapter": str(args.adapter),
        "tasks": str(args.tasks),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "lookup_data": str(args.lookup_data) if args.lookup_data else None,
        "lookup_data_sha256": (
            hashlib.sha256(args.lookup_data.read_bytes()).hexdigest() if args.lookup_data else None
        ),
        "seed": 42,
        "temperature": 0,
        "max_new_tokens": 512,
        "scope": (
            "Diagnostic first-call probe; no business task success is measured. "
            "Lookup controls replace email only; product wording is not factorially controlled."
        ),
        "conditions": {
            condition: {
                "calls": sum(row["condition"] == condition for row in records),
                "correct_email_calls": sum(
                    row["correct"] for row in records if row["condition"] == condition
                ),
            }
            for condition in conditions
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
