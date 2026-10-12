from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.augmentation import TRAINING_REQUESTS
from toolrl.environment import RetailToolEnvironment
from toolrl.io import load_tasks, write_jsonl
from toolrl.prompts import conversation_prompt
from toolrl.rollout import environment_tools
from toolrl.transformers_policy import TransformersToolPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="First-call wording probe, not task evaluation")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

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
        prompts = {"original": task.prompt}
        for index, template in enumerate(TRAINING_REQUESTS[task.family]):
            prompts[f"training-wording-{index}"] = template.format(
                email=email, old_name=products[0]["name"], new_name=products[1]["name"]
            )
        for condition, prompt in prompts.items():
            response = policy(conversation_prompt(prompt), tools)
            calls = response.get("tool_calls") or []
            function = calls[0].get("function", {}) if len(calls) == 1 else {}
            arguments = function.get("arguments", {})
            correct = (
                function.get("name") == "lookup_user"
                and isinstance(arguments, dict)
                and arguments == {"email": email}
            )
            records.append(
                {
                    "task_id": task.task_id,
                    "condition": condition,
                    "prompt": prompt,
                    "expected_email": email,
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
        "seed": 42,
        "temperature": 0,
        "max_new_tokens": 512,
        "scope": "Diagnostic first-call probe; no business task success is measured.",
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
