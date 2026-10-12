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

from toolrl.diagnostics import (
    ExplicitEmailPrompt,
    OracleFirstLookup,
    correct_lookup,
    factorial_cases,
)
from toolrl.environment import RetailToolEnvironment
from toolrl.io import load_tasks, write_jsonl
from toolrl.metrics import summarize_reports
from toolrl.prompts import conversation_prompt
from toolrl.rollout import environment_tools, run_episode
from toolrl.transformers_policy import TransformersToolPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw generation, factorial and oracle diagnostics")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("raw", "factorial", "oracle", "email-field"), required=True
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = load_tasks(args.tasks)[: args.limit]
    if not tasks or args.limit <= 0:
        raise ValueError("a positive, nonempty task sample is required")
    if args.mode == "factorial" and any(task.split != "train" for task in tasks):
        raise ValueError("factorial diagnostics only accept training tasks")

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
    policy = TransformersToolPolicy(model, tokenizer, max_new_tokens=512, capture_generation=True)
    records = []
    tools = environment_tools(RetailToolEnvironment())
    for task in tasks:
        email = task.initial_state["users"][0]["email"]
        if args.mode in {"oracle", "email-field"}:
            if email not in task.prompt:
                raise ValueError("oracle may only supply an email already visible in the request")
            generations = []

            def capture(messages, tool_functions, generations=generations):
                response = policy(messages, tool_functions)
                generations.append(policy.last_generation)
                return response

            episode_policy = (
                OracleFirstLookup(capture, email)
                if args.mode == "oracle"
                else ExplicitEmailPrompt(capture)
            )
            record = run_episode(task, episode_policy)
            record.update(raw_generations=generations)
            if args.mode == "oracle":
                record["oracle_first_lookup"] = True
            else:
                record["input_intervention"] = "explicit-email-field-v1"
            records.append(record)
        else:
            cases = (
                factorial_cases(task)
                if args.mode == "factorial"
                else [
                    {
                        "prompt": task.prompt,
                        "expected_email": email,
                        "frame": "original",
                        "email_structure": "original",
                    }
                ]
            )
            for case in cases:
                response = policy(conversation_prompt(case["prompt"]), tools)
                record = {
                    "task_id": task.task_id,
                    "family": task.family,
                    **case,
                    "response": response,
                    "generation": policy.last_generation,
                    "correct": correct_lookup(response, case["expected_email"]),
                }
                records.append(record)
        write_jsonl(args.output.with_suffix(".partial.jsonl"), records)
        print(json.dumps({"task_id": task.task_id, "records": len(records)}), flush=True)
    write_jsonl(args.output, records)
    args.output.with_suffix(".partial.jsonl").unlink()
    summary = {
        "mode": args.mode,
        "adapter": args.adapter,
        "tasks": str(args.tasks),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "seed": 42,
        "temperature": 0,
        "max_new_tokens": 512,
        "scope": "Development diagnostic only; oracle scores are assisted, not unaided results.",
    }
    if args.mode == "oracle":
        summary["assisted_metrics"] = summarize_reports(records)
    elif args.mode == "email-field":
        summary["scope"] = (
            "Development prompt intervention using only visible user text; all calls are generated. "
            "Not the original prompt protocol and not evidence of SFT or RL improvement."
        )
        summary["prompt_intervention_metrics"] = summarize_reports(records)
    else:
        summary["calls"] = len(records)
        summary["correct_email_calls"] = sum(row["correct"] for row in records)
        cells = sorted({(str(row["frame"]), row["email_structure"]) for row in records})
        summary["cells"] = {
            f"frame-{frame}/{structure}": {
                "calls": sum(
                    str(row["frame"]) == frame and row["email_structure"] == structure
                    for row in records
                ),
                "correct": sum(
                    row["correct"]
                    for row in records
                    if str(row["frame"]) == frame and row["email_structure"] == structure
                ),
            }
            for frame, structure in cells
        }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
