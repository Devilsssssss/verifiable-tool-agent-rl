from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.audit import audit_tasks
from toolrl.generator import generate_tasks
from toolrl.io import load_tasks, write_jsonl

REQUESTS = {
    "cancel_processing": "The email on my purchase is {email}. Please cancel the purchase of {old_name} and refund the payment.",
    "cancel_shipped": "The email on my purchase is {email}. Please cancel the purchase of {old_name} and refund the payment.",
    "exchange_cheaper": "The email on my purchase is {email}. Please change {old_name} to {new_name} and refund the difference in cost.",
    "return_eligible": "The email on my purchase is {email}. I would like a return and a refund for {old_name}.",
    "return_expired": "The email on my purchase is {email}. I would like a return and a refund for {old_name}.",
}


def transform_entities(value: Any, strings: dict, identifiers: dict) -> Any:
    if isinstance(value, dict):
        return {key: transform_entities(item, strings, identifiers) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [transform_entities(item, strings, identifiers) for item in value]
    if isinstance(value, str):
        return strings.get(value, value)
    if isinstance(value, int):
        return identifiers.get(value, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a fresh evaluation set before final model comparison"
    )
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/confirmation")
    parser.add_argument("--development-dir", type=Path, default=ROOT / "data/generated")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts/confirmation-audit.summary.json"
    )
    args = parser.parse_args()
    if args.count <= 0 or args.count % 5:
        raise ValueError("count must be a positive multiple of five")
    tasks = []
    for task in generate_tasks({"test": args.count}, seed=args.seed):
        code = hashlib.sha256(f"confirm:{args.seed}:{task.task_id}".encode()).hexdigest()[:10]
        email = f"buyer.{code}@heldout.test"
        products = task.initial_state["products"]
        rng = random.Random(code)
        brand = rng.choice(("Pine", "Maple", "Aspen", "Birch"))
        strings = {
            task.initial_state["users"][0]["email"]: email,
            products[0]["name"]: f"{brand} Headphones {code}",
            products[1]["name"]: f"{brand} Earbuds {code}",
            products[2]["name"]: f"{brand} Charger {code}",
        }
        identifiers = {
            row["id"]: row["id"] + 20_000_000
            for table in ("users", "products", "orders")
            for row in task.initial_state[table]
        }

        tasks.append(
            replace(
                task,
                task_id=task.task_id.replace("test-", "confirmation-", 1),
                split="confirmation",
                prompt=REQUESTS[task.family].format(
                    email=email,
                    old_name=strings[products[0]["name"]],
                    new_name=strings[products[1]["name"]],
                ),
                initial_state=transform_entities(task.initial_state, strings, identifiers),
                expert_plan=tuple(transform_entities(task.expert_plan, strings, identifiers)),
                goal=transform_entities(task.goal, strings, identifiers),
                template_id=f"confirmation-{task.family}-v1",
            )
        )
    development = [
        task
        for split in ("train", "validation", "test")
        for task in load_tasks(args.development_dir / f"tasks_{split}.jsonl")
    ]
    audit = audit_tasks(development + tasks)
    path = args.output_dir / "tasks_test.jsonl"
    if path.exists() and load_tasks(path) != tasks:
        raise ValueError("refusing to replace a different frozen confirmation set")
    write_jsonl(path, (task.to_dict() for task in tasks))
    summary = {
        "seed": args.seed,
        "confirmation_tasks": len(tasks),
        "audit": audit,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection_policy": "Do not use these outcomes to select prompts, checkpoints or data.",
        "scope": "Fresh entities and one fresh request template per family; target order remains first.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
