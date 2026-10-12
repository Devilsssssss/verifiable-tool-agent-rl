from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import RetailToolEnvironment
from toolrl.formatting import (
    INPUT_FORMAT,
    canonical_messages,
    canonical_tool_schemas,
    sort_template_json,
)
from toolrl.io import read_jsonl
from toolrl.rollout import environment_tools


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare canonical SFT/policy and native GRPO prompts"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from trl.chat_template_utils import add_response_schema, get_training_chat_template

    tokenizer = add_response_schema(AutoTokenizer.from_pretrained(args.model_id))
    tokenizer.chat_template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    grpo_tokenizer = add_response_schema(AutoTokenizer.from_pretrained(args.model_id))
    grpo_tokenizer.chat_template = (
        get_training_chat_template(grpo_tokenizer) or grpo_tokenizer.chat_template
    )
    sort_template_json(grpo_tokenizer)
    tools = environment_tools(RetailToolEnvironment())
    schemas = canonical_tool_schemas(tools)
    rows = [row for path in args.data for row in read_jsonl(path)]
    checked = 0
    mismatches = []
    # Inspect every assistant generation point, not just the initial user request.
    for row in rows:
        for index, message in enumerate(row["messages"]):
            if message["role"] != "assistant":
                continue
            history = row["messages"][:index]
            canonical = tokenizer.apply_chat_template(
                canonical_messages(history),
                tools=schemas,
                add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
            native = grpo_tokenizer.apply_chat_template(
                history,
                tools=tools,
                add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
            checked += 1
            if canonical != native:
                mismatches.append({"task_id": row["task_id"], "assistant_index": index})
    summary = {
        "input_format": INPUT_FORMAT,
        "samples": len(rows),
        "assistant_prefixes_checked": checked,
        "mismatched_prefixes": len(mismatches),
        "examples": mismatches[:5],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
