from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import RetailToolEnvironment
from toolrl.formatting import (
    INPUT_FORMAT,
    canonical_messages,
    canonical_tool_schemas,
    sft_conversation,
)
from toolrl.io import read_jsonl
from toolrl.rollout import environment_tools


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit actual SFT labels without training")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from trl.chat_template_utils import add_response_schema, get_training_chat_template

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    inference_tokenizer = add_response_schema(AutoTokenizer.from_pretrained(args.model_id))
    for current in (tokenizer, inference_tokenizer):
        patched = get_training_chat_template(current)
        if patched is not None:
            current.chat_template = patched
    tools = environment_tools(RetailToolEnvironment())
    schemas = canonical_tool_schemas(tools)
    rows = [row for path in args.data for row in read_jsonl(path)]
    conversations = [sft_conversation(row, schemas) for row in rows]
    # Inspect the installed trainer's real preprocessing and collator rather than
    # assuming how a particular TRL version combines the two masks.
    dataset = Dataset.from_list(conversations)
    normalized_conversations = list(dataset)
    with TemporaryDirectory(prefix="toolrl-mask-audit-") as temporary:
        trainer = SFTTrainer(
            model=args.model_id,
            args=SFTConfig(
                output_dir=temporary,
                assistant_only_loss=True,
                completion_only_loss=True,
                max_length=args.max_length,
                packing=False,
                report_to="none",
                model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        features = list(trainer.train_dataset)
    examples = []
    for row, conversation, feature in zip(rows, normalized_conversations, features, strict=True):
        boundary = int(row.get("completion_start", 2))
        prompt = conversation["prompt"]
        completion = conversation["completion"]
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tools=schemas, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        full = tokenizer.apply_chat_template(
            prompt + completion,
            tools=schemas,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = full["input_ids"]
        prefix_length = 0
        for left, right in zip(prompt_ids, input_ids):
            if left != right:
                break
            prefix_length += 1
        labels = trainer.data_collator([feature])["labels"][0].tolist()
        # Strip the generation prefix: this is the entire prior conversation,
        # including any deliberately invalid assistant call and its tool error.
        context_ids = tokenizer.apply_chat_template(prompt, tools=schemas, return_dict=True)[
            "input_ids"
        ]
        context_matches = feature["input_ids"][: len(context_ids)] == context_ids
        prompt_labels = labels[: min(prefix_length, len(labels))]
        inference_ids = inference_tokenizer.apply_chat_template(
            canonical_messages(row["messages"][:boundary]),
            tools=canonical_tool_schemas(tools),
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        canonical_inference_ids = inference_tokenizer.apply_chat_template(
            canonical_messages(row["messages"][:boundary]),
            tools=schemas,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        supervised = [token for token, label in zip(feature["input_ids"], labels) if label != -100]
        examples.append(
            {
                "task_id": row["task_id"],
                "family": row["family"],
                "repair": "completion_start" in row,
                "tokens": len(input_ids),
                "truncated": len(input_ids) > args.max_length,
                "supervised_tokens": len(supervised),
                "prompt_fully_masked": all(label == -100 for label in prompt_labels),
                "context_is_prefix": context_matches,
                "audited_tokens_match_trainer": feature["input_ids"]
                == input_ids[: args.max_length],
                "generation_prefix_is_prefix": prefix_length == len(prompt_ids),
                "generation_prefix_untrained_suffix": tokenizer.decode(prompt_ids[prefix_length:]),
                "train_inference_prompt_identical": prompt_ids == inference_ids,
                "schema_order_identical": canonical_inference_ids == inference_ids,
                "arrow_context_identical": prompt_ids == canonical_inference_ids,
                "supervised_preview": tokenizer.decode(supervised)[:220],
            }
        )
    summary = {
        "model_id": args.model_id,
        "input_format": INPUT_FORMAT,
        "data": [str(path) for path in args.data],
        "max_length": args.max_length,
        "samples": len(examples),
        "repair_samples": sum(row["repair"] for row in examples),
        "unmasked_prompt_samples": sum(not row["prompt_fully_masked"] for row in examples),
        "empty_supervision_samples": sum(row["supervised_tokens"] == 0 for row in examples),
        "context_mismatch_samples": sum(not row["context_is_prefix"] for row in examples),
        "audited_token_mismatch_samples": sum(
            not row["audited_tokens_match_trainer"] for row in examples
        ),
        "generation_prefix_mismatch_samples": sum(
            not row["generation_prefix_is_prefix"] for row in examples
        ),
        "train_inference_prompt_mismatch_samples": sum(
            not row["train_inference_prompt_identical"] for row in examples
        ),
        "schema_order_mismatch_samples": sum(not row["schema_order_identical"] for row in examples),
        "arrow_context_mismatch_samples": sum(
            not row["arrow_context_identical"] for row in examples
        ),
        "truncated_samples": sum(row["truncated"] for row in examples),
        "max_tokens": max(row["tokens"] for row in examples),
        "examples": [
            next(row for row in examples if row["repair"] == repair)
            for repair in (False, True)
            if any(row["repair"] == repair for row in examples)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return int(
        any(
            summary[key]
            for key in (
                "unmasked_prompt_samples",
                "empty_supervision_samples",
                "context_mismatch_samples",
                "audited_token_mismatch_samples",
                "generation_prefix_mismatch_samples",
                "train_inference_prompt_mismatch_samples",
                "truncated_samples",
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
