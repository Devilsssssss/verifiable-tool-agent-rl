from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import RetailToolEnvironment
from toolrl.io import read_jsonl
from toolrl.rollout import environment_tools


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA SFT on verified expert tool trajectories")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train", type=Path, default=ROOT / "data/generated/sft_train.jsonl")
    parser.add_argument(
        "--validation", type=Path, default=ROOT / "data/generated/sft_validation.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen3-1.7b-tool-sft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from transformers.utils import get_json_schema
    from trl import SFTConfig, SFTTrainer
    from trl.chat_template_utils import get_training_chat_template

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    patched_template = get_training_chat_template(tokenizer)
    if patched_template is not None:
        tokenizer.chat_template = patched_template
    tools = environment_tools(RetailToolEnvironment())
    tool_schemas = [get_json_schema(tool) for tool in tools]

    def as_conversations(rows: list[dict]) -> Dataset:
        return Dataset.from_list(
            [
                {
                    "task_id": row["task_id"],
                    "family": row["family"],
                    "messages": row["messages"],
                    "tools": json.dumps(tool_schemas, sort_keys=True),
                }
                for row in rows
            ]
        )

    train_rows = read_jsonl(args.train)
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    train_dataset = as_conversations(train_rows)
    eval_dataset = as_conversations(read_jsonl(args.validation))
    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        assistant_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        save_only_model=True,
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=args.model_id,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "stage": "sft",
        "base_model": args.model_id,
        "train_samples": len(train_dataset),
        "validation_samples": len(eval_dataset),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
