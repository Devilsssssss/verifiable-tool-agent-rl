from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from toolrl.environment import RetailToolEnvironment
from toolrl.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO for a stateful retail tool agent")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=("sparse", "shaped"), required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from trl.chat_template_utils import add_response_schema, get_training_chat_template

    train_path = args.train or ROOT / f"data/generated/grpo_train_{args.reward_mode}.jsonl"
    rows = read_jsonl(train_path)
    if args.max_train_samples is not None:
        rows = rows[: args.max_train_samples]
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer = add_response_schema(tokenizer)
    patched_template = get_training_chat_template(tokenizer)
    if patched_template is not None:
        tokenizer.chat_template = patched_template

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)
    model.config.use_cache = False
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=8,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=0.02,
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=25,
        save_only_model=True,
        report_to="none",
        seed=args.seed,
        use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=RetailToolEnvironment,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "stage": "grpo",
        "base_model": args.model_id,
        "sft_adapter": str(args.sft_adapter),
        "reward_mode": args.reward_mode,
        "train_samples": len(dataset),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
