from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_manifest(seed: int, adapter: Path) -> dict:
    manifest = read_json(adapter / "training_manifest.json")
    suffix = "" if seed == 45 else f"-seed{seed}"
    baseline = read_json(ROOT / f"artifacts/qwen3-sft-position-v10{suffix}-training.summary.json")
    controls = (
        "base_model", "train_samples", "primary_train_samples", "extra_train_samples",
        "validation_samples", "epochs", "learning_rate", "max_length", "seed",
        "assistant_only_loss", "completion_only_loss", "repair_prefix_supervised",
        "input_format", "batch_size", "gradient_accumulation", "optimizer_steps", "libraries",
    )
    mismatches = [key for key in controls if key not in manifest or manifest[key] != baseline[key]]
    if manifest.get("seed") != seed or mismatches:
        raise ValueError(f"candidate training controls differ from v10: {mismatches}")
    audit = read_json(ROOT / "artifacts/return-v11-training-audit.summary.json")
    expected = {
        "data/return-v11/sft_train.jsonl": audit["output_sha256"]["normal"],
        "data/return-v11/sft_repair_lookup_train.jsonl": audit["output_sha256"]["extra"],
        "data/generated/sft_validation.jsonl": baseline["dataset_sha256"][
            "data/generated/sft_validation.jsonl"
        ],
        "data/generated/sft_repair_validation.jsonl": baseline["dataset_sha256"][
            "data/generated/sft_repair_validation.jsonl"
        ],
    }
    if manifest.get("dataset_sha256") != expected:
        raise ValueError("candidate manifest does not match the frozen data digests")
    for name, digest in expected.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"training data changed after freeze: {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a final v11 adapter, replay it, and compare with its matched v10 seed"
    )
    parser.add_argument("--seed", type=int, choices=(45, 46), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--wait-for-training-pid", type=int)
    parser.add_argument("--training-timeout", type=int, default=5400)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    adapter = ROOT / f"outputs/qwen3-1.7b-tool-sft-return-v11-seed{args.seed}"
    manifest_path = adapter / "training_manifest.json"
    deadline = time.monotonic() + args.training_timeout
    while not manifest_path.exists() and args.wait_for_training_pid is not None:
        try:
            os.kill(args.wait_for_training_pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError(
                f"training process {args.wait_for_training_pid} exited before manifest appeared"
            ) from error
        if time.monotonic() >= deadline:
            raise TimeoutError("training manifest did not appear before timeout")
        time.sleep(10)
    checked_manifest(args.seed, adapter)
    if args.check_only:
        print("Frozen data and matched training controls verified", flush=True)
        return 0

    state_path = ROOT / f"artifacts/return-v11-seed{args.seed}-pipeline.summary.json"
    state = {"seed": args.seed, "status": "running", "completed_stages": []}

    def save() -> None:
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def run(stage: str, script: str, *arguments: str) -> None:
        state["current_stage"] = stage
        save()
        command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
        print(json.dumps({"stage": stage, "command": command}), flush=True)
        subprocess.run(command, check=True)
        state["completed_stages"].append(stage)
        save()

    save()
    try:
        shutil.copyfile(
            manifest_path,
            ROOT / f"artifacts/qwen3-sft-return-v11-seed{args.seed}-training.summary.json",
        )
        regression_baseline = (
            "artifacts/v10-position-v10-confirmation-120.jsonl" if args.seed == 45
            else f"artifacts/v10-seed{args.seed}-position-v10-confirmation-120.jsonl"
        )
        cases = (
            ("holdout", "data/return-holdout/tasks_test.jsonl",
             f"artifacts/return-holdout-seed{args.seed}.jsonl"),
            ("regression", "artifacts/position-v10-confirmation-tasks.jsonl", regression_baseline),
        )
        for label, tasks, baseline in cases:
            candidate = f"artifacts/return-v11-seed{args.seed}-{label}.jsonl"
            prefix = candidate.removesuffix(".jsonl")
            # Never silently replace an earlier model evaluation.
            if Path(prefix + ".partial.jsonl").exists():
                raise FileExistsError(f"incomplete evaluation needs explicit review: {candidate}")
            baseline_summary = read_json(Path(baseline).with_suffix(".summary.json"))
            if hashlib.sha256(Path(tasks).read_bytes()).hexdigest() != baseline_summary["tasks_sha256"]:
                raise ValueError(f"baseline and task file differ for {label}")
            if Path(candidate).exists():
                existing = read_json(Path(prefix + ".summary.json"))
                expected = {
                    "adapter": str(adapter.relative_to(ROOT)), "model_id": "models/Qwen3-1.7B",
                    "seed": 42, "limit": 120, "max_new_tokens": 512, "temperature": 0,
                    "input_format": "canonical-v4", "input_intervention": "explicit-email-field-v1",
                    "scoring_protocol": "strict-v2-target-inspection",
                    "tasks_sha256": baseline_summary["tasks_sha256"],
                    "run_name": f"return-v11-seed{args.seed}-{label}",
                }
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"existing evaluation has incompatible settings: {candidate}")
            else:
                run(label, "evaluate_model.py", "--model-id", "models/Qwen3-1.7B",
                    "--adapter", str(adapter.relative_to(ROOT)), "--tasks", tasks,
                    "--output", candidate, "--run-name", f"return-v11-seed{args.seed}-{label}",
                    "--limit", "120", "--max-new-tokens", "512", "--temperature", "0",
                    "--seed", "42", "--input-intervention", "explicit-email-field-v1")
            run(label + "-replay", "audit_rollouts.py", "--tasks", tasks,
                "--rollouts", candidate, "--output", prefix + "-replay.summary.json")
            if label == "holdout":
                run(label + "-analysis", "analyze_return_holdout.py", "--tasks", tasks,
                    "--rollouts", candidate, "--output", prefix + "-analysis.summary.json")
            run(label + "-paired", "analyze_position_probe.py", "--tasks", tasks,
                "--baseline", baseline, "--candidate", candidate,
                "--output", prefix + "-paired.summary.json")
        state["status"] = "completed"
        state["current_stage"] = None
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
