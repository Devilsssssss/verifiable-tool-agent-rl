from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def summarize_log(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in state["log_history"] if "reward" in row]
    if not rows:
        raise ValueError(f"no GRPO reward rows: {path}")
    required = (
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "grad_norm",
        "step",
        "completions/clipped_ratio",
        "tools/call_frequency",
        "tools/failure_frequency",
    )
    if any(key not in row for row in rows for key in required):
        raise ValueError("missing required GRPO diagnostics")
    if any(not math.isfinite(float(row[key])) for row in rows for key in required):
        raise ValueError("nonfinite GRPO diagnostics")
    if any(
        not 0 <= row[key] <= 1
        for row in rows
        for key in ("frac_reward_zero_std", "completions/clipped_ratio")
    ) or any(row["reward_std"] < 0 for row in rows):
        raise ValueError("invalid GRPO diagnostic range")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(set(steps)):
        raise ValueError("reward rows must have increasing unique step IDs")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_global_step": state["global_step"],
        "logged_reward_steps": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "mean_reward": sum(row["reward"] for row in rows) / len(rows),
        "mean_reward_std": sum(row["reward_std"] for row in rows) / len(rows),
        "mean_zero_std_group_fraction": sum(row["frac_reward_zero_std"] for row in rows)
        / len(rows),
        "steps_with_any_nonconstant_group": sum(row["frac_reward_zero_std"] < 1 for row in rows),
        "steps_with_positive_reward_std": sum(row["reward_std"] > 0 for row in rows),
        "mean_grad_norm": sum(row["grad_norm"] for row in rows) / len(rows),
        "mean_truncation_fraction": sum(row["completions/clipped_ratio"] for row in rows)
        / len(rows),
        "mean_tool_calls_per_completion": sum(row["tools/call_frequency"] for row in rows)
        / len(rows),
        "mean_tool_failures_per_completion": sum(row["tools/failure_frequency"] for row in rows)
        / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit actual GRPO Trainer state diagnostics")
    parser.add_argument("--states", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "runs": [summarize_log(path) for path in args.states],
        "scope": (
            "Means of per-step Trainer diagnostics, not a pooled rollout success rate. "
            "Constant reward groups have zero reward advantages; KL can still give gradients."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
