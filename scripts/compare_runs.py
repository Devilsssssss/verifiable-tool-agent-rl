from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolrl.io import read_jsonl

PROTOCOL_KEYS = (
    "tasks_sha256",
    "limit",
    "seed",
    "temperature",
    "max_new_tokens",
    "input_format",
    "scoring_protocol",
)


def compare_pair(baseline: Path, candidate: Path, *, seed: int = 42) -> dict:
    """Paired exploratory comparison; reject mismatched inference protocols."""
    summaries = [
        json.loads(path.with_suffix(".summary.json").read_text()) for path in (baseline, candidate)
    ]
    missing = [key for key in PROTOCOL_KEYS if any(key not in row for row in summaries)]
    mismatch = [key for key in PROTOCOL_KEYS if summaries[0].get(key) != summaries[1].get(key)]
    if missing or mismatch:
        raise ValueError(f"incomparable protocols: missing={missing}, mismatched={mismatch}")
    runs = [read_jsonl(path) for path in (baseline, candidate)]
    if any(row.get("oracle_first_lookup", False) for rows in runs for row in rows):
        raise ValueError("oracle-assisted diagnostics cannot enter unaided model comparisons")
    if any(row.get("input_intervention", "original") != "original" for rows in runs for row in rows):
        raise ValueError("input interventions require a separate comparison protocol")
    indexed = [{row["task_id"]: row for row in rows} for rows in runs]
    if any(len(index) != len(rows) for index, rows in zip(indexed, runs, strict=True)):
        raise ValueError("duplicate task IDs in rollouts")
    if not indexed[0] or set(indexed[0]) != set(indexed[1]):
        raise ValueError("paired comparison requires identical nonempty task IDs")
    for summary, rows in zip(summaries, runs, strict=True):
        if summary["overall"]["tasks"] != len(rows):
            raise ValueError("summary task count does not match saved rollouts")
        success = sum(bool(row["success"]) for row in rows) / len(rows)
        if abs(success - summary["overall"]["task_success"]) > 1e-12:
            raise ValueError("summary success does not match saved rollouts")
    task_ids = sorted(indexed[0])
    pairs = [
        (bool(indexed[0][key]["success"]), bool(indexed[1][key]["success"])) for key in task_ids
    ]
    if any(indexed[0][key]["family"] != indexed[1][key]["family"] for key in task_ids):
        raise ValueError("task family mismatch")
    differences = [int(after) - int(before) for before, after in pairs]
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(5000)
    )
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "paired_tasks": len(pairs),
        "protocol": {key: summaries[0][key] for key in PROTOCOL_KEYS},
        "baseline_successes": sum(before for before, _ in pairs),
        "candidate_successes": sum(after for _, after in pairs),
        "gained_tasks": sum(not before and after for before, after in pairs),
        "lost_tasks": sum(before and not after for before, after in pairs),
        "strict_success_difference": sum(differences) / len(differences),
        "paired_bootstrap_95_interval": [bootstrap[124], bootstrap[4874]],
        "bootstrap_seed": seed,
        "bootstrap_resamples": 5000,
        "scope": (
            "Exploratory paired task bootstrap; tasks share synthetic templates. "
            "Does not quantify model-training seed variability or establish causality. "
            "A degenerate all-zero interval is not evidence of population equivalence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare saved runs under identical protocols")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_pair(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
