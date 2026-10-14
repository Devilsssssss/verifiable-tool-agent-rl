from __future__ import annotations

from statistics import mean
from typing import Any

TRAINING_SIGNATURE_KEYS = (
    "base_model",
    "train_samples",
    "validation_samples",
    "epochs",
    "learning_rate",
    "max_length",
    "assistant_only_loss",
    "completion_only_loss",
    "repair_prefix_supervised",
    "input_format",
    "batch_size",
    "gradient_accumulation",
    "optimizer_steps",
    "dataset_sha256",
)


def summarize_seed_replications(
    reports: dict[str, dict[str, Any]],
    *,
    required_replications: list[str],
    position_2_threshold: int,
) -> dict[str, Any]:
    """Aggregate position analyses after validating equal-budget training manifests."""
    if not reports:
        raise ValueError("at least one seed report is required")
    if not required_replications or len(set(required_replications)) != len(
        required_replications
    ):
        raise ValueError("required replications must be nonempty and unique")
    if missing := set(required_replications) - set(reports):
        raise ValueError(f"missing required replications: {sorted(missing)}")
    if not 0 <= position_2_threshold <= 60:
        raise ValueError("position_2_threshold must be between 0 and 60")

    signatures = {
        label: _training_signature(report["training_manifest"])
        for label, report in reports.items()
    }
    reference = next(iter(signatures.values()))
    mismatched = [label for label, signature in signatures.items() if signature != reference]
    if mismatched:
        raise ValueError(f"training configuration mismatch: {mismatched}")

    training_seeds = [report["training_manifest"]["seed"] for report in reports.values()]
    if len(training_seeds) != len(set(training_seeds)):
        raise ValueError("training seeds must be distinct")

    per_seed = {}
    for label, report in reports.items():
        analysis = report["position_analysis"]
        strata = analysis["strata"]
        selection = analysis["target_order_selection"]["candidate"]
        entry = {
            "training_seed": report["training_manifest"]["seed"],
            "overall_successes": strata["overall"]["candidate_successes"],
            "position_1_successes": strata["position_1"]["candidate_successes"],
            "position_2_successes": strata["position_2"]["candidate_successes"],
            "position_2_target_order_selections": selection["position_2"][
                "target_order_selections"
            ],
        }
        entry["meets_position_2_threshold"] = (
            entry["position_2_successes"] >= position_2_threshold
        )
        entry["meets_position_2_selection_requirement"] = (
            entry["position_2_target_order_selections"] == 60
        )
        per_seed[label] = entry

    def aggregate(key: str) -> dict[str, float | int]:
        values = [entry[key] for entry in per_seed.values()]
        return {"mean": mean(values), "min": min(values), "max": max(values)}

    required_checks = {
        label: (
            per_seed[label]["meets_position_2_threshold"]
            and per_seed[label]["meets_position_2_selection_requirement"]
        )
        for label in required_replications
    }
    return {
        "training_signature": reference,
        "per_seed": per_seed,
        "aggregate": {
            "overall_successes": aggregate("overall_successes"),
            "position_1_successes": aggregate("position_1_successes"),
            "position_2_successes": aggregate("position_2_successes"),
            "position_2_target_order_selections": aggregate(
                "position_2_target_order_selections"
            ),
        },
        "replication_criterion": {
            "required_replications": required_replications,
            "position_2_success_threshold": position_2_threshold,
            "position_2_target_order_selection_requirement": 60,
            "per_replication_pass": required_checks,
            "passed": all(required_checks.values()),
        },
        "scope": (
            "Training-seed replication on one fixed synthetic confirmation set; does not "
            "measure task-generator or real-world distribution variability."
        ),
    }


def _training_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in TRAINING_SIGNATURE_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"training manifest missing keys: {missing}")
    return {key: manifest[key] for key in TRAINING_SIGNATURE_KEYS}
