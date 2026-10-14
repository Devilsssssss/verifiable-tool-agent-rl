from copy import deepcopy

import pytest

from toolrl.replication import TRAINING_SIGNATURE_KEYS, summarize_seed_replications


def _manifest(seed: int) -> dict:
    values = {
        "base_model": "model",
        "train_samples": 900,
        "validation_samples": 120,
        "epochs": 1.0,
        "learning_rate": 0.0002,
        "max_length": 4096,
        "assistant_only_loss": True,
        "completion_only_loss": True,
        "repair_prefix_supervised": False,
        "input_format": "canonical-v4",
        "batch_size": 1,
        "gradient_accumulation": 8,
        "optimizer_steps": 113,
        "dataset_sha256": {"train": "abc"},
        "seed": seed,
    }
    assert set(TRAINING_SIGNATURE_KEYS) <= set(values)
    return values


def _report(seed: int, *, position_1: int, position_2: int, selected: int = 60) -> dict:
    return {
        "training_manifest": _manifest(seed),
        "position_analysis": {
            "strata": {
                "overall": {"candidate_successes": position_1 + position_2},
                "position_1": {"candidate_successes": position_1},
                "position_2": {"candidate_successes": position_2},
            },
            "target_order_selection": {
                "candidate": {
                    "position_2": {"target_order_selections": selected},
                }
            },
        },
    }


def test_seed_replication_summary_applies_preregistered_criterion() -> None:
    reports = {
        "seed45": _report(45, position_1=58, position_2=47),
        "seed46": _report(46, position_1=60, position_2=42),
        "seed47": _report(47, position_1=59, position_2=30),
    }

    result = summarize_seed_replications(
        reports,
        required_replications=["seed46", "seed47"],
        position_2_threshold=30,
    )

    assert result["replication_criterion"]["passed"] is True
    assert result["aggregate"]["position_2_successes"] == {
        "mean": pytest.approx(39.666666666666664),
        "min": 30,
        "max": 47,
    }


def test_seed_replication_summary_rejects_training_mismatch() -> None:
    reports = {
        "seed45": _report(45, position_1=58, position_2=47),
        "seed46": _report(46, position_1=60, position_2=42),
    }
    reports["seed46"] = deepcopy(reports["seed46"])
    reports["seed46"]["training_manifest"]["learning_rate"] = 0.0001

    with pytest.raises(ValueError, match="training configuration mismatch"):
        summarize_seed_replications(
            reports,
            required_replications=["seed46"],
            position_2_threshold=30,
        )


def test_seed_replication_summary_requires_target_selection() -> None:
    reports = {
        "seed45": _report(45, position_1=58, position_2=47),
        "seed46": _report(46, position_1=60, position_2=45, selected=59),
    }

    result = summarize_seed_replications(
        reports,
        required_replications=["seed46"],
        position_2_threshold=30,
    )

    assert result["replication_criterion"]["passed"] is False
