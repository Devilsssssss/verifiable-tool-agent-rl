import importlib.util
import json
from pathlib import Path

import pytest

from toolrl.io import read_jsonl, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_runs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def save_run(path: Path, outcomes: list[bool], *, budget: int = 512) -> None:
    write_jsonl(
        path,
        (
            {"task_id": str(i), "family": "cancel_processing", "success": success}
            for i, success in enumerate(outcomes)
        ),
    )
    path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "tasks_sha256": "frozen",
                "limit": len(outcomes),
                "seed": 42,
                "temperature": 0,
                "max_new_tokens": budget,
                "input_format": "canonical-v4",
                "scoring_protocol": "strict",
                "overall": {"tasks": len(outcomes), "task_success": sum(outcomes) / len(outcomes)},
            }
        )
    )


def test_paired_comparison_counts_gains_and_losses(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    save_run(baseline, [True, False, False])
    save_run(candidate, [False, True, True])
    report = MODULE.compare_pair(baseline, candidate)
    assert report["gained_tasks"] == 2
    assert report["lost_tasks"] == 1
    assert report["strict_success_difference"] == pytest.approx(1 / 3)
    assert report["paired_bootstrap_95_interval"][0] <= 1 / 3
    assert report["paired_bootstrap_95_interval"][1] >= 1 / 3


def test_paired_comparison_rejects_changed_generation_budget(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    save_run(baseline, [False, False])
    save_run(candidate, [True, True], budget=1024)
    with pytest.raises(ValueError, match="max_new_tokens"):
        MODULE.compare_pair(baseline, candidate)


def test_paired_comparison_rejects_oracle_even_if_metadata_matches(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "before.jsonl", tmp_path / "assisted.jsonl"
    save_run(baseline, [False])
    save_run(candidate, [True])
    write_jsonl(
        candidate,
        [
            {
                "task_id": "0",
                "family": "cancel_processing",
                "success": True,
                "oracle_first_lookup": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="oracle-assisted"):
        MODULE.compare_pair(baseline, candidate)


def test_paired_comparison_rejects_input_intervention_even_if_metadata_matches(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "before.jsonl", tmp_path / "prompt-change.jsonl"
    save_run(baseline, [False])
    save_run(candidate, [True])
    candidate.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "tasks_sha256": "frozen",
                "limit": 1,
                "seed": 42,
                "temperature": 0,
                "max_new_tokens": 512,
                "input_format": "canonical-v4",
                "scoring_protocol": "strict",
                "input_intervention": "explicit-email-field-v1",
                "overall": {"tasks": 1, "task_success": 1.0},
            }
        )
    )
    write_jsonl(
        candidate,
        [
            {
                "task_id": "0",
                "family": "cancel_processing",
                "success": True,
                "input_intervention": "explicit-email-field-v1",
            }
        ],
    )
    with pytest.raises(ValueError, match="identical input interventions"):
        MODULE.compare_pair(baseline, candidate)


def test_paired_comparison_allows_same_input_intervention(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    save_run(baseline, [False])
    save_run(candidate, [True])
    for path in (baseline, candidate):
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        summary["input_intervention"] = "explicit-email-field-v1"
        path.with_suffix(".summary.json").write_text(json.dumps(summary))
        rows = [
            row | {"input_intervention": "explicit-email-field-v1"}
            for row in read_jsonl(path)
        ]
        write_jsonl(path, rows)
    report = MODULE.compare_pair(baseline, candidate)
    assert report["protocol"]["input_intervention"] == "explicit-email-field-v1"
    assert report["gained_tasks"] == 1
