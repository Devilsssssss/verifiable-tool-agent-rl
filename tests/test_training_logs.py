import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_training_logs.py"
SPEC = importlib.util.spec_from_file_location("audit_training_logs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def state_path(tmp_path: Path, **changes) -> Path:
    row = {
        "step": 1,
        "reward": 1.0,
        "reward_std": 0.0,
        "frac_reward_zero_std": 1.0,
        "grad_norm": 0.01,
        "completions/clipped_ratio": 0.0,
        "tools/call_frequency": 4.0,
        "tools/failure_frequency": 0.25,
    }
    row.update(changes)
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"global_step": 1, "log_history": [row]}))
    return path


def test_high_mean_reward_does_not_imply_learning_signal(tmp_path: Path) -> None:
    report = MODULE.summarize_log(state_path(tmp_path))
    assert report["mean_reward"] == 1.0
    assert report["steps_with_any_nonconstant_group"] == 0
    assert report["mean_tool_failures_per_completion"] == 0.25


def test_incomplete_or_invalid_metrics_cannot_be_reported_as_zero(tmp_path: Path) -> None:
    path = state_path(tmp_path)
    data = json.loads(path.read_text())
    del data["log_history"][0]["completions/clipped_ratio"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing"):
        MODULE.summarize_log(path)
    with pytest.raises(ValueError, match="nonfinite"):
        MODULE.summarize_log(state_path(tmp_path, reward_std=float("nan")))
    with pytest.raises(ValueError, match="range"):
        MODULE.summarize_log(state_path(tmp_path, frac_reward_zero_std=1.2))
