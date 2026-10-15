import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "evaluate_return_candidate", ROOT / "scripts/evaluate_return_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    baseline = json.loads(
        (ROOT / "artifacts/qwen3-sft-position-v10-training.summary.json").read_text()
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "qwen3-sft-position-v10-training.summary.json").write_text(json.dumps(baseline))
    manifest = dict(baseline)
    manifest["dataset_sha256"] = {}
    names = [
        "data/return-v11/sft_train.jsonl", "data/return-v11/sft_repair_lookup_train.jsonl",
        "data/generated/sft_validation.jsonl", "data/generated/sft_repair_validation.jsonl",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        manifest["dataset_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if "validation" in name:
            baseline["dataset_sha256"][name] = manifest["dataset_sha256"][name]
    (artifacts / "qwen3-sft-position-v10-training.summary.json").write_text(json.dumps(baseline))
    audit = {"output_sha256": {
        "normal": manifest["dataset_sha256"][names[0]],
        "extra": manifest["dataset_sha256"][names[1]],
    }}
    (artifacts / "return-v11-training-audit.summary.json").write_text(json.dumps(audit))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "training_manifest.json").write_text(json.dumps(manifest))
    return module, tmp_path, adapter, manifest


def test_candidate_requires_matched_controls_and_frozen_training_bytes(candidate):
    module, root, adapter, manifest = candidate
    assert module.checked_manifest(45, adapter) == manifest
    (root / "data/return-v11/sft_train.jsonl").write_bytes(b"changed after freeze")
    with pytest.raises(ValueError, match="training data changed"):
        module.checked_manifest(45, adapter)


def test_candidate_rejects_changed_training_budget(candidate):
    module, _, adapter, manifest = candidate
    manifest["epochs"] = 2
    (adapter / "training_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="controls differ"):
        module.checked_manifest(45, adapter)


def test_candidate_uses_frozen_confirmation_task_artifact():
    text = (ROOT / "scripts/evaluate_return_candidate.py").read_text()
    assert '"artifacts/position-v10-confirmation-tasks.jsonl"' in text
