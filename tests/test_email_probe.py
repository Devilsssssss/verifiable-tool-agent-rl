import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from toolrl.curriculum import lookup_training_row
from toolrl.generator import generate_tasks

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_email.py"
SPEC = importlib.util.spec_from_file_location("probe_email", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_lookup_probe_controls_cross_email_without_changing_request() -> None:
    train = generate_tasks({"train": 5}, seed=42)[0]
    heldout = generate_tasks({"validation": 5}, seed=42)[0]
    row = lookup_training_row(train, profile="v7")
    controls = MODULE.lookup_controls(heldout, row)
    unseen = heldout.initial_state["users"][0]["email"]
    seen = train.initial_state["users"][0]["email"]
    assert controls["original-wording-training-email"] == (
        heldout.prompt.replace(unseen, seen),
        seen,
    )
    prompt, expected = controls["lookup-wording-unseen-email"]
    assert expected == unseen and unseen in prompt and seen not in prompt
    assert controls["lookup-exact-training-row"] == (row["messages"][1]["content"], seen)
    with pytest.raises(ValueError, match="same-family training"):
        MODULE.lookup_controls(replace(heldout, family="return_expired"), row)
    with pytest.raises(ValueError, match="same-family training"):
        MODULE.lookup_controls(heldout, {**row, "split": "validation"})
