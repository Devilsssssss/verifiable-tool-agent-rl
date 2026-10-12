import json

import pytest

from toolrl.curriculum import lookup_training_row
from toolrl.environment import RetailToolEnvironment
from toolrl.expert import reset_environment
from toolrl.formatting import sft_conversation
from toolrl.generator import generate_tasks


def test_lookup_curriculum_preserves_real_training_entities_and_masks_context() -> None:
    for task in generate_tasks({"train": 15, "validation": 5, "test": 5}):
        if task.split != "train":
            with pytest.raises(ValueError, match="training"):
                lookup_training_row(task)
            continue
        row = lookup_training_row(task)
        assert row == lookup_training_row(task)
        formatted = sft_conversation(row, [])
        assert len(formatted["prompt"]) == 2
        assert len(formatted["completion"]) == 1
        email = task.initial_state["users"][0]["email"]
        assert email in row["messages"][1]["content"]
        env = RetailToolEnvironment()
        reset_environment(env, task)
        result = json.loads(env.lookup_user(email))
        assert result["user"]["id"] == task.initial_state["users"][0]["id"]
        assert not env.reward_report["terminal_goal_satisfied"]
