import pytest

from toolrl.augmentation import diversify_training_task
from toolrl.expert import replay_expert
from toolrl.generator import entity_tokens, generate_tasks


def test_augmented_tasks_preserve_goals_and_replay_across_all_families() -> None:
    tasks = generate_tasks({"train": 15, "validation": 5, "test": 5})
    augmented = []
    for task in tasks:
        if task.split != "train":
            with pytest.raises(ValueError, match="training"):
                diversify_training_task(task)
            continue
        varied = diversify_training_task(task)
        assert varied == diversify_training_task(task)
        assert varied.goal == task.goal
        assert varied.max_steps == task.max_steps
        assert varied.initial_state["users"][0]["email"] in varied.prompt
        assert varied.initial_state["products"][0]["name"] in varied.prompt
        assert replay_expert(varied)["reward_report"]["success"]
        assert task.initial_state["users"][0]["email"].startswith("train.")
        augmented.append(varied)
    held_out = [task for task in tasks if task.split != "train"]
    assert entity_tokens(augmented).isdisjoint(entity_tokens(held_out))
