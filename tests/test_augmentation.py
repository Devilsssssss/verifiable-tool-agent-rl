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


def test_numeric_email_augmentation_replays_and_has_no_reserved_split_prefix() -> None:
    tasks = generate_tasks({"train": 60, "validation": 10, "test": 10})
    varied = [
        diversify_training_task(task, profile="v8") for task in tasks if task.split == "train"
    ]
    assert any(task.initial_state["users"][0]["email"].endswith("@example.test") for task in varied)
    assert any(task.initial_state["users"][0]["email"].endswith("@mailbox.test") for task in varied)
    for task in varied:
        email = task.initial_state["users"][0]["email"]
        assert not email.startswith(("train.", "validation.", "test."))
        assert replay_expert(task)["reward_report"]["success"]
    assert entity_tokens(varied).isdisjoint(
        entity_tokens(task for task in tasks if task.split != "train")
    )
