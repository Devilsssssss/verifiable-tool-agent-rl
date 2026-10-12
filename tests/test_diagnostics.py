from dataclasses import replace

import pytest

from toolrl.diagnostics import (
    ExplicitEmailPrompt,
    OracleFirstLookup,
    correct_lookup,
    factorial_cases,
)
from toolrl.expert import replay_expert
from toolrl.factorial_training import training_variants
from toolrl.generator import generate_tasks
from toolrl.rollout import run_episode


def test_factorial_crosses_all_cells_and_rejects_heldout_tasks():
    task = generate_tasks({"train": 5}, seed=42)[0]
    cases = factorial_cases(task)
    assert len(cases) == 20
    assert len({(row["frame"], row["email_structure"]) for row in cases}) == 20
    for structure in {row["email_structure"] for row in cases}:
        rows = [row for row in cases if row["email_structure"] == structure]
        assert len({row["expected_email"] for row in rows}) == 1
        assert all(row["expected_email"] in row["prompt"] for row in rows)
    assert factorial_cases(task) == cases
    with pytest.raises(ValueError, match="training tasks"):
        factorial_cases(replace(task, split="validation"))


def test_oracle_only_supplies_first_call_and_keeps_strict_scoring():
    task = generate_tasks({"train": 5}, seed=42)[0]
    email = task.initial_state["users"][0]["email"]
    calls = []

    def bad_policy(messages, tools):
        calls.append(messages)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "nonexistent", "arguments": {}}}],
        }

    report = run_episode(task, OracleFirstLookup(bad_policy, email), max_turns=2)
    assert len(calls) == 1
    assert calls[0][-1]["role"] == "tool"
    assert report["steps"] == 2
    assert not report["success"]
    assert "invalid_tool_call" in report["violations"]
    assert report["trace"][0]["tool"] == "lookup_user"


def test_lookup_diagnostic_rejects_extra_arguments_and_multiple_calls():
    response = {
        "tool_calls": [{"function": {"name": "lookup_user", "arguments": {"email": "a@test"}}}]
    }
    assert correct_lookup(response, "a@test")
    assert not correct_lookup(response, "b@test")
    response["tool_calls"][0]["function"]["arguments"]["user_id"] = 1
    assert not correct_lookup(response, "a@test")
    assert not correct_lookup({"tool_calls": response["tool_calls"] * 2}, "a@test")


def test_factorial_training_keeps_state_and_expert_bound_to_email():
    for task in generate_tasks({"train": 5}, seed=42):
        original = task.to_dict()
        variants = training_variants(task)
        assert len({variant.task_id for variant in variants}) == 20
        for variant in variants:
            email = variant.initial_state["users"][0]["email"]
            assert email in variant.prompt
            assert variant.expert_plan[0]["arguments"] == {"email": email}
            assert "subscriber." not in email
            assert replay_expert(variant)["reward_report"]["success"]
        assert task.to_dict() == original
    with pytest.raises(ValueError, match="training tasks"):
        training_variants(replace(task, split="validation"))


def test_email_field_uses_user_text_without_inventing_calls_or_mutating_history():
    history = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "For account a.0001@example.test, cancel my order."},
    ]
    observed = []
    final = {"role": "assistant", "content": "model answer"}

    def policy(messages, tools):
        observed.append(messages)
        return final

    assert ExplicitEmailPrompt(policy)(history, []) is final
    assert observed[0][1]["content"] == (
        "Customer email: a.0001@example.test\nRequest: " + history[1]["content"]
    )
    assert ExplicitEmailPrompt(policy, variant="v2")(history, []) is final
    assert observed[1][1]["content"] == (
        "Request: " + history[1]["content"] + "\nCustomer email: a.0001@example.test"
    )
    assert history[1]["content"].startswith("For account")
    with pytest.raises(ValueError, match="one email"):
        ExplicitEmailPrompt(policy)(history[:1] + [{"role": "user", "content": "No email."}], [])
