import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import arena
from scripts.heuristic_agent import HeuristicAgent


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_human_perfect_easy_round():
    truth = {"email_1": "spam", "email_2": "low", "email_3": "medium", "email_4": "urgent"}
    actions = [{"action_type": "classify", "email_id": k, "classification": v} for k, v in truth.items()]
    result = arena.play("easy", "You", actions)
    assert result.score == 1.0 and result.percent == 1.0 and result.tags == []


def test_failure_tags_name_the_mistakes():
    actions = [
        {"action_type": "classify", "email_id": "ghost", "classification": "low"},
        {"action_type": "schedule", "email_id": "h6", "schedule_time": "10:00", "schedule_duration": 30},
        {"action_type": "delegate", "email_id": "h4", "delegate_to": "Zed"},
        {"action_type": "classify", "email_id": "h1", "classification": "low"},
        {"action_type": "classify", "email_id": "h1", "classification": "low"},
    ]
    tags = arena.play("hard", "You", actions).tags
    assert "acted on a phantom email" in tags and "double-booked the calendar" in tags
    assert "delegated to nobody" in tags and "repeated itself x1" in tags
    assert any(t.startswith("never touched") for t in tags)


def test_agent_run_and_shuffle_is_deterministic_per_seed():
    plain = arena.play_agent("hard", "bot", HeuristicAgent().act)
    assert plain.score >= 5.0 and plain.steps
    order = lambda seed: [e.id for e in arena.make_env("hard", seed).unwrapped.initial_emails_payload]
    assert order(3) == order(3) and order(3) != order(None)
    assert sorted(order(3)) == sorted(order(None))


def test_keyword_bot_does_not_depend_on_inbox_order():
    scores = {arena.play_agent("easy", "bot", HeuristicAgent().act, seed).score for seed in (None, 1, 2, 3)}
    assert scores == {1.0}


def test_step_budget_is_enforced():
    spam = [{"action_type": "classify", "email_id": "email_1", "classification": "spam"}] * 50
    assert len(arena.play("easy", "You", spam).steps) == arena.TASKS["easy"].max_steps
