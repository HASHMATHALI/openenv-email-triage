import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from graders.easy_grader import grade_easy
from graders.hard_grader import grade_hard
from graders.medium_grader import grade_medium
from scripts.baseline_inference import run_agent
from tasks.easy import get_task_easy_env
from tasks.hard import get_task_hard_env
from tasks.medium import get_task_medium_env


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_reset_loads_task_emails_and_calendar():
    env = get_task_hard_env()
    obs, _ = env.reset()
    assert len(obs["inbox"]["emails"]) == 10
    assert len(obs["inbox"]["calendar"]) == 2


def test_unknown_email_and_calendar_conflict_are_penalised():
    env = get_task_hard_env()
    env.reset()
    _, reward, *_ = env.step({"action_type": "classify", "email_id": "nope", "classification": "low"})
    assert reward < 0
    _, reward, *_ = env.step({"action_type": "schedule", "email_id": "h9", "schedule_time": "10:00", "schedule_duration": 30})
    assert reward < 0  # Sprint Planning is already at 10:00
    _, reward, *_ = env.step({"action_type": "schedule", "email_id": "h9", "schedule_time": "11:00", "schedule_duration": 30})
    assert reward == 0


def test_invalid_action_schema_does_not_crash():
    env = get_task_easy_env()
    env.reset()
    obs, reward, terminated, truncated, _ = env.step({"action": "classify"})
    assert reward == -0.5 and not terminated and obs["last_action_success"] is False


def test_easy_grader_counts_each_email_once():
    env = get_task_easy_env()
    emails = env.unwrapped.initial_emails_payload
    spam_repeat = [({"action_type": "classify", "email_id": "email_1", "classification": "spam"}, 0.0)] * 10
    assert grade_easy(spam_repeat, emails)["score"] == pytest.approx(0.25)


def test_medium_grader_greeting_is_a_whole_word():
    emails = get_task_medium_env().unwrapped.initial_emails_payload
    rude = [({"action_type": "reply", "reply_text": "This is within our API v2 auth token roadmap, wait."}, 0.0)]
    polite = [({"action_type": "reply", "reply_text": "Hello, we are investigating the API v2 auth 401 now."}, 0.0)]
    assert not any("Tone: Check passed" in d for d in grade_medium(rude, emails)["details"])
    assert grade_medium(polite, emails)["score"] == pytest.approx(1.0)


def test_offline_baseline_scores():
    env = get_task_easy_env()
    history, initials = run_agent(env, max_steps=8)
    assert grade_easy(history, initials)["score"] == 1.0

    env = get_task_medium_env()
    history, initials = run_agent(env, max_steps=5)
    assert grade_medium(history, initials)["score"] == pytest.approx(1.0)

    env = get_task_hard_env()
    history, initials = run_agent(env, max_steps=15)
    result = grade_hard(history, initials, env.unwrapped.initial_calendar)
    assert result["score"] >= 5.0
