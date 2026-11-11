"""Inbox Arena: run any contestant (agent or human) through a task and grade it.

Pure logic, no Streamlit imports, so it is unit-testable.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from graders.easy_grader import grade_easy
from graders.hard_grader import grade_hard
from graders.medium_grader import grade_medium
from tasks.easy import get_task_easy_env
from tasks.hard import get_task_hard_env
from tasks.medium import get_task_medium_env


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    brief: str
    factory: Callable
    max_score: float
    max_steps: int
    allowed: tuple  # action types a human may choose for each email


TASKS: Dict[str, Task] = {
    "easy": Task("easy", "Sort the pile", "Label every email: spam, low, medium or urgent.",
                 get_task_easy_env, 1.0, 8, ("classify",)),
    "medium": Task("medium", "Answer the VIP", "One angry client. Reply briefly, accurately and politely.",
                   get_task_medium_env, 1.0, 5, ("reply", "classify")),
    "hard": Task("hard", "Inbox zero", "Ten emails, three fires. Deal with the urgent ones first and get the client call booked.",
                 get_task_hard_env, 10.0, 15, ("classify", "reply", "delegate", "schedule")),
}


@dataclass
class Step:
    action: Dict[str, Any]
    reward: float
    feedback: str
    ok: bool


@dataclass
class Result:
    task: str
    contestant: str
    score: float
    max_score: float
    details: List[str]
    steps: List[Step] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        return self.score / self.max_score if self.max_score else 0.0


def make_env(task_key: str, shuffle_seed: int | None = None):
    """Build the task environment; optionally shuffle inbox order.

    Shuffling is a robustness probe: a contestant that only does well because
    the urgent emails happen to sit at fixed positions will drop here.
    """
    env = TASKS[task_key].factory()
    if shuffle_seed is not None:
        emails = list(env.unwrapped.initial_emails_payload)
        random.Random(shuffle_seed).shuffle(emails)
        env.unwrapped.initial_emails_payload = emails
    return env


def _grade(task_key: str, env, history) -> Dict[str, Any]:
    emails = env.unwrapped.initial_emails_payload
    if task_key == "easy":
        return grade_easy(history, emails)
    if task_key == "medium":
        return grade_medium(history, emails)
    return grade_hard(history, emails, getattr(env.unwrapped, "initial_calendar", []))


def failure_tags(steps: List[Step], env) -> List[str]:
    """Name what went wrong, not just how much."""
    tags = []
    if any("Invalid action schema" in s.feedback for s in steps):
        tags.append("malformed action")
    if any("does not exist" in s.feedback for s in steps):
        tags.append("acted on a phantom email")
    if any("calendar conflict" in s.feedback for s in steps):
        tags.append("double-booked the calendar")
    if any("not a valid team member" in s.feedback for s in steps):
        tags.append("delegated to nobody")
    seen, repeats = set(), 0
    for s in steps:
        key = (s.action.get("action_type"), s.action.get("email_id"))
        repeats += key in seen
        seen.add(key)
    if repeats:
        tags.append(f"repeated itself x{repeats}")
    urgent = {e.id for e in env.unwrapped.initial_emails_payload if e.urgency_ground_truth == "urgent"}
    touched = {s.action.get("email_id") for s in steps}
    if urgent - touched:
        tags.append(f"never touched {len(urgent - touched)} urgent email(s)")
    return tags


def play(task_key: str, contestant: str, actions: List[Dict[str, Any]], shuffle_seed: int | None = None) -> Result:
    """Run a fixed list of actions (a human's choices) through the environment."""
    env = make_env(task_key, shuffle_seed)
    env.reset()
    steps = []
    for action in actions[: TASKS[task_key].max_steps]:
        obs, reward, terminated, truncated, _ = env.step(action)
        steps.append(Step(action, reward, obs.get("feedback") or "", bool(obs.get("last_action_success", True))))
        if terminated or truncated:
            break
    graded = _grade(task_key, env, env.history)
    return Result(task_key, contestant, float(graded["score"]), TASKS[task_key].max_score, graded["details"], steps, failure_tags(steps, env))


def play_agent(task_key: str, contestant: str, policy: Callable[[Dict[str, Any]], Dict[str, Any]],
               shuffle_seed: int | None = None) -> Result:
    """Run a policy (observation -> action) until it waits or runs out of steps."""
    env = make_env(task_key, shuffle_seed)
    obs, _ = env.reset()
    steps = []
    for _ in range(TASKS[task_key].max_steps):
        action = policy(obs)
        if action.get("action_type") == "wait":
            break
        obs, reward, terminated, truncated, _ = env.step(action)
        steps.append(Step(action, reward, obs.get("feedback") or "", bool(obs.get("last_action_success", True))))
        if terminated or truncated:
            break
    graded = _grade(task_key, env, env.history)
    return Result(task_key, contestant, float(graded["score"]), TASKS[task_key].max_score, graded["details"], steps, failure_tags(steps, env))
