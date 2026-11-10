"""Deterministic, offline baseline agent.

Used when no LLM API key is configured so the environment, graders and
dashboard can be exercised end-to-end without network access. It is a simple
keyword policy, not a learned model - its scores are a floor to beat.
"""
import re
from typing import Any, Dict, List, Optional, Set

SPAM_HINTS = ("lottery", "claim your", "bank details", "buy our", "you have won", "newsletter.com", "scam")
URGENT_HINTS = ("sev-1", "sev1", "production", "down", "immediately", "overdue", "suspended", "churn",
                "blocking", "before the", "asap", "cancel")
MEDIUM_HINTS = ("blocked", "can we", "can you", "please put", "moved", "review pr", "meeting")
SCHEDULE_HINTS = ("call at", "on my calendar", "get on a call", "schedule a ")
DELEGATE_HINTS = ("invoice", "payment", "billing")
TIME_PATTERN = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)


def classify_email(email: Dict[str, Any]) -> str:
    text = f"{email.get('sender', '')} {email.get('subject', '')} {email.get('body', '')}".lower()
    if any(h in text for h in SPAM_HINTS):
        return "spam"
    if email.get("deadline") or any(h in text for h in URGENT_HINTS):
        return "urgent"
    if any(h in text for h in MEDIUM_HINTS):
        return "medium"
    return "low"


def _extract_time(text: str) -> Optional[str]:
    match = TIME_PATTERN.search(text)
    if not match:
        return None
    hour = int(match.group(1)) % 12
    if match.group(3).lower() == "pm":
        hour += 12
    return f"{hour:02d}:{match.group(2) or '00'}"


def _follow_up(email: Dict[str, Any], team: List[str]) -> Dict[str, Any]:
    text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
    if email.get("requires_scheduling") or any(h in text for h in SCHEDULE_HINTS):
        return {
            "action_type": "schedule",
            "email_id": email["id"],
            "schedule_time": _extract_time(text) or email.get("deadline") or "15:00",
            "schedule_duration": 30,
        }
    if team and (email.get("requires_delegation") or any(h in text for h in DELEGATE_HINTS)):
        return {"action_type": "delegate", "email_id": email["id"], "delegate_to": team[0]}
    subject = email.get("subject", "your message")
    return {
        "action_type": "reply",
        "email_id": email["id"],
        "reply_text": f"Hello, thanks for flagging \"{subject}\". We are investigating this now "
                      "and will send you an update within the hour.",
    }


class HeuristicAgent:
    """Handles urgent emails first (follow-up action), then classifies the rest."""

    def __init__(self) -> None:
        self.followed_up: Set[str] = set()
        self.classified: Set[str] = set()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        inbox = obs.get("inbox", {})
        emails = inbox.get("emails", [])
        team = inbox.get("team_members", [])
        labels = {e["id"]: classify_email(e) for e in emails}

        # 1. Urgent work first, earliest deadline first.
        urgent = sorted(
            (e for e in emails if labels[e["id"]] == "urgent" and e["id"] not in self.followed_up),
            key=lambda e: e.get("deadline") or "99:99",
        )
        if urgent:
            self.followed_up.add(urgent[0]["id"])
            return _follow_up(urgent[0], team)

        # 2. Then label everything.
        for email in emails:
            if email["id"] not in self.classified:
                self.classified.add(email["id"])
                return {"action_type": "classify", "email_id": email["id"], "classification": labels[email["id"]]}

        return {"action_type": "wait"}
