from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    urgency_ground_truth: Optional[Literal["spam", "low", "medium", "urgent"]] = None
    thread_history: List[str] = Field(default_factory=list)
    requires_scheduling: bool = False
    requires_delegation: bool = False
    deadline: Optional[str] = None

class CalendarEvent(BaseModel):
    title: str
    time: str
    participants: List[str]

class TaskDelegate(BaseModel):
    task_id: str
    assignee: str
    status: Literal["pending", "completed"]

class InboxState(BaseModel):
    emails: List[Email]
    calendar: List[CalendarEvent] = Field(default_factory=list)
    team_members: List[str] = Field(default_factory=list)
    delegated_tasks: List[TaskDelegate] = Field(default_factory=list)

class AgentAction(BaseModel):
    action_type: Literal["classify", "reply", "delegate", "schedule", "read", "archive", "wait"]
    email_id: Optional[str] = None
    classification: Optional[Literal["spam", "low", "medium", "urgent"]] = None
    reply_text: Optional[str] = None
    delegate_to: Optional[str] = None
    schedule_time: Optional[str] = None
    schedule_duration: Optional[int] = None # in minutes

class ObservationSpace(BaseModel):
    inbox: InboxState
    current_time: str
    feedback: Optional[str] = None
    last_action_success: bool = True
