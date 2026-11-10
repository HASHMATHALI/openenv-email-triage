import gymnasium as gym
from typing import Any, Dict, Tuple, Optional
from .models import InboxState, AgentAction, ObservationSpace, CalendarEvent, TaskDelegate

class EmailEnvironment(gym.Env):
    """
    Core Gymnasium-style interface for the Email Triage OpenEnv instance.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # We define a custom action space & observation space schema using Pydantic,
        # but to satisfy Gymnasium, we could map these to discrete/box spaces or 
        # use Custom spaces. For OpenEnv text-based agents, it's common to use Dict or 
        # just handle Python objects if the framework allows it. Here, we'll keep 
        # it simple and assume the framework interacts with step(action_dict).
        self.action_space = gym.spaces.Dict({}) 
        self.observation_space = gym.spaces.Dict({})
        
        self.state: Optional[InboxState] = None
        self.current_step = 0
        self.max_steps = self.config.get("max_steps", 50)
        self.history = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.history = []
        
        # Task factories attach their data as `initial_emails_payload` / `initial_calendar`;
        # explicit reset options take precedence.
        options = options or {}
        initial_emails = options.get("initial_emails", getattr(self, "initial_emails_payload", []))
        initial_calendar = options.get("initial_calendar", getattr(self, "initial_calendar", []))
        self.state = InboxState(
            emails=[e.model_copy(deep=True) for e in initial_emails],
            calendar=[c.model_copy(deep=True) for c in initial_calendar],
            team_members=["Alice", "Bob"],
            delegated_tasks=[],
        )
        
        obs = self._get_obs()
        info = self._get_info()
        return obs.model_dump(), info

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes an action in the environment.
        """
        try:
            action = AgentAction(**action_dict)
        except Exception as e:
            # Invalid action
            obs = self._get_obs(feedback=f"Invalid action schema: {str(e)}", success=False)
            return obs.model_dump(), -0.5, False, False, self._get_info()

        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        feedback = f"Action {action.action_type} executed."
        
        known_ids = {e.id for e in self.state.emails}
        needs_email = action.action_type in ("classify", "reply", "delegate", "read", "archive")

        # Handle state transitions based on action type
        if needs_email and action.email_id not in known_ids:
            feedback = f"Error: email '{action.email_id}' does not exist in the inbox."
            reward -= 0.5  # Penalty for acting on a non-existent email

        elif action.action_type == "classify":
            feedback = f"Classified email {action.email_id} as {action.classification}."
            # the grader handles reward for classification, no immediate reward here
            
        elif action.action_type == "reply":
            feedback = f"Replied to email {action.email_id}."
            
        elif action.action_type == "delegate":
            if action.delegate_to not in self.state.team_members:
                feedback = f"Error: {action.delegate_to} is not a valid team member."
                reward -= 0.5  # Penalty for invalid action
            else:
                feedback = f"Delegated email {action.email_id} to {action.delegate_to}."
                self.state.delegated_tasks.append(
                    TaskDelegate(task_id=action.email_id, assignee=action.delegate_to, status="pending")
                )
                
        elif action.action_type == "schedule":
            if not action.schedule_time:
                feedback = "Error: schedule requires schedule_time."
                reward -= 0.5
            elif any(event.time == action.schedule_time for event in self.state.calendar):
                feedback = f"Error: calendar conflict at {action.schedule_time}."
                reward -= 0.5
            else:
                feedback = f"Scheduled meeting at {action.schedule_time} for {action.schedule_duration} mins."
                self.state.calendar.append(
                    CalendarEvent(title=f"Meeting re: {action.email_id or 'ad hoc'}", time=action.schedule_time, participants=[])
                )

        self.history.append((action.model_dump(), reward))
        obs = self._get_obs(feedback=feedback, success=reward >= 0)
        
        # In a real inbox, time might advance or new emails arrive here.
        return obs.model_dump(), reward, terminated, truncated, self._get_info()

    def _get_obs(self, feedback: str = None, success: bool = True) -> ObservationSpace:
        return ObservationSpace(
            inbox=self.state,
            current_time=f"Day 1, Step {self.current_step}",
            feedback=feedback,
            last_action_success=success
        )

    def _get_info(self) -> Dict[str, Any]:
        return {"current_step": self.current_step, "history_len": len(self.history)}
