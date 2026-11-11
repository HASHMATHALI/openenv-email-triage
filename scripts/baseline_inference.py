import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from tasks.easy import get_task_easy_env
from tasks.medium import get_task_medium_env
from tasks.hard import get_task_hard_env
from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard
from env.models import AgentAction
from scripts.heuristic_agent import HeuristicAgent


def format_openai_prompt(obs: dict) -> str:
    """Format the observation space into an LLM prompt."""
    schema = AgentAction.model_json_schema()
    return f"""
    You are an intelligent workplace assistant managing an email inbox.
    Current Time: {obs['current_time']}
    Feedback from last action: {obs['feedback']}

    Inbox State:
    {json.dumps(obs['inbox'], indent=2)}

    Output a single valid JSON object representing your next action. It MUST strictly match this Pydantic schema:
    {json.dumps(schema, indent=2)}

    Crucially, the key for the action type is 'action_type', NOT 'action'.
    Allowed action_types: classify, reply, delegate, schedule, read, wait.
    """


def _build_agent():
    """Returns (client, model, fallback_agent).

    GROQ_API_KEY -> Groq (OpenAI-compatible endpoint); OPENAI_API_KEY -> OpenAI.
    With neither set, a deterministic offline heuristic agent is used instead.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if groq_key:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        return client, os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"), None
    if openai_key:
        return OpenAI(api_key=openai_key), os.environ.get("MODEL_NAME", "gpt-4o-mini"), None
    print("No GROQ_API_KEY / OPENAI_API_KEY set. Using the offline heuristic agent.")
    return None, None, HeuristicAgent()


def make_policy():
    """Returns (name, policy) where policy maps an observation dict to an action dict.

    Used by the Inbox Arena dashboard. A policy that cannot produce a valid
    action returns {"action_type": "wait"}, which ends its run.
    """
    client, model, fallback_agent = _build_agent()
    if fallback_agent is not None:
        return "Keyword bot (offline)", fallback_agent.act

    def policy(obs: dict) -> dict:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": format_openai_prompt(obs)}],
                response_format={"type": "json_object"},
            )
            action = json.loads(response.choices[0].message.content)
            AgentAction(**action)
            return action
        except Exception as e:
            print(f"Agent failed to generate a valid action: {e}")
            return {"action_type": "wait"}

    return f"LLM ({model})", policy


def run_agent(env, max_steps=10):
    client, model, fallback_agent = _build_agent()

    # reset() loads the task's emails and calendar into the inbox.
    obs, info = env.reset()

    for step in range(max_steps):
        try:
            if fallback_agent is not None:
                action_dict = fallback_agent.act(obs)
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": format_openai_prompt(obs)}],
                    response_format={"type": "json_object"},
                )
                action_dict = json.loads(response.choices[0].message.content)

            # Validate schema
            AgentAction(**action_dict)

        except Exception as e:
            print(f"Agent failed to generate a valid action: {e}")
            break

        if fallback_agent is not None and action_dict.get("action_type") == "wait":
            break  # the heuristic agent has nothing left to do

        obs, reward, terminated, truncated, info = env.step(action_dict)
        if terminated or truncated:
            break

    return env.history, env.unwrapped.initial_emails_payload


def main():
    print("--- Running Easy Task ---")
    easy_env = get_task_easy_env()
    history, initials = run_agent(easy_env, max_steps=8)
    easy_score = grade_easy(history, initials)
    print(f"Easy Score: {easy_score['score']}\nDetails: {easy_score['details']}\n")

    print("--- Running Medium Task ---")
    medium_env = get_task_medium_env()
    history, initials = run_agent(medium_env, max_steps=5)
    medium_score = grade_medium(history, initials)
    print(f"Medium Score: {medium_score['score']}\nDetails: {medium_score['details']}\n")

    print("--- Running Hard Task ---")
    hard_env = get_task_hard_env()
    history, initials = run_agent(hard_env, max_steps=15)
    calendar = hard_env.unwrapped.initial_calendar
    hard_score = grade_hard(history, initials, calendar)
    print(f"Hard Score: {hard_score['score']}\nDetails: {hard_score['details']}\n")

    print("--- Final Evaluation ---")
    total = easy_score['score'] + medium_score['score'] + hard_score['score']
    print(f"Total Score (across 3 tasks): {total:.2f}")


if __name__ == "__main__":
    main()
