import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import ValidationError
from openai import OpenAI
from tasks.easy import get_task_easy_env
from tasks.medium import get_task_medium_env
from tasks.hard import get_task_hard_env
from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard
from env.models import AgentAction

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

def run_agent(env, max_steps=10):
    api_key = os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy-key"))
    
    # We use the OpenAI client but point it to Groq's high-speed inference engine!
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    
    obs, info = env.reset()
    
    for step in range(max_steps):
        prompt = format_openai_prompt(obs)
        
        try:
            if api_key == "dummy-key":
                print("Running without real OpenAI API Key. Mocking a random action...")
                mock_action = {
                    "action_type": "classify",
                    "email_id": env.state.emails[0].id if env.state.emails else "unknown",
                    "classification": "urgent"
                }
                action_text = json.dumps(mock_action)
            else:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                action_text = response.choices[0].message.content
                
            action_dict = json.loads(action_text)
            # Validate schema
            AgentAction(**action_dict)
            
        except Exception as e:
            print(f"Agent failed to generate a valid action: {e}")
            break
            
        obs, reward, terminated, truncated, info = env.step(action_dict)
        if terminated or truncated:
            break
            
    return env.history, env.unwrapped.initial_emails_payload

def main():
    print("--- Running Easy Task ---")
    easy_env = get_task_easy_env()
    history, initials = run_agent(easy_env, max_steps=5)
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
