from env.environment import EmailEnvironment
from env.models import Email

def get_task_medium_env() -> EmailEnvironment:
    """
    Sets up the Medium Task: Triaging and drafting a reply.
    """
    env = EmailEnvironment(config={"max_steps": 25})
    
    emails = [
        Email(
            id="email_m1",
            sender="client.vip@bigcorp.com",
            subject="Integration issue with API v2",
            body="Hi team, we're trying to migrate to API v2 as discussed last week but the '/auth' endpoint returns 401 on refresh tokens. This is blocking our launch scheduled for tomorrow. We need help immediately.",
            timestamp="10:00",
            urgency_ground_truth="urgent",
            thread_history=["Last week: Discussed v2 migration and shared documentation link."]
        )
    ]
    
    env.unwrapped.initial_emails_payload = emails
    return env
