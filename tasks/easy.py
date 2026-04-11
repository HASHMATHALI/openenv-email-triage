from env.environment import EmailEnvironment
from env.models import Email

def get_task_easy_env() -> EmailEnvironment:
    """
    Sets up the Easy Task: Urgency Classification.
    Provides a batch of emails that the agent must strictly classify.
    """
    env = EmailEnvironment(config={"max_steps": 10})
    
    # Ground truth data setup
    emails = [
        Email(
            id="email_1",
            sender="prince.nigeria@scam.com",
            subject="URGENT: Claim your $1,000,000",
            body="You have won the lottery! Please send your bank details immediately to claim.",
            timestamp="09:00",
            urgency_ground_truth="spam"
        ),
        Email(
            id="email_2",
            sender="newsletter@marketing.brand.com",
            subject="Weekly Digest: Top 10 tips for clean code",
            body="Here are our top tips for writing better Python...",
            timestamp="09:15",
            urgency_ground_truth="low"
        ),
        Email(
            id="email_3",
            sender="boss@company.com",
            subject="Project status meeting moved?",
            body="Can we move the sync to 3 PM instead of 2 PM today?",
            timestamp="09:30",
            urgency_ground_truth="medium"
        ),
        Email(
            id="email_4",
            sender="pagerduty@company.com",
            subject="[SEV-1] Production Database Down",
            body="Investigating immediate latency spikes and db connection failures on primary cluster.",
            timestamp="09:40",
            urgency_ground_truth="urgent"
        )
    ]
    
    # Normally, an environment is reset via `env.reset(options={"initial_emails": emails})`
    # We will let the baseline script call reset(). This function just holds the config state.
    env.unwrapped.initial_emails_payload = emails
    return env
