from env.environment import EmailEnvironment
from env.models import Email, CalendarEvent

def get_task_hard_env() -> EmailEnvironment:
    """
    Sets up the Hard Task: Full Inbox Zero challenge.
    10 emails, cascading dependencies, requiring prioritize, delegate, reply, schedule.
    """
    env = EmailEnvironment(config={"max_steps": 100})
    
    # 10 emails simulating a real chaotic corporate inbox
    emails = [
        Email(id="h1", sender="hr@company.com", subject="Mandatory Security Training", body="Due Friday.", timestamp="08:00", urgency_ground_truth="low"),
        Email(id="h2", sender="ceo@company.com", subject="Board deck numbers", body="I need the Q3 revenue numbers updated in the deck before the 1 PM meeting.", timestamp="08:30", urgency_ground_truth="urgent", deadline="13:00"),
        Email(id="h3", sender="alice.dev@company.com", subject="Code review blocked", body="Can you review PR 402? I'm blocked.", timestamp="08:45", urgency_ground_truth="medium"),
        Email(id="h4", sender="vendor@saas.com", subject="Invoice #90210 Overdue", body="Your account will be suspended if not paid in 24 hrs.", timestamp="09:12", urgency_ground_truth="urgent", requires_delegation=True),
        Email(id="h5", sender="bob.design@company.com", subject="New mockups for homepage", body="Take a look when you have time.", timestamp="09:20", urgency_ground_truth="low"),
        Email(id="h6", sender="manager@company.com", subject="1-on-1 scheduling", body="Please put 30 mins on my calendar tomorrow.", timestamp="09:30", requires_scheduling=True, urgency_ground_truth="medium"),
        Email(id="h7", sender="marketing.spam@newsletter.com", subject="Buy our new tool", body="", timestamp="09:45", urgency_ground_truth="spam"),
        Email(id="h8", sender="legal@company.com", subject="Terms of Service update", body="Review by next week.", timestamp="10:00", urgency_ground_truth="low"),
        Email(id="h9", sender="customer.success@company.com", subject="Major client churn risk", body="Client X wants to cancel. Can we get on a call at 11 AM?", timestamp="10:15", urgency_ground_truth="urgent", requires_scheduling=True, deadline="11:00"),
        Email(id="h10", sender="sysadmin@company.com", subject="VPN Maintenance", body="VPN will be down Saturday.", timestamp="10:30", urgency_ground_truth="low")
    ]
    
    env.unwrapped.initial_emails_payload = emails
    env.unwrapped.initial_calendar = [
        CalendarEvent(title="Sprint Planning", time="10:00", participants=["Alice", "Bob"]),
        CalendarEvent(title="Lunch", time="12:00", participants=[])
    ]
    return env
