from typing import List, Dict, Any, Tuple

def grade_hard(history: List[Tuple[Dict[str, Any], float]], initial_emails: List[Any], initial_calendar: List[Any]) -> Dict[str, Any]:
    """
    Grader for the Hard sequence simulation.
    Rewards partial credit for correct priority execution, penalizes for loops.
    """
    score = 0.0
    details = []
    
    # Analyze trajectory
    actions_taken = [a.get("action_type") for a, _ in history]
    
    # Penalty for loop / redundancy
    if len(actions_taken) > 15:
        score -= 0.5
        details.append("Penalty: Iteration limit exceeded (possible loop).")
    
    # Track when urgent emails were handled
    urgent_handles = []
    for step, (a, _) in enumerate(history):
        if a.get("action_type") in ["reply", "delegate", "schedule"]:
            eid = a.get("email_id")
            # If this is one of our urgent ones
            if eid in ["h2", "h4", "h9"]:
                if eid not in urgent_handles:
                    urgent_handles.append((eid, step))
                    
    # Bonus for early handling
    for eid, step in urgent_handles:
        if step < 5:
            score += 2.0
            details.append(f"Bonus: Urgent email {eid} handled early in trajectory.")
        else:
            score += 1.0
            details.append(f"Credit: Urgent email {eid} handled.")
            
    # Check dependencies (e.g., h9 scheduling)
    has_scheduled_h9 = any(a.get("action_type") == "schedule" and a.get("email_id") == "h9" for a, _ in history)
    if has_scheduled_h9:
        score += 1.0
        details.append("Credit: Properly scheduled h9.")
    else:
        score -= 1.0
        details.append("Penalty: Failed to schedule h9 meeting before deadline.")
        
    return {"score": max(0.0, min(10.0, score)), "details": details}
