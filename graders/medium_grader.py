from typing import List, Dict, Any, Tuple

def grade_medium(history: List[Tuple[Dict[str, Any], float]], initial_emails: List[Any]) -> Dict[str, Any]:
    """
    Rubric-based grader for the Medium task.
    In a real implementation, this might use an LLM-as-a-judge (like GPT-4).
    For this baseline, we use deterministic heuristic checks to approximate a rubric:
    1. Brevity: Word count < 50
    2. Factual Accuracy: Contains key phrases related to context.
    3. Tone: Does not contain unprofessional words or overly brief/rude responses.
    """
    score = 0.0
    details = []
    
    reply_action = next((a for a, _ in history if a.get("action_type") == "reply"), None)
    
    if not reply_action:
        return {"score": 0.0, "details": ["Agent did not reply."]}
        
    reply_text = reply_action.get("reply_text", "")
    
    # 1. Brevity
    words = reply_text.split()
    if 5 <= len(words) <= 50:
        score += 0.33
        details.append("Brevity: Check passed.")
    else:
        details.append("Brevity: Failed (too long or too short).")
        
    # 2. Factual Accuracy (heuristics for 'client.vip@bigcorp.com' API v2 401 issue)
    key_phrases = ["api", "v2", "401", "auth", "token", "investigat", "look into"]
    if any(phrase in reply_text.lower() for phrase in key_phrases):
        score += 0.33
        details.append("Factual: Check passed (relevant keywords found).")
    else:
        details.append("Factual: Failed (missing relevant context).")
        
    # 3. Tone
    unprofessional_words = ["chill", "whatever", "dunno", "idiot"]
    if not any(word in reply_text.lower() for word in unprofessional_words) and "hello" in reply_text.lower() or "hi" in reply_text.lower():
        score += 0.34
        details.append("Tone: Check passed (professional greeting, no forbidden words).")
    else:
        details.append("Tone: Failed (missing greeting or unprofessional).")
        
    return {"score": min(1.0, score), "details": details}
