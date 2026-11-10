from typing import List, Dict, Any, Tuple

def grade_easy(history: List[Tuple[Dict[str, Any], float]], initial_emails: List[Any]) -> Dict[str, Any]:
    """
    Deterministic grader for the Easy task.
    Expects actions to be 'classify' with the correct ground truth.
    """
    gt_map = {e.id: e.urgency_ground_truth for e in initial_emails}
    score = 0.0
    penalties = 0.0
    total = len(gt_map)
    details = []

    graded = set()  # only the first classification of each email counts
    for action_dict, reward in history:
        if action_dict.get("action_type") == "classify":
            eid = action_dict.get("email_id")
            if eid not in gt_map or eid in graded:
                continue
            graded.add(eid)
            pred = action_dict.get("classification")
            gt = gt_map.get(eid)
            
            if gt == pred:
                score += 1.0
                details.append(f"Correct: {eid} -> {pred}")
            else:
                if gt == "urgent" and pred == "spam":
                    penalties += 1.0
                    details.append(f"CRITICAL ERROR: {eid} (urgent) marked as spam.")
                else:
                    details.append(f"Incorrect: {eid} -> {pred} (expected {gt})")
                    
    final_score = min(1.0, max(0.0, (score - penalties) / total if total > 0 else 0))
    
    return {
        "score": final_score,
        "details": details
    }
