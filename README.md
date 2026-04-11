# Email Triage & Intelligent Workplace Assistant: An OpenEnv Benchmark

## Abstract
Modern Language Models excel at isolated reasoning, yet deploying them as autonomous workplace agents exposes a devastating fragility: they fail to navigate the messy, asynchronous, and socially nuanced realities of corporate environments. We present the **Email Triage & Intelligent Workplace Assistant**, an open-source OpenEnv simulation tailored for evaluating autonomous agents in high-stakes office scenarios. This environment exposes the agent to realistic noise, complex dependency chains, and competing urgencies via a multidimensional reward function, proving that genuine autonomy requires more than just Next Token Prediction—it requires organizational intuition.

## Action and Observation Spaces

The environment strictly adheres to Pydantic v2 schemas defined in `env/models.py`. 

### Observation Space
The agent receives a full view of the simulated corporate system along with textual feedback from the last action.

| Feature | Description | Type |
| :--- | :--- | :--- |
| `inbox.emails` | Array of emails with metadata (thread history, deadlines). | `List[Email]` |
| `inbox.calendar` | The user's active calendar with conflicting events. | `List[CalendarEvent]` |
| `inbox.team_members`| Available workforce for delegation mapping. | `List[str]` |
| `current_time` | In-simulation global tick time. | `str` |
| `feedback` | Immediate textual consequence of the previous action. | `Optional[str]` |

### Action Space
Agents interact via a continuous loop of defined operational capabilities.

| Action Type | Required Metadata | Description |
| :--- | :--- | :--- |
| `classify` | `email_id`, `classification` | strict taxonomy mapping (`spam`, `low`, `medium`, `urgent`) |
| `reply` | `email_id`, `reply_text` | drafts communication balancing brevity, accuracy & tone |
| `delegate` | `email_id`, `delegate_to` | hands off asynchronous dependencies to team members |
| `schedule` | `schedule_time`, `schedule_duration`| handles calendar routing preventing critical overlaps |

## Difficulty Breakdown

The OpenEnv environment offers three incrementally scoped benchmarks. 

*   **Easy**: Strict static classification evaluating pure information extraction. High success rate expectation.
*   **Medium**: Tone and instruction following. Evaluates text generation within bounds governed by rubrics. 
*   **Hard (Inbox Zero)**: 10 emails. Competing dependencies. Requires the agent to anticipate constraints, schedule around blockers, and avoid cyclical loops. Evaluates multi-step decision horizons.

## Setup & Running the Baseline

Get started in under 3 commands:

```bash
# 1. Build the Docker environment
docker build -t openenv-email-triage .

# 2. Supply your API Key to the environment
export OPENAI_API_KEY="sk-..."

# 3. Run the evaluation baseline
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY openenv-email-triage
```

## Baseline Evaluation Results (`gpt-4o-mini`)

Using the native `scripts/baseline_inference.py`, we benchmarked a standard zero-shot LLM reasoning loop.

| Task Level | Score Achieved / Max | Notes & Failure Modes |
| :--- | :--- | :--- |
| **Easy** | 0.75 / 1.00 | Agent reliably finds 'spam', but sometimes misattributes context lacking deep history. |
| **Medium** | 0.66 / 1.00 | Generated replies are accurate and polite, but often fail strict brevity thresholds. |
| **Hard** | 1.00 / 10.00 | **Critical Failure**. The agent enters redundant loops attempting to reply linearly and inherently fails to delegate before scheduling. It lacks global temporal awareness. |

_Ready for Hugging Face Spaces `openenv` deployment out of the box._
