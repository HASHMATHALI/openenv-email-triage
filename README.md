# Inbox Arena — an email-triage benchmark you can play

An OpenEnv/Gymnasium environment for evaluating agents on realistic inbox work, with one twist:
**you can sit in the other corner.** The dashboard hands you the same inbox the agent gets, runs
your decisions through the same environment and scores you with the same graders.

Benchmarks usually report a number. Playing the round yourself is the fastest way to learn what
that number rewards, where the grader is strict, and whether an agent's 7/10 is impressive or lazy.

## What the arena adds

- **Player vs agent.** Choose an action and a handling order for every email, lock it in, then
  release the agent. Scores sit side by side with a play-by-play of each step and the referee's notes.
- **Named failures, not just scores.** Every run is tagged with what went wrong: *acted on a
  phantom email*, *double-booked the calendar*, *delegated to nobody*, *repeated itself*, *never
  touched an urgent email*.
- **Shuffle probe.** Re-deal the inbox with a seed. A contestant that depends on where the urgent
  emails sit, rather than what they say, loses points.
- **Any agent in the corner.** `GROQ_API_KEY` or `OPENAI_API_KEY` puts an LLM there; with neither, an
  offline keyword bot stands in so everything runs without network access.
- **Session board.** Every run in the session is kept for comparison.

```bash
pip install -r requirements.txt
streamlit run app.py                    # the arena
python scripts/baseline_inference.py    # headless benchmark, all three rounds
pip install pytest && pytest -q         # 11 tests
```

`arena.py` holds the runner (`play`, `play_agent`, `failure_tags`, `make_env`) with no UI imports,
so it doubles as a small harness for scripting your own agents:

```python
import arena
result = arena.play_agent("hard", "my-agent", my_policy, shuffle_seed=3)   # policy: observation dict -> action dict
print(result.score, result.tags)
```

---

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

### Local

```bash
pip install -r requirements.txt

# Headless benchmark (all three tasks)
python scripts/baseline_inference.py

# Interactive dashboard
streamlit run app.py

# Tests
pip install pytest && pytest -q
```

### Choosing the agent

| Environment variable | Agent used |
| :--- | :--- |
| `GROQ_API_KEY` | Groq, `llama-3.3-70b-versatile` (override with `MODEL_NAME`) |
| `OPENAI_API_KEY` | OpenAI, `gpt-4o-mini` (override with `MODEL_NAME`) |
| *(neither)* | Deterministic offline keyword agent in `scripts/heuristic_agent.py` |

### Docker / Hugging Face Spaces

```bash
docker build -t openenv-email-triage .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY openenv-email-triage   # dashboard on :7860
docker run --rm openenv-email-triage python scripts/baseline_inference.py           # headless benchmark
```

## Baseline Results

Reproducible offline floor (no API key, `scripts/heuristic_agent.py`):

| Task Level | Score / Max |
| :--- | :--- |
| **Easy** | 1.00 / 1.00 |
| **Medium** | 1.00 / 1.00 |
| **Hard** | 7.00 / 10.00 |

The keyword agent is tuned to these fixed task emails, so treat it as a smoke test of the
environment and graders rather than a measure of difficulty. LLM agents are evaluated by
setting one of the API keys above.

_Ready for Hugging Face Spaces `openenv` deployment out of the box._
