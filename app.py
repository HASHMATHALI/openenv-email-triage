import streamlit as st
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tasks.easy import get_task_easy_env
from tasks.medium import get_task_medium_env
from tasks.hard import get_task_hard_env
from scripts.baseline_inference import run_agent
from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard

st.set_page_config(page_title="OpenEnv Baseline Dashboard", page_icon="📧", layout="wide")

# Dashboard UI Styling
st.title("📧 Email Triage & Intelligent Workplace Assistant")
st.markdown("### Meta OpenEnv Hackathon Submission - Interactive Dashboard")
st.markdown("This dashboard wraps the standard `OpenEnv` Gym simulation and evaluates the baseline performance.")

st.info("Set GROQ_API_KEY or OPENAI_API_KEY to evaluate an LLM agent. Without a key, a deterministic offline heuristic agent runs instead.")

task_selection = st.selectbox("Select Benchmark Task Category", ["Easy (Urgency Classification)", "Medium (Triage & Reply)", "Hard (Inbox Zero)"])

if st.button("Run Simulation", width="stretch"):
    with st.spinner("Initializing OpenEnv Context and Agent Reasoning Loop..."):
        if "Easy" in task_selection:
            env = get_task_easy_env()
            st.subheader("📨 Initial Inbox State")
            st.dataframe([e.model_dump() for e in env.unwrapped.initial_emails_payload])
            
            history, initials = run_agent(env, max_steps=8)
            score = grade_easy(history, initials)
            
        elif "Medium" in task_selection:
            env = get_task_medium_env()
            st.subheader("📨 Initial Inbox State")
            st.dataframe([e.model_dump() for e in env.unwrapped.initial_emails_payload])
            
            history, initials = run_agent(env, max_steps=5)
            score = grade_medium(history, initials)
            
        else:
            env = get_task_hard_env()
            st.subheader("📨 Initial Inbox State")
            st.dataframe([e.model_dump() for e in env.unwrapped.initial_emails_payload])
            
            history, initials = run_agent(env, max_steps=15)
            score = grade_hard(history, initials, env.unwrapped.initial_calendar)

        st.subheader("🤖 Agent Action Trajectory")
        for step, (a_dict, reward) in enumerate(history):
            st.markdown(f"**Action {step+1}:** Agent executed `{a_dict.get('action_type')}` on email `{a_dict.get('email_id', 'None')}`")
            with st.expander("View Full Pydantic Action Payload"):
                st.json(a_dict)

        st.divider()
        st.subheader("📊 Grader Outcome Summary")
        if score['score'] > 0:
            st.success(f"Final Outcome Score: **{score['score']:.2f}**")
        else:
            st.warning(f"Final Outcome Score: **{score['score']:.2f}**")
        
        st.markdown("**Granular Grader Details:**")
        for det in score['details']:
            st.markdown(f"- {det}")
