"""Inbox Arena - you against the agent, same inbox, same referee."""
import html
import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import arena
from scripts.baseline_inference import make_policy

NIGHT, PANEL, INKY, PINK, LIME, GOLD, GREY = "#14101F", "#1E1830", "#F4F0FF", "#FF3D8B", "#B6FF3C", "#FFD23F", "#8C84A8"

st.set_page_config(page_title="Inbox Arena", page_icon="🥊", layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');
.stApp {{ background:{NIGHT}; }}
html, body, [class*="css"], .stMarkdown p, label {{ font-family:'DM Sans',sans-serif; color:{INKY}; }}
.marquee {{ font-family:'Silkscreen',monospace; font-size:2.6rem; line-height:1; color:{INKY}; text-shadow:4px 4px 0 {PINK}; }}
.sub {{ color:{GREY}; margin:8px 0 18px; }}
.px {{ font-family:'Silkscreen',monospace; letter-spacing:.04em; }}
.card {{ background:{PANEL}; border:2px solid #2E2646; padding:14px 16px; margin-bottom:10px; box-shadow:4px 4px 0 #0B0814; }}
.card .from {{ color:{GREY}; font-size:.8rem; }} .card .subj {{ font-weight:700; margin:2px 0 4px; }}
.card .body {{ color:#CFC8E6; font-size:.92rem; }}
.score {{ background:{PANEL}; border:2px solid; padding:14px 18px; text-align:center; box-shadow:4px 4px 0 #0B0814; }}
.score .who {{ font-family:'Silkscreen',monospace; font-size:.8rem; color:{GREY}; }}
.score .num {{ font-family:'Silkscreen',monospace; font-size:2.4rem; line-height:1.1; }}
.bar {{ height:10px; background:#2E2646; margin-top:8px; }} .bar > div {{ height:10px; }}
.tag {{ display:inline-block; font-family:'Silkscreen',monospace; font-size:.68rem; padding:2px 8px; margin:3px 5px 0 0; border:1px solid {PINK}; color:{PINK}; }}
.step {{ font-size:.9rem; padding:5px 0; border-bottom:1px dashed #2E2646; }} .step b {{ font-family:'Silkscreen',monospace; font-weight:400; }}
.stButton > button {{ font-family:'Silkscreen',monospace; border-radius:0; border:2px solid {LIME}; color:{LIME}; background:transparent; box-shadow:4px 4px 0 #0B0814; }}
.stButton > button:hover {{ background:{LIME}; color:{NIGHT}; border-color:{LIME}; }}
</style>
<div class="marquee">INBOX ARENA</div>
<div class="sub">An OpenEnv email-triage benchmark you can play. Same inbox, same referee - can you outscore the agent?</div>
""",
    unsafe_allow_html=True,
)

if "board" not in st.session_state:
    st.session_state.board = []   # every finished run this session
if "last" not in st.session_state:
    st.session_state.last = {}    # latest Result per (task, side)

# ------------------------------------------------------------------- set-up
left, mid, right = st.columns([2, 2, 1])
task_key = left.radio("Round", list(arena.TASKS), format_func=lambda k: f"{arena.TASKS[k].title}", horizontal=True)
task = arena.TASKS[task_key]
shuffle = mid.toggle("Shuffle the inbox", help="Reorders the emails with a seed. A contestant that relies on position rather than content will slip.")
seed = right.number_input("Seed", 0, 9999, 7, disabled=not shuffle) if shuffle else None
st.markdown(f"<span class='px' style='color:{GOLD}'>BRIEF</span> &nbsp;{task.brief}", unsafe_allow_html=True)

env = arena.make_env(task_key, seed)
emails = env.unwrapped.initial_emails_payload
calendar = getattr(env.unwrapped, "initial_calendar", [])
round_id = f"{task_key}:{seed}"

you_col, agent_col = st.columns(2)


def scorecard(col, who: str, result, colour: str):
    if result is None:
        col.markdown(f"<div class='score' style='border-color:#2E2646'><div class='who'>{who}</div><div class='num' style='color:#2E2646'>--</div></div>", unsafe_allow_html=True)
        return
    col.markdown(
        f"<div class='score' style='border-color:{colour}'><div class='who'>{html.escape(who)}</div>"
        f"<div class='num' style='color:{colour}'>{result.score:.2f}<span style='font-size:1rem;color:{GREY}'> / {result.max_score:g}</span></div>"
        f"<div class='bar'><div style='width:{max(0, min(1, result.percent)) * 100:.0f}%;background:{colour}'></div></div></div>",
        unsafe_allow_html=True,
    )
    if result.tags:
        col.markdown("".join(f"<span class='tag'>{html.escape(t)}</span>" for t in result.tags), unsafe_allow_html=True)


def replay(col, result):
    with col.expander("Play-by-play and referee notes"):
        for i, s in enumerate(result.steps, 1):
            a = s.action
            extra = a.get("classification") or a.get("delegate_to") or a.get("schedule_time") or (a.get("reply_text") or "")[:60]
            mark = "" if s.ok else f" <span style='color:{PINK}'>({s.reward:+.1f})</span>"
            st.markdown(f"<div class='step'><b>{i:02d}</b> {a.get('action_type')} · {html.escape(str(a.get('email_id')))} "
                        f"{'· ' + html.escape(str(extra)) if extra else ''}{mark}<br><span style='color:{GREY}'>{html.escape(s.feedback)}</span></div>", unsafe_allow_html=True)
        st.markdown("**Referee**")
        for d in result.details:
            st.markdown(f"- {d}")


# ---------------------------------------------------------------- your side
with you_col:
    st.markdown(f"<span class='px' style='color:{LIME}'>PLAYER 1 · YOU</span>", unsafe_allow_html=True)
    if calendar:
        st.caption("Already on the calendar: " + ", ".join(f"{c.title} {c.time}" for c in calendar))
    with st.form(f"you-{round_id}"):
        choices = []
        for n, e in enumerate(emails, 1):
            st.markdown(
                f"<div class='card'><div class='from'>{html.escape(e.timestamp)} · {html.escape(e.sender)}"
                f"{' · due ' + html.escape(e.deadline) if e.deadline else ''}</div><div class='subj'>{html.escape(e.subject)}</div>"
                f"<div class='body'>{html.escape(e.body) or '<i>(no body)</i>'}</div></div>", unsafe_allow_html=True)
            cols = st.columns([1, 2, 3])
            order = cols[0].number_input("Do it", 1, len(emails), n, key=f"o-{round_id}-{e.id}", help="1 = handle first")
            verbs = ("skip",) + task.allowed if len(emails) > 1 else task.allowed
            verb = cols[1].selectbox("Action", verbs, key=f"v-{round_id}-{e.id}", index=1 if len(emails) > 1 else 0)
            if verb == "classify":
                detail = cols[2].selectbox("As", ["spam", "low", "medium", "urgent"], index=1, key=f"d-{round_id}-{e.id}-c")
                action = {"action_type": "classify", "email_id": e.id, "classification": detail}
            elif verb == "reply":
                detail = cols[2].text_area("Reply", key=f"d-{round_id}-{e.id}-r", height=80, placeholder="Hello, …")
                action = {"action_type": "reply", "email_id": e.id, "reply_text": detail}
            elif verb == "delegate":
                detail = cols[2].selectbox("To", ["Alice", "Bob"], key=f"d-{round_id}-{e.id}-g")
                action = {"action_type": "delegate", "email_id": e.id, "delegate_to": detail}
            elif verb == "schedule":
                detail = cols[2].text_input("At (HH:MM)", "11:00", key=f"d-{round_id}-{e.id}-s")
                action = {"action_type": "schedule", "email_id": e.id, "schedule_time": detail, "schedule_duration": 30}
            else:
                action = None
            if action:
                choices.append((order, n, action))
        submitted = st.form_submit_button("LOCK IN MY MOVES")
    if submitted:
        ordered = [a for _, _, a in sorted(choices, key=lambda c: (c[0], c[1]))]
        result = arena.play(task_key, "You", ordered, seed)
        st.session_state.last[(round_id, "you")] = result
        st.session_state.board.append(result)

# --------------------------------------------------------------- agent side
with agent_col:
    name, policy = make_policy()
    st.markdown(f"<span class='px' style='color:{PINK}'>PLAYER 2 · {html.escape(name.upper())}</span>", unsafe_allow_html=True)
    st.caption("Set GROQ_API_KEY or OPENAI_API_KEY to put an LLM in this corner. Without one, a keyword bot stands in.")
    if st.button("RELEASE THE AGENT"):
        result = arena.play_agent(task_key, name, policy, seed)
        st.session_state.last[(round_id, "agent")] = result
        st.session_state.board.append(result)

# ------------------------------------------------------------------ results
st.markdown(f"<br><span class='px' style='color:{GOLD}'>SCORE</span>", unsafe_allow_html=True)
s1, s2 = st.columns(2)
mine, theirs = st.session_state.last.get((round_id, "you")), st.session_state.last.get((round_id, "agent"))
scorecard(s1, "YOU", mine, LIME)
scorecard(s2, name.upper(), theirs, PINK)
if mine:
    replay(s1, mine)
if theirs:
    replay(s2, theirs)
if mine and theirs:
    verdict = "You win this round." if mine.score > theirs.score else "The agent takes it." if theirs.score > mine.score else "Dead heat."
    st.markdown(f"<div class='px' style='text-align:center;font-size:1.3rem;color:{GOLD};margin:14px 0'>{verdict}</div>", unsafe_allow_html=True)

if st.session_state.board:
    st.markdown(f"<span class='px' style='color:{GOLD}'>SESSION BOARD</span>", unsafe_allow_html=True)
    board = pd.DataFrame([{"round": arena.TASKS[r.task].title, "contestant": r.contestant, "score": round(r.score, 2), "of": r.max_score,
                           "%": round(r.percent * 100), "steps": len(r.steps), "what went wrong": ", ".join(r.tags) or "-"} for r in st.session_state.board])
    st.dataframe(board.iloc[::-1], width="stretch", hide_index=True)
