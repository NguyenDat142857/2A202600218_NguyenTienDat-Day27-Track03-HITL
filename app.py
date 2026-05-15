from __future__ import annotations

import asyncio
import sqlite3
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""

if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None

if "final" not in st.session_state:
    st.session_state.final = None

if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False


# ─────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HITL PR Review Agent",
    layout="wide",
)

st.title("🤖 HITL PR Review Agent")
st.caption(
    "Human-in-the-loop GitHub PR review system using "
    "LangGraph + OpenRouter + SQLite audit trail"
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_recent_sessions():

    try:
        conn = sqlite3.connect(db_path())

        query = """
        SELECT
            thread_id,
            pr_url,
            MAX(timestamp) as last_event,
            MAX(confidence) as max_confidence,
            MAX(risk_level) as risk_level,
            MAX(action) as latest_action
        FROM audit_events
        GROUP BY thread_id
        ORDER BY last_event DESC
        LIMIT 20
        """

        df = pd.read_sql_query(query, conn)

        conn.close()

        return df

    except Exception:
        return pd.DataFrame()


def load_thread_events(thread_id: str):

    try:
        conn = sqlite3.connect(db_path())

        query = """
        SELECT
            timestamp,
            action,
            confidence,
            risk_level,
            decision,
            reviewer_id,
            reason,
            execution_time_ms
        FROM audit_events
        WHERE thread_id = ?
        ORDER BY timestamp ASC
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(thread_id,),
        )

        conn.close()

        return df

    except Exception:
        return pd.DataFrame()


def risk_badge(risk: str):

    if risk == "low":
        return "🟢 LOW"

    if risk == "med":
        return "🟡 MED"

    return "🔴 HIGH"


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.header("📚 Recent Sessions")

    sessions_df = load_recent_sessions()

    if sessions_df.empty:

        st.caption("No previous sessions")

    else:

        for _, row in sessions_df.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"""
**Thread:** `{row['thread_id'][:8]}...`

**Risk:** {risk_badge(row['risk_level'])}

**Action:** `{row['latest_action']}`

**Confidence:** `{row['max_confidence']:.0%}`

**PR:**  
{row['pr_url']}
"""
                )

                if st.button(
                    "Load Session",
                    key=row["thread_id"]
                ):
                    st.session_state.thread_id = row["thread_id"]
                    st.session_state.pr_url = row["pr_url"]
                    st.session_state.history_loaded = True
                    st.rerun()


# ─────────────────────────────────────────────────────────────
# Start Form
# ─────────────────────────────────────────────────────────────
with st.form("start"):

    pr_url = st.text_input(
        "GitHub PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )

    submitted = st.form_submit_button(
        "🚀 Run Review"
    )


# ─────────────────────────────────────────────────────────────
# Approval UI
# ─────────────────────────────────────────────────────────────
def render_approval_card(payload):

    conf = payload["confidence"]

    st.warning(
        f"Human approval required — confidence {conf:.0%}"
    )

    st.caption(payload["confidence_reasoning"])

    st.markdown("## Summary")

    st.markdown(payload["summary"])

    st.markdown("## Suggested Review Comments")

    for c in payload.get("comments", []):

        severity = c["severity"].upper()

        if severity == "HIGH":
            st.error(
                f"[{severity}] "
                f"{c['file']}:{c.get('line') or '?'}\n\n"
                f"{c['body']}"
            )

        elif severity == "MED":
            st.warning(
                f"[{severity}] "
                f"{c['file']}:{c.get('line') or '?'}\n\n"
                f"{c['body']}"
            )

        else:
            st.info(
                f"[{severity}] "
                f"{c['file']}:{c.get('line') or '?'}\n\n"
                f"{c['body']}"
            )

    with st.expander("📄 Diff Preview"):

        st.code(
            payload.get("diff_preview", ""),
            language="diff",
        )

    feedback = st.text_area(
        "Reviewer feedback",
        key="approval_feedback",
        height=120,
    )

    col1, col2, col3 = st.columns(3)

    if col1.button(
        "✅ Approve",
        type="primary",
        use_container_width=True,
    ):
        return {
            "choice": "approve",
            "feedback": feedback,
        }

    if col2.button(
        "❌ Reject",
        use_container_width=True,
    ):
        return {
            "choice": "reject",
            "feedback": feedback,
        }

    if col3.button(
        "✏️ Request Edit",
        use_container_width=True,
    ):
        return {
            "choice": "edit",
            "feedback": feedback,
        }

    return None


# ─────────────────────────────────────────────────────────────
# Escalation UI
# ─────────────────────────────────────────────────────────────
def render_escalation_card(payload):

    conf = payload["confidence"]

    st.error(
        f"Escalation required — confidence {conf:.0%}"
    )

    st.caption(payload["confidence_reasoning"])

    st.markdown("## Review Summary")

    st.markdown(payload["summary"])

    if payload.get("risk_factors"):

        st.markdown("## Risk Factors")

        for risk in payload["risk_factors"]:
            st.error(risk)

    st.markdown("## Reviewer Questions")

    with st.form("escalation_form"):

        answers = {}

        for idx, q in enumerate(payload["questions"]):

            answers[q] = st.text_area(
                f"Q{idx + 1}. {q}",
                height=120,
                key=f"question_{idx}",
            )

        submitted = st.form_submit_button(
            "Submit Answers"
        )

        if submitted:

            missing = [
                q for q, ans in answers.items()
                if not ans.strip()
            ]

            if missing:
                st.warning(
                    "Please answer all questions"
                )
                return None

            return answers

    return None


# ─────────────────────────────────────────────────────────────
# Graph Runner
# ─────────────────────────────────────────────────────────────
async def run_graph(
    pr_url: str,
    thread_id: str,
    resume_value=None,
):

    async with AsyncSqliteSaver.from_conn_string(
        db_path()
    ) as cp:

        await cp.setup()

        app = build_graph(cp)

        cfg = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        if resume_value is None:

            result = await app.ainvoke(
                {
                    "pr_url": pr_url,
                    "thread_id": thread_id,
                },
                cfg,
            )

        else:

            result = await app.ainvoke(
                Command(resume=resume_value),
                cfg,
            )

        return result


# ─────────────────────────────────────────────────────────────
# Run New Review
# ─────────────────────────────────────────────────────────────
if submitted and pr_url:

    st.session_state.pr_url = pr_url

    st.session_state.thread_id = str(uuid.uuid4())

    st.session_state.interrupt_payload = None

    st.session_state.final = None

    with st.spinner(
        "Fetching PR + running LLM review..."
    ):

        result = asyncio.run(
            run_graph(
                pr_url,
                st.session_state.thread_id,
            )
        )

    if "__interrupt__" in result:

        st.session_state.interrupt_payload = (
            result["__interrupt__"][0].value
        )

    else:

        st.session_state.final = result


# ─────────────────────────────────────────────────────────────
# Interrupt handling
# ─────────────────────────────────────────────────────────────
payload = st.session_state.interrupt_payload

if payload is not None:

    kind = payload["kind"]

    if kind == "approval_request":

        answer = render_approval_card(payload)

    else:

        answer = render_escalation_card(payload)

    if answer is not None:

        with st.spinner("Resuming graph..."):

            result = asyncio.run(
                run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                )
            )

        if "__interrupt__" in result:

            st.session_state.interrupt_payload = (
                result["__interrupt__"][0].value
            )

        else:

            st.session_state.interrupt_payload = None

            st.session_state.final = result

        st.rerun()


# ─────────────────────────────────────────────────────────────
# Final State
# ─────────────────────────────────────────────────────────────
if st.session_state.final is not None:

    st.divider()

    st.header("✅ Final Result")

    final = st.session_state.final

    action = final.get("final_action", "?")

    analysis = final.get("analysis")

    if action.startswith("auto"):

        st.success(
            f"✓ {action} — auto review posted"
        )

    elif action.startswith("committed"):

        st.success(
            f"✓ {action} — comment posted to GitHub PR"
        )

    elif action == "rejected":

        st.warning(
            "Review rejected — no comment posted"
        )

    else:

        st.info(
            f"final_action = {action}"
        )

    if analysis is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Confidence",
            f"{analysis.confidence:.0%}",
        )

        col2.metric(
            "Comments",
            len(analysis.comments),
        )

        col3.metric(
            "Risk Factors",
            len(analysis.risk_factors),
        )

        st.markdown("## Final Summary")

        st.markdown(analysis.summary)

    st.caption(
        f"""
thread_id = {st.session_state.thread_id}

Replay:
`python -m uv run python -m audit.replay --thread {st.session_state.thread_id}`
"""
    )


# ─────────────────────────────────────────────────────────────
# Audit Timeline
# ─────────────────────────────────────────────────────────────
if st.session_state.thread_id:

    st.divider()

    st.header("📜 Audit Timeline")

    events_df = load_thread_events(
        st.session_state.thread_id
    )

    if not events_df.empty:

        for _, row in events_df.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"""
### {row['action']}

- Confidence: `{row['confidence']:.0%}`
- Risk: `{row['risk_level']}`
- Decision: `{row['decision']}`
- Reviewer: `{row['reviewer_id'] or '-'}`
- Duration: `{row['execution_time_ms']} ms`

**Reason**
{row['reason']}
"""
                )

    else:

        st.caption("No audit events")