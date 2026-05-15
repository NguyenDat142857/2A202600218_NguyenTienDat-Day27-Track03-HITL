from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)

console = Console()
AGENT_ID = "pr-review-agent@v0.1"


# ─────────────────────────────────────────
# SAFE PARSER (ROBUST)
# ─────────────────────────────────────────
def parse_analysis(text: str) -> PRAnalysis:
    try:
        text = text.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text)
        return PRAnalysis.model_validate(data)

    except Exception:
        return PRAnalysis(
            summary=text[:500],
            confidence=0.5,
            confidence_reasoning="fallback parse failed",
            comments=[],
            risk_factors=["invalid_json"],
            escalation_questions=["Manual review required"],
        )


async def audit(state, entry: AuditEntry):
    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
    )


# ─────────────────────────────────────────
# FETCH PR
# ─────────────────────────────────────────
async def node_fetch_pr(state):
    t0 = time.monotonic()
    pr = fetch_pr(state["pr_url"])

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="low",
        decision="pending",
        reviewer_id=None,
        reason=f"{len(pr.files_changed)} files",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


# ─────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────
async def node_analyze(state):
    t0 = time.monotonic()
    llm = get_llm()

    resp = await llm.ainvoke([
        {
            "role": "system",
            "content": (
                "Return ONLY valid JSON.\n"
                "No markdown. No explanation."
            )
        },
        {
            "role": "user",
            "content": f"""
TITLE:
{state["pr_title"]}

DIFF:
{state["pr_diff"]}
"""
        }
    ])

    analysis = parse_analysis(resp.content)

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=analysis.confidence,
        risk_level=risk_level_for(analysis.confidence),
        decision="pending",
        reviewer_id=None,
        reason=analysis.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))

    return {"analysis": analysis}


# ─────────────────────────────────────────
# ROUTE
# ─────────────────────────────────────────
async def node_route(state):
    c = state["analysis"].confidence

    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"

    return {"decision": decision}


# ─────────────────────────────────────────
# HUMAN APPROVAL (FIXED PAYLOAD)
# ─────────────────────────────────────────
async def node_human_approval(state):
    a = state["analysis"]

    resp = interrupt({
        "kind": "approval_request",
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    return {
        "human_choice": resp.get("choice"),
        "human_feedback": resp.get("feedback"),
    }


# ─────────────────────────────────────────
# ESCALATE (FIXED PAYLOAD)
# ─────────────────────────────────────────
async def node_escalate(state):
    a = state["analysis"]

    questions = a.escalation_questions or ["What is the intent?"]

    answers = interrupt({
        "kind": "escalation",
        "confidence": a.confidence,
        "summary": a.summary,
        "questions": questions,
    })

    return {"escalation_answers": answers}


# ─────────────────────────────────────────
# SYNTHESIZE
# ─────────────────────────────────────────
async def node_synthesize(state):
    llm = get_llm()

    qa = "\n".join(
        f"Q:{k}\nA:{v}"
        for k, v in (state.get("escalation_answers") or {}).items()
    )

    resp = await llm.ainvoke([
        {"role": "system", "content": "Return ONLY JSON"},
        {"role": "user", "content": f"DIFF:\n{state['pr_diff']}\n\nQA:\n{qa}"}
    ])

    refined = parse_analysis(resp.content)

    return {"analysis": refined}


# ─────────────────────────────────────────
# POST COMMENT (FIXED SAFE)
# ─────────────────────────────────────────
def _post(state):
    try:
        post_review_comment(
            state["pr_url"],
            state["analysis"].summary,
        )
        return "committed"
    except Exception as e:
        console.print(f"[red]post failed: {e}[/red]")
        return "failed"


# ─────────────────────────────────────────
# COMMIT (FIXED FLOW)
# ─────────────────────────────────────────
async def node_commit(state):
    choice = state.get("human_choice")

    if choice == "approve":
        action = _post(state)
    else:
        action = "rejected"

    return {"final_action": action}


# ─────────────────────────────────────────
# AUTO APPROVE
# ─────────────────────────────────────────
async def node_auto_approve(state):
    action = _post(state)
    return {"final_action": f"auto_{action}"}


# ─────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────
def build_graph(checkpointer):
    g = StateGraph(ReviewState)

    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)
    g.add_node("synthesize", node_synthesize)
    g.add_node("commit", node_commit)
    g.add_node("auto_approve", node_auto_approve)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")

    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )

    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")

    return g.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)

        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke(
            {"pr_url": pr_url, "thread_id": thread_id},
            cfg,
        )

        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=payload), cfg)

        console.print("FINAL:", result.get("final_action"))


def main():
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread")

    args = p.parse_args()

    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()