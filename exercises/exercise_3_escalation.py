"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)

console = Console()


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")

    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])

    console.print(
        f"  [green]✓[/green] "
        f"{len(pr.files_changed)} files, "
        f"head {pr.head_sha[:7]}"
    )

    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")

    llm = get_llm().with_structured_output(PRAnalysis)

    with console.status("[dim]LLM reviewing the diff...[/dim]"):

        analysis = llm.invoke([
            {
                "role": "system",
                "content": """
You are a senior software engineer reviewing pull requests.

Analyze the PR carefully.

Return:
- summary
- confidence
- confidence_reasoning
- review comments
- risk factors

IMPORTANT:

If confidence < 0.60:
- populate escalation_questions
- generate 2–4 very specific questions
- reference exact files/functions/security risks
- ask questions that help clarify intent/safety

Examples:
- Why is MD5 used in auth.py?
- Why is SQL query built using string interpolation?
- Is SYNC_URL guaranteed to be HTTPS in production?
                """,
            },
            {
                "role": "user",
                "content": f"""
Title:
{state["pr_title"]}

Changed Files:
{", ".join(state["pr_files"])}

Diff:
{state["pr_diff"]}
                """,
            },
        ])

    console.print(
        f"  [green]✓[/green] "
        f"confidence={analysis.confidence:.0%}, "
        f"{len(analysis.escalation_questions)} question(s)"
    )

    return {
        "analysis": analysis
    }


def node_route(state):

    console.print("[cyan]→ route[/cyan]")

    c = state["analysis"].confidence

    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"

    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"

    else:
        decision = "human_approval"

    console.print(
        f"  [green]✓[/green] "
        f"decision=[bold]{decision}[/bold] "
        f"(confidence={c:.0%})"
    )

    return {
        "decision": decision
    }


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions."""

    console.print("[yellow]→ escalate[/yellow]")

    a = state["analysis"]

    questions = a.escalation_questions

    if not questions:
        questions = [
            "What is the intent of this PR?",
            "Any migration or security concerns?",
        ]

    payload = {
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": getattr(a, "risk_factors", []),
        "questions": questions,
    }

    answers = interrupt(payload)

    console.print(
        f"  [green]✓[/green] "
        f"received {len(answers)} escalation answers"
    )

    return {
        "escalation_answers": answers
    }


def node_synthesize(state: ReviewState) -> dict:
    """
    Re-prompt the LLM with reviewer answers
    and generate a refined review.
    """

    console.print("[cyan]→ synthesize[/cyan]")

    original = state["analysis"]
    answers = state["escalation_answers"]

    qa_text = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in answers.items()]
    )

    llm = get_llm().with_structured_output(PRAnalysis)

    prompt = f"""
You are performing a second-pass PR review.

Original PR summary:
{original.summary}

Original confidence:
{original.confidence}

Original reasoning:
{original.confidence_reasoning}

Original comments:
{original.comments}

Reviewer answers:
{qa_text}

Now generate a refined review.

You may:
- increase confidence if answers reduce risk
- reduce confidence if answers introduce more risk
- refine comments
- refine reasoning

PR Diff:
{state["pr_diff"]}
    """

    with console.status("[dim]Synthesizing refined review...[/dim]"):
        refined = llm.invoke(prompt)

    console.print(
        f"  [green]✓[/green] "
        f"refined confidence={refined.confidence:.0%}"
    )

    return {
        "analysis": refined
    }


def node_human_approval(state):

    console.print("[cyan]→ human_approval[/cyan]")

    a = state["analysis"]

    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    return {
        "human_choice": response.get("choice"),
        "human_feedback": response.get("feedback"),
    }


def _render_comment_body(state) -> str:

    a = state["analysis"]

    lines = [
        f"### Automated review (confidence {a.confidence:.0%})",
        "",
        a.summary,
        "",
    ]

    for c in a.comments:

        lines.append(
            f"- **[{c.severity}]** "
            f"`{c.file}:{c.line or '?'}` "
            f"— {c.body}"
        )

    if state.get("human_feedback"):

        lines.append(
            f"\n_Reviewer note: "
            f"{state['human_feedback']}_"
        )

    if state.get("escalation_answers"):

        lines.append(
            "\n_Reviewer answered escalation questions:_"
        )

        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")

    return "\n".join(lines)


def _post(state, label: str) -> str:

    try:

        post_review_comment(
            state["pr_url"],
            _render_comment_body(state),
        )

        console.print(
            f"  [green]✓[/green] "
            f"posted comment to {state['pr_url']}"
        )

        return label

    except Exception as e:

        console.print(f"  [red]✗[/red] post failed: {e}")

        return "commit_failed"


def node_commit(state):

    console.print("[cyan]→ commit[/cyan]")

    # Escalation path
    if state.get("escalation_answers"):

        return {
            "final_action": _post(
                state,
                "committed_after_escalation",
            )
        }

    # Human approval path
    if state.get("human_choice") == "approve":

        return {
            "final_action": _post(
                state,
                "committed",
            )
        }

    console.print(
        f"  [yellow]·[/yellow] "
        f"skipping comment "
        f"(choice={state.get('human_choice')})"
    )

    return {
        "final_action": "rejected"
    }


def node_auto_approve(state):

    console.print(
        "[cyan]→ auto_approve[/cyan]  "
        "[dim]high confidence — posting directly[/dim]"
    )

    return {
        "final_action": _post(
            state,
            "auto_approved",
        )
    }


def build_graph():

    g = StateGraph(ReviewState)

    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)

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

    # Escalation path
    g.add_edge("escalate", "synthesize")

    g.add_edge("synthesize", "commit")

    return g.compile(
        checkpointer=MemorySaver()
    )


def handle_interrupt(payload):

    kind = payload["kind"]

    # Normal approval
    if kind == "approval_request":

        console.print(
            Panel.fit(
                payload["summary"],
                title=f"Approve? conf={payload['confidence']:.0%}",
                border_style="green",
            )
        )

        choice = console.input(
            "approve/reject/edit? "
        ).strip().lower()

        feedback = console.input(
            "Feedback: "
        ).strip()

        return {
            "choice": choice,
            "feedback": feedback,
        }

    # Escalation flow
    if kind == "escalation":

        console.print(
            Panel.fit(
                payload["summary"],
                title=f"Escalation conf={payload['confidence']:.0%}",
                border_style="yellow",
            )
        )

        if payload.get("risk_factors"):

            console.print("\n[bold red]Risk factors:[/bold red]")

            for r in payload["risk_factors"]:
                console.print(f" • {r}")

        answers = {}

        for q in payload["questions"]:

            ans = console.input(
                f"\n[bold]Q:[/bold] {q}\n[bold]A:[/bold] "
            ).strip()

            answers[q] = ans

        return answers

    raise ValueError(kind)


def main():

    load_dotenv()

    p = argparse.ArgumentParser()

    p.add_argument("--pr", required=True)

    args = p.parse_args()

    console.rule(
        "[bold]Exercise 3 — escalation with reviewer Q&A[/bold]"
    )

    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()

    thread_id = str(uuid.uuid4())

    cfg = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    console.print(
        f"[dim]thread_id = {thread_id}[/dim]\n"
    )

    result = app.invoke(
        {
            "pr_url": args.pr,
            "thread_id": thread_id,
        },
        cfg,
    )

    while "__interrupt__" in result:

        payload = result["__interrupt__"][0].value

        answer = handle_interrupt(payload)

        result = app.invoke(
            Command(resume=answer),
            cfg,
        )

    console.rule("Final")

    console.print(
        f"final_action = {result.get('final_action')}"
    )

    if "analysis" in result:

        console.print(
            f"final confidence = "
            f"{result['analysis'].confidence:.0%}"
        )


if __name__ == "__main__":
    main()