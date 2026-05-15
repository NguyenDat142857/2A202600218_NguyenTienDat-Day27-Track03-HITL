from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


Decision = Literal["auto_approve", "human_approval", "escalate"]
HumanChoice = Literal["approve", "reject", "edit"]

AUTO_APPROVE_THRESHOLD = 0.73
ESCALATE_THRESHOLD = 0.58


def risk_level_for(confidence: float) -> str:
    if confidence >= AUTO_APPROVE_THRESHOLD:
        return "low"
    if confidence < ESCALATE_THRESHOLD:
        return "high"
    return "med"


class ReviewComment(BaseModel):
    file: str
    line: int | None = None
    severity: Literal["nit", "suggestion", "issue", "blocker"]
    body: str


class PRAnalysis(BaseModel):
    summary: str
    risk_factors: list[str] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
    confidence_reasoning: str

    escalation_questions: list[str] = Field(default_factory=list)


class AuditEntry(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    agent_id: str
    action: str
    confidence: float
    risk_level: str

    reviewer_id: str | None = None
    decision: str

    reason: str | None = None
    execution_time_ms: int


class ReviewState(TypedDict, total=False):
    pr_url: str
    thread_id: str

    pr_title: str
    pr_diff: str
    pr_files: list[str]
    pr_head_sha: str

    analysis: PRAnalysis
    decision: Decision

    human_choice: HumanChoice | None
    human_feedback: str | None

    escalation_answers: dict[str, str] | None
    final_action: str | None