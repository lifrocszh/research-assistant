from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class AgentSpec(BaseModel):
    name: str
    description: str
    sources: list[str]
    skills: list[str]
    core_tools: list[str]
    keywords: list[str] = Field(default_factory=list)


class SkillSpec(BaseModel):
    name: str
    description: str
    keywords: list[str] = Field(default_factory=list)


class ToolSpec(BaseModel):
    name: str
    description: str
    source_types: list[str]
    keywords: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    claim: str
    source: str
    source_type: str
    evidence: str
    confidence: float = Field(ge=0, le=1)
    citation: str | None = None
    topic: str | None = None
    entities: list[str] = Field(default_factory=list)
    time_period: str | None = None


class Contradiction(BaseModel):
    claim: str
    finding_indexes: list[int]
    reason: str


class EvidenceStore(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    claims: dict[str, list[int]] = Field(default_factory=dict)
    contradictions: list[Contradiction] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)


class Action(BaseModel):
    agent: str
    task: str
    topic: str | None = None
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    follow_up: bool = False


class Decision(BaseModel):
    rationale: str
    actions: list[Action] = Field(default_factory=list)
    parallel: bool = False
    finish: bool = False

    @model_validator(mode="after")
    def valid_shape(self) -> Decision:
        if self.finish == bool(self.actions):
            raise ValueError("decision must either finish or contain actions")
        if self.parallel and len(self.actions) < 2:
            raise ValueError("parallel decisions need at least two actions")
        return self


class AgentResult(BaseModel):
    agent: str
    task: str
    topic: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    tool_calls: int = 0
    errors: list[str] = Field(default_factory=list)


class RunEvent(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    agent: str | None = None
    task: str | None = None
    status: str
    result_summary: str | None = None


class RuntimeLimits(BaseModel):
    max_parallel_agents: int = Field(default=3, ge=1)
    max_total_agents: int = Field(default=12, ge=1)
    max_research_depth: int = Field(default=5, ge=1)
    max_runtime_seconds: float = Field(default=180, gt=0)
    max_tool_calls_per_agent: int = Field(default=8, ge=1)
    max_tokens_per_run: int = Field(default=100_000, ge=1)


class ResearchState(BaseModel):
    thread_id: str
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    mode: Literal["fixture", "live"] = "fixture"
    documents: list[str] = Field(default_factory=list)
    fixture_path: str | None = None
    log_path: str | None = None
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    evidence: EvidenceStore = Field(default_factory=EvidenceStore)
    decision: Decision | None = None
    pending_results: list[AgentResult] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    used_agents: list[str] = Field(default_factory=list)
    agent_calls: dict[str, int] = Field(default_factory=dict)
    agent_tool_calls: dict[str, int] = Field(default_factory=dict)
    depth: int = 0
    total_agents: int = 0
    tool_calls: int = 0
    approximate_tokens: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pause_after_turn: bool = False
    status: Literal["running", "paused", "finished", "failed"] = "running"
    final_answer: str | None = None
