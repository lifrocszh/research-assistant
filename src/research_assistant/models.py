from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    description: str
    model: str
    system_prompt: str
    skills: list[str]
    tools: list[str]
    input_schema: str
    output_schema: str
    permissions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def valid_version(cls, value: str) -> str:
        if not re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", value):
            raise ValueError("version must be SemVer")
        return value

    @computed_field
    @property
    def name(self) -> str:
        return self.id

    @computed_field
    @property
    def core_tools(self) -> list[str]:
        return self.tools


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    description: str
    entrypoint: str = "SKILL.md"
    required_tools: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def valid_version(cls, value: str) -> str:
        return AgentSpec.valid_version(value)

    @computed_field
    @property
    def name(self) -> str:
        return self.id


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    description: str
    execution_type: Literal["python", "mcp"]
    input_schema: str
    output_schema: str
    permissions: list[str] = Field(default_factory=list)
    source_types: list[str]
    keywords: list[str] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def valid_version(cls, value: str) -> str:
        return AgentSpec.valid_version(value)

    @computed_field
    @property
    def name(self) -> str:
        return self.id


class ResearchTask(BaseModel):
    question: str = Field(min_length=1)
    documents: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    findings: list["Finding"] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SearchInput(BaseModel):
    query: str = Field(min_length=1)


class UrlInput(BaseModel):
    url: str = Field(min_length=1)


class DocumentInput(BaseModel):
    path: str = Field(min_length=1)


class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=1)


class SourceItem(BaseModel):
    title: str
    url: str
    content: str
    source_type: str
    claim: str | None = None
    confidence: float = Field(ge=0, le=1)


class SourceRecords(BaseModel):
    records: list[SourceItem] = Field(default_factory=list)


class AgentCapability(BaseModel):
    id: str
    version: str
    description: str
    skills: list[str]
    tools: list[str]
    input_schema: str
    output_schema: str


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
    catalog_fingerprint: str | None = None
    log_path: str | None = None
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    evidence: EvidenceStore = Field(default_factory=EvidenceStore)
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
