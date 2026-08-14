from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeVar

from .models import AgentSpec, SkillSpec, ToolSpec

Spec = TypeVar("Spec", AgentSpec, SkillSpec, ToolSpec)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class CapabilityRegistry:
    def __init__(self) -> None:
        self.agents = {item.name: item for item in _agents()}
        self.skills = {item.name: item for item in _skills()}
        self.tools = {item.name: item for item in _tools()}

    def discover_agents(self, query: str) -> list[AgentSpec]:
        return self._discover(query, self.agents.values())

    def discover_skills(self, query: str) -> list[SkillSpec]:
        return self._discover(query, self.skills.values())

    def discover_tools(self, query: str) -> list[ToolSpec]:
        return self._discover(query, self.tools.values())

    @staticmethod
    def _discover(query: str, specs: Iterable[Spec]) -> list[Spec]:
        wanted = _terms(query)
        ranked = []
        for spec in specs:
            text = " ".join([spec.name, spec.description, *spec.keywords])
            score = len(wanted & _terms(text))
            if score:
                ranked.append((score, spec.name, spec))
        return [spec for _, _, spec in sorted(ranked, reverse=True)]


def _agents() -> list[AgentSpec]:
    return [
        AgentSpec(name="web_researcher", description="Research current public web sources", sources=["web"], skills=["web_research"], core_tools=["web_search", "fetch_url"], keywords=["current", "news", "web", "compare", "source", "general"]),
        AgentSpec(name="academic_researcher", description="Research papers and scholarly evidence", sources=["arXiv", "academic"], skills=["academic_research"], core_tools=["fetch_arxiv"], keywords=["paper", "study", "research", "academic", "method", "model", "benchmark"]),
        AgentSpec(name="company_researcher", description="Research companies using primary financial and regulatory sources", sources=["SEC", "company filings"], skills=["company_research"], core_tools=["fetch_sec", "web_search"], keywords=["company", "financial", "revenue", "margin", "filing", "stock", "sec", "10-k", "10-q"]),
        AgentSpec(name="document_researcher", description="Search user-provided Markdown and text documents", sources=["local documents"], skills=["document_research"], core_tools=["read_document", "search_document"], keywords=["document", "file", "report", "notes", "local"]),
        AgentSpec(name="synthesis_critic", description="Detect gaps and conflicts, then synthesize cited answers", sources=["evidence store"], skills=["evidence_critique", "synthesis"], core_tools=[], keywords=["synthesize", "conflict", "gap", "answer", "evidence"]),
    ]


def _skills() -> list[SkillSpec]:
    return [
        SkillSpec(name="web_research", description="Search and verify public web evidence", keywords=["web", "current", "news"]),
        SkillSpec(name="academic_research", description="Find scholarly sources", keywords=["academic", "paper", "study"]),
        SkillSpec(name="company_research", description="Prioritize filings and financial sources", keywords=["company", "financial", "sec"]),
        SkillSpec(name="document_research", description="Find evidence in local text", keywords=["document", "file", "local"]),
        SkillSpec(name="evidence_critique", description="Detect evidence gaps and conflicts", keywords=["gap", "conflict", "evidence"]),
        SkillSpec(name="synthesis", description="Write an evidence-grounded cited answer", keywords=["answer", "report", "synthesize"]),
    ]


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(name="web_search", description="Search Tavily or deterministic fixtures", source_types=["web"], keywords=["search", "web"]),
        ToolSpec(name="fetch_url", description="Fetch a bounded HTTP page", source_types=["web"], keywords=["url", "web", "page"]),
        ToolSpec(name="fetch_arxiv", description="Search public arXiv metadata", source_types=["academic"], keywords=["paper", "arxiv", "academic"]),
        ToolSpec(name="fetch_sec", description="Search public SEC submissions", source_types=["regulatory"], keywords=["sec", "filing", "company"]),
        ToolSpec(name="read_document", description="Read bounded Markdown or text documents", source_types=["document"], keywords=["read", "document", "file"]),
        ToolSpec(name="search_document", description="Search loaded documents", source_types=["document"], keywords=["search", "document", "file"]),
    ]
