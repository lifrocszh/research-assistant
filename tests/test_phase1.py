from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from research_assistant.engine import DynamicOrchestrator, ResearchRuntime
from research_assistant.models import Action, Decision, ResearchState, RuntimeLimits
from research_assistant.registry import CapabilityRegistry
from research_assistant.tools import FixtureSearchAdapter, ResearchTools, ToolError


def test_registry_discovery_and_decision_validation() -> None:
    registry = CapabilityRegistry()
    assert registry.discover_agents("SEC company margin")[0].name == "company_researcher"
    assert registry.discover_skills("academic paper")[0].name == "academic_research"
    assert registry.discover_tools("search document")[0].name == "search_document"
    with pytest.raises(ValidationError):
        Decision(rationale="invalid", finish=True, actions=[Action(agent="web_researcher", task="x")])


def test_dynamic_paths_and_dispatch_modes(tmp_path) -> None:
    web_events, research_events = [], []
    with ResearchRuntime(tmp_path / "web.sqlite", web_events.append) as runtime:
        web = runtime.run("What does LangGraph provide?", thread_id="web")
    with ResearchRuntime(tmp_path / "research.sqlite", research_events.append) as runtime:
        research = runtime.run("Research dynamic delegation using multiple sources", thread_id="research")
    assert web.used_agents == ["web_researcher"]
    assert len(research.used_agents) > 1
    assert not any(event.event_type == "parallel_started" for event in web_events)
    assert any(event.event_type == "parallel_started" for event in research_events)


def test_partial_parallel_failure_keeps_successes(tmp_path) -> None:
    with ResearchRuntime(tmp_path / "partial.sqlite") as runtime:
        state = runtime.run(
            "Research dynamic delegation with local document and multiple sources",
            thread_id="partial",
            documents=[str(tmp_path / "missing.md")],
        )
    assert state.status == "finished"
    assert state.evidence.findings
    assert any(event.status == "failed" for event in state.events if event.event_type == "tool_finished")


def test_bounds_and_tool_call_limit(tmp_path) -> None:
    limits = RuntimeLimits(max_parallel_agents=1, max_total_agents=1, max_research_depth=1, max_runtime_seconds=10, max_tool_calls_per_agent=1, max_tokens_per_run=10_000)
    with ResearchRuntime(tmp_path / "bounds.sqlite") as runtime:
        state = runtime.run("Acme company margin filing", thread_id="bounds", limits=limits)
    assert state.total_agents == 1
    assert state.depth == 1
    assert state.tool_calls == 1

    token_limits = RuntimeLimits(max_total_agents=1, max_tokens_per_run=1)
    with ResearchRuntime(tmp_path / "tokens.sqlite") as runtime:
        token_state = runtime.run("What does LangGraph provide?", thread_id="tokens", limits=token_limits)
    assert token_state.approximate_tokens <= 1


def test_conflict_and_gap_detection(tmp_path) -> None:
    with ResearchRuntime(tmp_path / "critique.sqlite") as runtime:
        conflict = runtime.run("Research Acme 2025 margin decline with multiple sources", thread_id="conflict")
        gap = runtime.run("Question with no matching fixture zebrawombat", thread_id="gap")
    assert conflict.evidence.contradictions
    assert any(event.event_type == "conflict_detected" for event in conflict.events)
    assert gap.evidence.unanswered_questions == [gap.question]
    assert "No supported answer" in (gap.final_answer or "")


def test_sqlite_pause_and_resume(tmp_path) -> None:
    checkpoint = tmp_path / "resume.sqlite"
    with ResearchRuntime(checkpoint) as runtime:
        paused = runtime.run("What does LangGraph provide?", thread_id="resume-me", pause_after_turn=True)
    assert paused.status == "paused"
    assert paused.depth == 1
    with ResearchRuntime(checkpoint) as runtime:
        finished = runtime.resume("resume-me")
    assert finished.status == "finished"
    assert finished.final_answer
    assert finished.run_id == paused.run_id


def test_fixture_determinism_and_custom_file(tmp_path) -> None:
    query = "What does LangGraph provide?"
    assert FixtureSearchAdapter().search(query) == FixtureSearchAdapter().search(query)
    fixture = tmp_path / "fixtures.json"
    fixture.write_text(json.dumps([{"title": "One", "url": "https://example.test/1", "content": "Orchid fact is stable.", "source_type": "web"}]), encoding="utf-8")
    assert FixtureSearchAdapter(str(fixture)).search("orchid")[0].title == "One"


def test_http_errors_and_missing_live_credentials(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tools = ResearchTools("live", timeout=0.1)
    try:
        with pytest.raises(ToolError, match="TAVILY_API_KEY"):
            tools.web_search("test")
        tools.client.close()
        tools.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503, request=request)))
        with pytest.raises(ToolError, match="URL fetch failed"):
            tools.fetch_url("https://example.test")
    finally:
        tools.close()


def test_live_runtime_fails_clearly_without_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with ResearchRuntime(tmp_path / "live.sqlite") as runtime:
        with pytest.raises(ValueError, match="live mode requires TAVILY_API_KEY"):
            runtime.run("current news", thread_id="live", mode="live")


def test_jsonl_events_and_markdown_citations(tmp_path) -> None:
    lines = []
    with ResearchRuntime(tmp_path / "events.sqlite", lambda event: lines.append(event.model_dump_json())) as runtime:
        state = runtime.run("What does LangGraph provide?", thread_id="events")
    events = [json.loads(line) for line in lines]
    assert events[0]["event_type"] == "run_started"
    assert events[-1]["event_type"] == "run_finished"
    assert all(item["run_id"] == state.run_id for item in events)
    assert "[1]" in (state.final_answer or "")


def test_document_search(tmp_path) -> None:
    document = tmp_path / "notes.md"
    document.write_text("# Notes\n\nOrchid latency fell by 20 percent after caching.\n", encoding="utf-8")
    with ResearchRuntime(tmp_path / "document.sqlite") as runtime:
        state = runtime.run("What do the document notes say about orchid latency?", thread_id="document", documents=[str(document)])
    assert any(finding.source_type == "document" for finding in state.evidence.findings)
    assert "Orchid latency" in (state.final_answer or "")


def test_orchestrator_stops_on_expired_runtime() -> None:
    state = ResearchState(thread_id="expired", question="x", limits=RuntimeLimits(max_runtime_seconds=0.000001))
    state.started_at = datetime.now(UTC) - timedelta(seconds=1)
    decision = DynamicOrchestrator(CapabilityRegistry()).decide(state)
    assert decision.finish
