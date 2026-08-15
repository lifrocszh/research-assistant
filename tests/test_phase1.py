from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from research_assistant.engine import DynamicOrchestrator, ResearchRuntime
from research_assistant.models import Action, AgentResult, Decision, Finding, ResearchState, RuntimeLimits
from research_assistant.registry import CapabilityRegistry
from research_assistant.tools import FixtureSearchAdapter, ResearchTools, SourceRecord, TavilySearchAdapter, ToolError


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
    assert "company_researcher" not in research.used_agents
    assert {finding.source_type for finding in research.evidence.findings} == {"academic", "web"}
    assert not research.evidence.contradictions
    assert not any(event.event_type == "parallel_started" for event in web_events)
    assert any(event.event_type == "parallel_started" for event in research_events)


def test_multi_topic_research_is_planned_and_sectioned(tmp_path) -> None:
    fixture = tmp_path / "tiktok.json"
    fixture.write_text(json.dumps([
        {
            "title": "TikTok recommendation architecture",
            "url": "https://example.test/tiktok",
            "content": "TikTok recommendation architecture uses candidate generation and ranking stages.",
            "source_type": "web",
            "claim": "TikTok uses candidate generation and ranking stages.",
        },
        {
            "title": "YouTube recommendation engineering",
            "url": "https://example.test/youtube",
            "content": "YouTube recommendation systems use candidate generation and ranking.",
            "source_type": "web",
            "claim": "YouTube uses candidate generation and ranking.",
        },
    ]), encoding="utf-8")
    question = "Find out more about TikTok recommendation system stack how implemented, competitors in other companies and a summary"
    state = ResearchState(thread_id="topics", question=question, fixture_path=str(fixture), limits=RuntimeLimits(max_parallel_agents=2, max_total_agents=2, max_research_depth=1))
    decision = DynamicOrchestrator(CapabilityRegistry()).decide(state)
    assert decision.parallel
    assert {action.topic for action in decision.actions} == {"architecture", "competitors"}
    assert all(action.agent == "web_researcher" for action in decision.actions)

    with ResearchRuntime(tmp_path / "topics.sqlite") as runtime:
        result = runtime.run(question, thread_id="topics", fixture_path=str(fixture), limits=RuntimeLimits(max_parallel_agents=2, max_total_agents=2, max_research_depth=1))
    assert "## Architecture" in (result.final_answer or "")
    assert "## Competitors" in (result.final_answer or "")
    assert "TikTok uses candidate generation" in (result.final_answer or "")
    assert "YouTube uses candidate generation" in (result.final_answer or "")


def test_noisy_search_claim_is_dropped() -> None:
    noisy = SourceRecord(
        "Search result",
        "https://example.test/noise",
        "Consulting Consulting Integration Integration Odoo Odoo Manufacturing Manufacturing.",
        "web",
    )
    assert ResearchRuntime._findings([noisy]) == []


def test_full_content_extraction_skips_boilerplate_and_keeps_distinct_claims() -> None:
    record = SourceRecord(
        "DeepSeek Harness developer preview",
        "https://example.test/deepseek-harness",
        "Written by DeepSeek Research. Advertisement: Start your free trial today. DeepSeek Harness uses plugins to connect tools and models to agent sessions. Its sandboxes provide agents with isolated filesystems and a UI.",
        "web",
    )
    findings = ResearchRuntime._findings([record], topic="components", question="What does the DeepSeek Harness do?")
    assert [finding.claim for finding in findings] == [
        "DeepSeek Harness uses plugins to connect tools and models to agent sessions.",
        "Its sandboxes provide agents with isolated filesystems and a UI.",
    ]
    assert all(finding.citation == record.url and finding.source_type == "web" for finding in findings)


def test_promotional_and_social_card_claims_are_dropped() -> None:
    record = SourceRecord(
        "DeepSeek Harness developer preview",
        "https://example.test/deepseek-harness",
        "Remy is the world's most powerful product manager agent. Try Remy today for free. Models [...] Follow @DeepSeek for updates. DeepSeek Harness uses plugins to connect tools and models to agent sessions.",
        "web",
    )
    findings = ResearchRuntime._findings([record], topic="components", question="What does the DeepSeek Harness do?")
    assert [finding.claim for finding in findings] == ["DeepSeek Harness uses plugins to connect tools and models to agent sessions."]


def test_markdown_deduplicates_claim_bullets_and_source_rows() -> None:
    state = ResearchState(thread_id="citations", question="Describe DeepSeek Harness")
    state.evidence.findings = [
        Finding(claim="DeepSeek Harness uses plugins for tools.", source="One", source_type="web", evidence="x", confidence=0.8, citation="https://example.test/one"),
        Finding(claim="DeepSeek Harness uses plugins for tools.", source="Two", source_type="web", evidence="x", confidence=0.8, citation="https://example.test/two"),
        Finding(claim="DeepSeek Harness has sandboxed sessions.", source="One", source_type="web", evidence="x", confidence=0.8, citation="https://example.test/one"),
    ]
    answer = ResearchRuntime._markdown(state)
    assert answer.count("DeepSeek Harness uses plugins for tools.") == 1
    assert "- DeepSeek Harness uses plugins for tools. [1, 2]" in answer
    assert answer.count("https://example.test/one") == 1
    assert answer.count("https://example.test/two") == 1


def test_broad_question_groups_coverage_and_reports_gaps(tmp_path) -> None:
    fixture = tmp_path / "deepseek-harness.json"
    fixture.write_text(json.dumps([
        {
            "title": "DeepSeek Harness developer preview",
            "url": "https://example.test/deepseek-harness",
            "content": "DeepSeek Harness helps developers build and run coding agents. Its plugin architecture connects tools and models to agent sessions. The harness orchestrates sandboxed loops for each agent. Teams can use it to automate repository tasks. It is available as a developer preview. The preview does not support custom production deployments.",
            "source_type": "web",
        },
        {
            "title": "DeepSeek Harness sponsor",
            "url": "https://example.test/sponsor",
            "content": "Advertisement: Subscribe for a free cookbook today. This recipe uses a harness for climbing gear.",
            "source_type": "web",
        },
    ]), encoding="utf-8")
    with ResearchRuntime(tmp_path / "deepseek.sqlite") as runtime:
        state = runtime.run(
            "What does the DeepSeek Harness do?",
            thread_id="deepseek",
            fixture_path=str(fixture),
            limits=RuntimeLimits(max_parallel_agents=3, max_total_agents=6, max_research_depth=4),
        )
    answer = state.final_answer or ""
    for area in ("Purpose", "Components", "Operation", "Use Cases", "Status", "Limitations"):
        assert f"## {area}" in answer
    assert "plugin architecture connects tools and models" in answer
    assert "sandboxed loops" in answer
    assert "developer preview" in answer
    assert "cookbook" not in answer
    assert all(finding.citation == "https://example.test/deepseek-harness" for finding in state.evidence.findings)

    with ResearchRuntime(tmp_path / "gaps.sqlite") as runtime:
        gaps = runtime.run(
            "What does the DeepSeek Harness do?",
            thread_id="gaps",
            fixture_path=str(fixture),
            limits=RuntimeLimits(max_parallel_agents=3, max_total_agents=3, max_research_depth=1),
        )
    assert "## Unanswered questions" in (gaps.final_answer or "")
    assert "Need evidence for: status" in (gaps.final_answer or "")


def test_live_search_query_drops_instruction_prose(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        TavilySearchAdapter(client).search(
            "TikTok recommendation system architecture implementation\n"
            "Focus only on publicly documented architecture."
        )
    finally:
        client.close()
    assert seen["query"] == "TikTok recommendation system architecture implementation"


def test_incomplete_architecture_claim_is_rejected() -> None:
    record = SourceRecord(
        "TikTok recommendation architecture",
        "https://example.test/tiktok-architecture",
        "The source discusses TikTok recommendation architecture.",
        "web",
        "Efficiency, consistency, and scalability.",
    )
    findings = ResearchRuntime._findings(
        [record],
        topic="architecture",
        question="How is the TikTok recommendation system implemented?",
    )
    assert findings == []


def test_competitor_claim_requires_named_comparison() -> None:
    record = SourceRecord(
        "TikTok recommendation system",
        "https://example.test/tiktok-competitors",
        "A generic overview of TikTok recommendation systems.",
        "web",
        "Competitors use candidate generation and ranking.",
    )
    findings = ResearchRuntime._findings(
        [record],
        topic="competitors",
        question="Compare TikTok recommendation systems with competitors.",
    )
    assert findings == []


def test_unrelated_academic_followup_is_not_accepted(tmp_path) -> None:
    fixture = tmp_path / "followup.json"
    fixture.write_text(json.dumps([
        {
            "title": "TikTok recommendation architecture",
            "url": "https://example.test/tiktok-architecture",
            "content": "TikTok recommendation architecture uses candidate generation and ranking stages.",
            "source_type": "web",
            "claim": "TikTok uses candidate generation and ranking stages.",
        },
        {
            "title": "CMS detector paper",
            "url": "https://example.test/cms",
            "content": "TikTok recommendation architecture query matched an unrelated CMS detector paper about particle physics.",
            "source_type": "academic",
            "claim": "The CMS detector measures particle collisions.",
        },
    ]), encoding="utf-8")

    with ResearchRuntime(tmp_path / "followup.sqlite") as runtime:
        state = runtime.run(
            "How is the TikTok recommendation system implemented?",
            thread_id="followup",
            fixture_path=str(fixture),
            limits=RuntimeLimits(max_parallel_agents=1, max_total_agents=2, max_research_depth=2),
        )

    assert not any(finding.source_type == "academic" for finding in state.evidence.findings)


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


def test_parallel_duplicate_agent_respects_shared_tool_budget(tmp_path, monkeypatch) -> None:
    class OneTool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def invoke(self, name: str, query: str, documents: list[str]) -> list[SourceRecord]:
            return [SourceRecord("Source", "https://example.test/source", "A supported claim.", "web", "A supported claim.")]

        def close(self) -> None:
            pass

    monkeypatch.setattr("research_assistant.engine.ResearchTools", OneTool)
    state = ResearchState(
        thread_id="shared-tool-budget",
        question="Research multiple sources",
        decision=Decision(
            rationale="parallel topics",
            actions=[
                Action(agent="web_researcher", task="topic one", tools=["web_search"]),
                Action(agent="web_researcher", task="topic two", tools=["web_search"]),
            ],
            parallel=True,
        ),
        limits=RuntimeLimits(max_parallel_agents=2, max_total_agents=2, max_tool_calls_per_agent=1),
    )
    with ResearchRuntime(tmp_path / "shared-tool-budget.sqlite") as runtime:
        dispatched = runtime._dispatch({"data": state.model_dump(mode="json")})
        collected = runtime._collect(dispatched)
    result = ResearchState.model_validate(collected["data"])
    assert result.tool_calls == 1
    assert result.agent_tool_calls == {"web_researcher": 1}


def test_live_finding_ingestion_does_not_charge_finding_text(tmp_path) -> None:
    finding = Finding(
        claim="A supported claim.",
        source="Source",
        source_type="web",
        evidence="A supported claim.",
        confidence=0.9,
        citation="https://example.test/source",
    )
    state = ResearchState(
        thread_id="live-collect",
        question="Research",
        mode="live",
        approximate_tokens=123,
        pending_results=[AgentResult(agent="web_researcher", task="Research", findings=[finding], tool_calls=1)],
    )
    with ResearchRuntime(tmp_path / "live-collect.sqlite") as runtime:
        collected = runtime._collect({"data": state.model_dump(mode="json")})
    result = ResearchState.model_validate(collected["data"])
    assert result.approximate_tokens == 123
    assert len(result.evidence.findings) == 1


def test_findings_cap_sentences_per_record() -> None:
    record = SourceRecord(
        "Source",
        "https://example.test/source",
        "LangGraph supports durable checkpointed execution for research workflows. Enterprise deployments isolate tools using explicit capability registries and policies. Researchers compare retrieval quality using source citations and confidence scores. Runtime state persists across interruptions through SQLite checkpoints.",
        "web",
    )
    assert len(ResearchRuntime._findings([record], question="Research")) == 3


def test_conflict_and_gap_detection(tmp_path) -> None:
    with ResearchRuntime(tmp_path / "critique.sqlite") as runtime:
        conflict = runtime.run("Research Acme 2025 margin decline with multiple sources", thread_id="conflict")
        gap = runtime.run("Question with no matching fixture zebrawombat", thread_id="gap")
    assert conflict.evidence.contradictions
    assert any(event.event_type == "conflict_detected" for event in conflict.events)
    assert gap.evidence.unanswered_questions == [gap.question]
    assert "No supported answer" in (gap.final_answer or "")


def test_filing_metadata_is_not_a_conflict(tmp_path) -> None:
    fixture = tmp_path / "filings.json"
    fixture.write_text(json.dumps([
        {"title": "LAM 10-K", "url": "https://example.test/10-k", "content": "LAM filed 10-K on 2026-08-07 for period 2026-06-28.", "source_type": "regulatory"},
        {"title": "LAM 8-K", "url": "https://example.test/8-k", "content": "LAM filed 8-K on 2026-07-29 for period 2026-07-29.", "source_type": "regulatory"},
    ]), encoding="utf-8")
    with ResearchRuntime(tmp_path / "filings.sqlite") as runtime:
        state = runtime.run("Research SEC filings with multiple sources", thread_id="filings", fixture_path=str(fixture))
    assert not state.evidence.contradictions


def test_run_writes_detailed_log(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    with ResearchRuntime(tmp_path / "logging.sqlite", log_dir=log_dir) as runtime:
        state = runtime.run("Research dynamic delegation using multiple sources", thread_id="logging")
    logs = list(log_dir.glob("*.log"))
    assert len(logs) == 1
    assert state.log_path == str(logs[0].resolve())
    content = logs[0].read_text(encoding="utf-8")
    for marker in ("session_start", "decision", "agent_spawn", "subtask_started", "tool_call", "tool_result", "agent_result", "result_collected", "final_answer", "session_end"):
        assert f'"kind": "{marker}"' in content
    assert "Dynamic delegation overview" in content
    entries = [json.loads(line) for line in content.splitlines()]
    assert all("result" not in entry for entry in entries if entry["kind"] in {"agent_result", "result_collected"})


def test_each_run_gets_a_distinct_log(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    with ResearchRuntime(tmp_path / "logging.sqlite", log_dir=log_dir) as runtime:
        runtime.run("What does LangGraph provide?", thread_id="one")
        runtime.run("What does Python 3.12 improve?", thread_id="two")
    assert len(list(log_dir.glob("*.log"))) == 2


def test_sqlite_pause_and_resume(tmp_path) -> None:
    checkpoint = tmp_path / "resume.sqlite"
    log_dir = tmp_path / "logs"
    with ResearchRuntime(checkpoint, log_dir=log_dir) as runtime:
        paused = runtime.run("What does LangGraph provide?", thread_id="resume-me", pause_after_turn=True)
    assert paused.status == "paused"
    assert paused.depth == 1
    with ResearchRuntime(checkpoint, log_dir=log_dir) as runtime:
        finished = runtime.resume("resume-me")
    assert finished.status == "finished"
    assert finished.final_answer
    assert finished.run_id == paused.run_id
    assert len(list(log_dir.glob("*.log"))) == 1
    assert '"kind": "session_resume"' in next(log_dir.glob("*.log")).read_text(encoding="utf-8")


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
