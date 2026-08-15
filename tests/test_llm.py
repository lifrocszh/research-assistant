from __future__ import annotations

import sqlite3

import pytest

from research_assistant.engine import ResearchRuntime
from research_assistant.models import Finding, ResearchState
from research_assistant.platform import FixtureChatModel
from research_assistant.registry import CapabilityRegistry
from research_assistant.tools import SourceRecord


class FakeLiveTools:
    calls: list[tuple[str, str]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def invoke(self, name: str, query: str, documents: list[str]) -> list[SourceRecord]:
        self.calls.append((name, query))
        return [
            SourceRecord(
                "Primary source",
                "https://example.test/source",
                "LangGraph provides durable state and checkpointed execution for long-running research workflows.",
                "web",
                "LangGraph provides durable state and checkpointed execution for long-running research workflows.",
                0.9,
            )
        ]

    def close(self) -> None:
        pass


def _live_env(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_live_mode_requires_openai_key_without_injected_model(tmp_path, monkeypatch) -> None:
    _live_env(monkeypatch)
    with ResearchRuntime(tmp_path / "missing-openai.sqlite") as runtime:
        with pytest.raises(ValueError, match="live mode requires OPENAI_API_KEY"):
            runtime.run("current research", thread_id="missing-openai", mode="live")


def test_injected_live_langchain_model_runs_through_deep_agents(tmp_path, monkeypatch) -> None:
    _live_env(monkeypatch)
    FakeLiveTools.calls.clear()
    monkeypatch.setattr("research_assistant.platform.ResearchTools", FakeLiveTools)
    registry = CapabilityRegistry()
    model = FixtureChatModel(registry, [])

    captured = []
    from research_assistant import platform

    real_create = platform.create_deep_agent

    def tracked_create(**kwargs):
        captured.append(kwargs)
        return real_create(**kwargs)

    monkeypatch.setattr(platform, "create_deep_agent", tracked_create)
    with ResearchRuntime(tmp_path / "live.sqlite", model=model) as runtime:
        state = runtime.run("What does LangGraph provide?", thread_id="live", mode="live")

    assert model.selected_agents == ["web_researcher"]
    assert FakeLiveTools.calls and FakeLiveTools.calls[0][0] == "web_search"
    assert captured and captured[0]["tools"] == []
    assert state.used_agents == ["web_researcher"]
    assert "[Primary source](https://example.test/source)" in (state.final_answer or "")


def test_fixture_uses_scripted_model_not_injected_live_model(tmp_path) -> None:
    injected = FixtureChatModel(CapabilityRegistry(), [])
    with ResearchRuntime(tmp_path / "fixture.sqlite", model=injected) as runtime:
        state = runtime.run("What does LangGraph provide?", thread_id="fixture")
    assert state.status == "finished"
    assert injected.selected_agents == []


def test_live_answer_falls_back_when_claims_are_not_source_grounded() -> None:
    state = ResearchState(thread_id="grounding", question="What does LangGraph provide?", mode="live")
    state.evidence.findings.append(
        Finding(
            claim="LangGraph supports durable checkpointed agent execution.",
            source="LangGraph docs",
            source_type="web",
            evidence="LangGraph supports durable checkpointed agent execution.",
            confidence=0.9,
            citation="https://example.test/langgraph",
        )
    )
    answer = ResearchRuntime._ground_live_or_fallback("# Answer\n\nUnsupported uncited claim.", state)
    assert "Unsupported uncited claim" not in answer
    assert "[LangGraph docs](https://example.test/langgraph)" in answer


def test_pause_finishes_delegated_turn_and_resume_only_synthesizes(tmp_path, monkeypatch) -> None:
    _live_env(monkeypatch)
    monkeypatch.setattr("research_assistant.platform.ResearchTools", FakeLiveTools)
    checkpoint = tmp_path / "resume.sqlite"
    first_model = FixtureChatModel(CapabilityRegistry(), [])
    with ResearchRuntime(checkpoint, model=first_model) as runtime:
        paused = runtime.run("What does LangGraph provide?", thread_id="resume", mode="live", pause_after_turn=True)
    assert paused.status == "paused"
    assert paused.evidence.findings

    unused_resume_model = FixtureChatModel(CapabilityRegistry(), [])
    with ResearchRuntime(checkpoint, model=unused_resume_model) as runtime:
        finished = runtime.resume("resume")
    assert finished.status == "finished"
    assert finished.run_id == paused.run_id
    assert unused_resume_model.selected_agents == []


def test_resume_distinguishes_unknown_and_legacy_threads(tmp_path) -> None:
    checkpoint = tmp_path / "resume-errors.sqlite"
    connection = sqlite3.connect(checkpoint)
    connection.execute("CREATE TABLE checkpoints(thread_id TEXT NOT NULL)")
    connection.execute("INSERT INTO checkpoints(thread_id) VALUES (?)", ("legacy",))
    connection.commit()
    connection.close()

    with ResearchRuntime(checkpoint) as runtime:
        with pytest.raises(ValueError, match="legacy or incomplete checkpoint"):
            runtime.resume("legacy")
        with pytest.raises(ValueError, match="thread not found: missing"):
            runtime.resume("missing")
