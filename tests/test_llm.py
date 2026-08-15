from __future__ import annotations

import json

import httpx
import pytest

from research_assistant.engine import ResearchRuntime
from research_assistant.llm import LLMError, OpenAIClient
from research_assistant.models import Finding, ResearchState, RuntimeLimits
from research_assistant.tools import SourceRecord


class FakeLLM:
    def __init__(self, plans: list[dict] | None = None, answers: list[str] | None = None) -> None:
        self.plans = plans or []
        self.answers = answers or []
        self.calls: list[str] = []

    def complete_json(self, system: str, prompt: str, *, max_tokens: int, timeout: float) -> tuple[dict, int]:
        self.calls.append("plan")
        return self.plans.pop(0), 5

    def complete_text(self, system: str, prompt: str, *, max_tokens: int, timeout: float) -> tuple[str, int]:
        self.calls.append("synthesize")
        return self.answers.pop(0), 7


class FakeLiveTools:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def invoke(self, name: str, query: str, documents: list[str]) -> list[SourceRecord]:
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


class AdaptiveFakeLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(answers=["# Answer\n\n- LangGraph supports durable execution through persisted checkpoints. [S1]"])
        self.evidence_urls: list[list[str]] = []
        self.next_tools: list[str] = []

    def complete_json(self, system: str, prompt: str, *, max_tokens: int, timeout: float) -> tuple[dict, int]:
        self.calls.append("plan")
        payload = json.loads(prompt)
        if "last_call" in payload:
            last_tool = payload["last_call"]["tool"]
            self.next_tools.append(last_tool)
            if last_tool == "web_search":
                return {"finish": False, "tool": "fetch_url", "query": "https://example.test/overview"}, 5
            return {"finish": True}, 5
        evidence_urls = [item["url"] for item in payload["evidence"]]
        self.evidence_urls.append(evidence_urls)
        return _plan() if not evidence_urls else _plan(finish=True), 5


class AdaptiveLiveTools:
    calls: list[tuple[str, str]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def invoke(self, name: str, query: str, documents: list[str]) -> list[SourceRecord]:
        self.calls.append((name, query))
        if name == "web_search":
            return [
                SourceRecord(
                    "LangGraph overview",
                    "https://example.test/overview",
                    "LangGraph supports durable execution through persisted checkpoints.",
                    "web",
                    "LangGraph supports durable execution through persisted checkpoints.",
                    0.9,
                )
            ]
        return [
            SourceRecord(
                "LangGraph overview",
                "https://example.test/overview",
                "LangGraph resumes durable workflows from persisted state after interruptions.",
                "web",
                "LangGraph resumes durable workflows from persisted state after interruptions.",
                0.9,
            )
        ]

    def close(self) -> None:
        pass


def _plan(*, finish: bool = False) -> dict:
    return {
        "rationale": "Evidence sufficient" if finish else "Search primary sources",
        "actions": [] if finish else [{"agent": "web_researcher", "task": "LangGraph durable execution", "tools": ["web_search"]}],
        "parallel": False,
        "finish": finish,
    }


def _live_env(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")


def test_live_mode_requires_openai_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with ResearchRuntime(tmp_path / "missing-openai.sqlite") as runtime:
        with pytest.raises(ValueError, match="live mode requires OPENAI_API_KEY"):
            runtime.run("current research", thread_id="missing-openai", mode="live")


def test_mocked_planning_response_selects_python_tool(tmp_path) -> None:
    llm = FakeLLM(plans=[_plan()])
    with ResearchRuntime(tmp_path / "planning.sqlite", llm_client=llm) as runtime:
        state = ResearchState(thread_id="planning", question="How does LangGraph persist state?", mode="live")
        decision = runtime._llm_decide(state)
    assert decision.actions[0].agent == "web_researcher"
    assert decision.actions[0].tools == ["web_search"]
    assert llm.calls == ["plan"]


def test_planner_replaces_fetch_url_without_url(tmp_path) -> None:
    plan = _plan()
    plan["actions"][0]["tools"] = ["fetch_url"]
    llm = FakeLLM(plans=[plan])
    with ResearchRuntime(tmp_path / "fetch-url.sqlite", llm_client=llm) as runtime:
        state = ResearchState(thread_id="fetch-url", question="Research LangGraph", mode="live")
        decision = runtime._llm_decide(state)
    assert decision.actions[0].tools == ["web_search"]


def test_mocked_synthesis_is_grounded_to_source_url(tmp_path) -> None:
    llm = FakeLLM(answers=["# Answer\n\n- LangGraph supports durable execution. [S1]"])
    state = ResearchState(thread_id="synthesis", question="What does LangGraph provide?", mode="live")
    state.evidence.findings.append(
        Finding(
            claim="LangGraph supports durable execution.",
            source="LangGraph docs",
            source_type="web",
            evidence="LangGraph supports durable execution through checkpoints.",
            confidence=0.9,
            citation="https://example.test/langgraph",
        )
    )
    with ResearchRuntime(tmp_path / "synthesis.sqlite", llm_client=llm) as runtime:
        answer = runtime._llm_synthesis(state)
    assert "LangGraph supports durable execution. [1]" in answer
    assert "[LangGraph docs](https://example.test/langgraph)" in answer
    assert llm.calls == ["synthesize"]


def test_synthesis_uses_markdown_when_prompt_exceeds_remaining_budget(tmp_path) -> None:
    llm = FakeLLM(answers=["unused"])
    state = ResearchState(
        thread_id="small-budget-synthesis",
        question="What does LangGraph provide?",
        mode="live",
        limits=RuntimeLimits(max_tokens_per_run=100),
    )
    state.evidence.findings.append(
        Finding(
            claim="Supported claim " + ("with detail " * 80),
            source="Source",
            source_type="web",
            evidence="Supported evidence.",
            confidence=0.9,
            citation="https://example.test/source",
        )
    )
    with ResearchRuntime(tmp_path / "small-budget-synthesis.sqlite", llm_client=llm) as runtime:
        answer = runtime._llm_synthesis(state)
    assert "Token budget reached before synthesis" not in answer
    assert "Supported claim" in answer
    assert llm.calls == []


def test_provider_rejects_malformed_model_output() -> None:
    client = OpenAIClient("test", "test-model")
    client.client.close()
    client.client = httpx.Client(
        base_url="https://example.test/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": "not json"}}]},
            )
        ),
    )
    try:
        with pytest.raises(LLMError, match="malformed JSON"):
            client.complete_json("system", "prompt", max_tokens=10, timeout=1)
    finally:
        client.close()


def test_planner_rejects_malformed_model_shape(tmp_path) -> None:
    llm = FakeLLM(plans=[{"rationale": "missing decision fields"}])
    with ResearchRuntime(tmp_path / "malformed-plan.sqlite", llm_client=llm) as runtime:
        state = ResearchState(thread_id="malformed-plan", question="Research this", mode="live")
        with pytest.raises(LLMError, match="invalid research plan"):
            runtime._llm_decide(state)


def test_synthesis_rejects_unknown_or_missing_citations() -> None:
    findings = [
        Finding(
            claim="Supported claim has enough words to remain valid.",
            source="Source",
            source_type="web",
            evidence="Supported claim has enough words to remain valid.",
            confidence=0.8,
            citation="https://example.test/source",
        )
    ]
    with pytest.raises(LLMError, match="unknown source citation"):
        ResearchRuntime._ground_synthesis("Unsupported claim. [S9]", findings)
    with pytest.raises(LLMError, match="unsupported claim"):
        ResearchRuntime._ground_synthesis("Unsupported claim without citation.", findings)


def test_synthesis_allows_markdown_labels_but_not_uncited_claims() -> None:
    findings = [
        Finding(
            claim="Supported claim has enough words to remain valid.",
            source="Source",
            source_type="web",
            evidence="Supported claim has enough words to remain valid.",
            confidence=0.8,
            citation="https://example.test/source",
        )
    ]
    answer = ResearchRuntime._ground_synthesis("**Key drivers**\n\nSupported claim. [S1]", findings)
    assert "**Key drivers**" in answer
    with pytest.raises(LLMError, match="unsupported claim"):
        ResearchRuntime._ground_synthesis("**This is an unsupported factual claim.**", findings)


def test_fixture_mode_makes_zero_llm_calls(tmp_path) -> None:
    llm = FakeLLM()
    with ResearchRuntime(tmp_path / "fixture.sqlite", llm_client=llm) as runtime:
        state = runtime.run("What does LangGraph provide?", thread_id="fixture")
    assert state.status == "finished"
    assert llm.calls == []


def test_live_mode_uses_new_evidence_to_choose_followup_tool(tmp_path, monkeypatch) -> None:
    _live_env(monkeypatch)
    AdaptiveLiveTools.calls.clear()
    monkeypatch.setattr("research_assistant.engine.ResearchTools", AdaptiveLiveTools)
    llm = AdaptiveFakeLLM()

    with ResearchRuntime(tmp_path / "adaptive.sqlite", llm_client=llm) as runtime:
        state = runtime.run(
            "How does LangGraph persist research state?",
            thread_id="adaptive",
            mode="live",
            limits=RuntimeLimits(max_parallel_agents=1, max_total_agents=2, max_research_depth=3),
        )

    assert state.status == "finished"
    assert AdaptiveLiveTools.calls == [
        ("web_search", "LangGraph durable execution"),
        ("fetch_url", "https://example.test/overview"),
    ]
    assert llm.next_tools == ["web_search", "fetch_url"]
    assert llm.evidence_urls[0] == []
    assert set(llm.evidence_urls[1]) == {"https://example.test/overview"}
    assert llm.calls[-1] == "synthesize"


def test_live_checkpoint_resume_keeps_llm_workflow(tmp_path, monkeypatch) -> None:
    _live_env(monkeypatch)
    monkeypatch.setattr("research_assistant.engine.ResearchTools", FakeLiveTools)
    checkpoint = tmp_path / "resume.sqlite"
    logs = tmp_path / "logs"
    first_llm = FakeLLM(plans=[_plan()])
    with ResearchRuntime(checkpoint, log_dir=logs, llm_client=first_llm) as runtime:
        paused = runtime.run(
            "What does LangGraph provide?",
            thread_id="live-resume",
            mode="live",
            pause_after_turn=True,
            limits=RuntimeLimits(max_parallel_agents=1, max_total_agents=2, max_research_depth=2),
        )
    second_llm = FakeLLM(plans=[_plan(finish=True)], answers=["# Answer\n\n- LangGraph provides checkpointed execution. [S1]"])
    with ResearchRuntime(checkpoint, log_dir=logs, llm_client=second_llm) as runtime:
        finished = runtime.resume("live-resume")
    assert paused.status == "paused"
    assert finished.status == "finished"
    assert finished.run_id == paused.run_id
    assert second_llm.calls == ["plan", "synthesize"]
