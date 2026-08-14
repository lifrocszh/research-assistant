from __future__ import annotations

import re
import os
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .models import Action, AgentResult, Contradiction, Decision, Finding, ResearchState, RunEvent, RuntimeLimits
from .registry import CapabilityRegistry
from .tools import ResearchTools, SourceRecord, ToolError

EventSink = Callable[[RunEvent], None]


class GraphState(TypedDict):
    data: dict[str, Any]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class DynamicOrchestrator:
    _multi_terms = {"compare", "multiple", "sources", "evidence", "research", "investigate", "report"}

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def decide(self, state: ResearchState) -> Decision:
        reason = self._limit_reason(state)
        if reason:
            return Decision(rationale=reason, finish=True)
        if state.depth == 0:
            agents = self._initial_agents(state)
            return self._actions(state, agents, False, "Selected by capability match")
        follow_up = self._follow_up_agent(state)
        if follow_up:
            return self._actions(state, [follow_up], True, "Targeted follow-up for evidence gap or conflict")
        return Decision(rationale="Evidence sufficient or research bounds reached", finish=True)

    def _initial_agents(self, state: ResearchState) -> list[str]:
        discovered = [spec.name for spec in self.registry.discover_agents(state.question) if spec.name != "synthesis_critic"]
        if state.documents:
            discovered.insert(0, "document_researcher")
        if not discovered:
            discovered = ["web_researcher"]
        unique = list(dict.fromkeys(discovered))
        wanted = 3 if self._multi_terms & _words(state.question) else 1
        return unique[: min(wanted, state.limits.max_parallel_agents)]

    def _follow_up_agent(self, state: ResearchState) -> str | None:
        if state.depth >= state.limits.max_research_depth:
            return None
        needs_more = bool(state.evidence.unanswered_questions or state.evidence.contradictions)
        if not needs_more:
            return None
        candidates = ["web_researcher", "academic_researcher", "company_researcher"]
        if state.documents:
            candidates.insert(0, "document_researcher")
        return next((name for name in candidates if name not in state.used_agents), None)

    def _actions(self, state: ResearchState, agents: list[str], follow_up: bool, rationale: str) -> Decision:
        remaining = state.limits.max_total_agents - state.total_agents
        agents = agents[:remaining]
        if not agents:
            return Decision(rationale="Agent budget exhausted", finish=True)
        actions = []
        for name in agents:
            spec = self.registry.agents[name]
            tools = list(spec.core_tools)
            if name == "web_researcher":
                tools = ["web_search"]
            if name == "document_researcher":
                tools = ["search_document"]
            actions.append(Action(agent=name, task=state.question, skills=spec.skills, tools=tools, follow_up=follow_up))
        return Decision(rationale=rationale, actions=actions, parallel=len(actions) > 1)

    @staticmethod
    def _limit_reason(state: ResearchState) -> str | None:
        elapsed = (datetime.now(UTC) - state.started_at).total_seconds()
        if elapsed >= state.limits.max_runtime_seconds:
            return "Runtime limit reached"
        if state.total_agents >= state.limits.max_total_agents:
            return "Agent limit reached"
        if state.depth >= state.limits.max_research_depth:
            return "Research depth limit reached"
        if state.approximate_tokens >= state.limits.max_tokens_per_run:
            return "Token budget reached"
        return None


class ResearchRuntime:
    def __init__(self, checkpoint_path: str | Path = ".research-assistant/checkpoints.sqlite", event_sink: EventSink | None = None) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.registry = CapabilityRegistry()
        self.orchestrator = DynamicOrchestrator(self.registry)
        self.event_sink = event_sink
        self._event_lock = threading.Lock()
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ResearchRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(
        self,
        question: str,
        *,
        thread_id: str,
        mode: str = "fixture",
        documents: list[str] | None = None,
        fixture_path: str | None = None,
        limits: RuntimeLimits | None = None,
        pause_after_turn: bool = False,
    ) -> ResearchState:
        if not question.strip():
            raise ValueError("question must not be empty")
        if mode == "live" and not os.getenv("TAVILY_API_KEY"):
            raise ValueError("live mode requires TAVILY_API_KEY")
        state = ResearchState(thread_id=thread_id, question=question.strip(), mode=mode, documents=documents or [], fixture_path=fixture_path, limits=limits or RuntimeLimits(), pause_after_turn=pause_after_turn)
        self._emit(state, "run_started", "running", summary=f"mode={mode}")
        return self._invoke(state)

    def resume(self, thread_id: str, *, pause_after_turn: bool = False) -> ResearchState:
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.values or "data" not in snapshot.values:
            raise ValueError(f"thread not found: {thread_id}")
        state = ResearchState.model_validate(snapshot.values["data"])
        if state.status == "finished":
            return state
        state.status = "running"
        state.pause_after_turn = pause_after_turn
        return self._invoke(state)

    def _invoke(self, state: ResearchState) -> ResearchState:
        config = self._config(state.thread_id)
        config["recursion_limit"] = max(25, state.limits.max_research_depth * 4 + 4)
        result = self.graph.invoke({"data": state.model_dump(mode="json")}, config)
        return ResearchState.model_validate(result["data"])

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GraphState)
        graph.add_node("decide", self._decide)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("collect", self._collect)
        graph.add_node("critique", self._critique)
        graph.add_node("synthesize", self._synthesize)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", lambda value: "finish" if ResearchState.model_validate(value["data"]).decision.finish else "research", {"finish": "synthesize", "research": "dispatch"})
        graph.add_edge("dispatch", "collect")
        graph.add_edge("collect", "critique")
        graph.add_conditional_edges("critique", lambda value: "pause" if ResearchState.model_validate(value["data"]).status == "paused" else "continue", {"pause": END, "continue": "decide"})
        graph.add_edge("synthesize", END)
        return graph

    def _decide(self, value: GraphState) -> GraphState:
        state = ResearchState.model_validate(value["data"])
        state.decision = self.orchestrator.decide(state)
        event_type = "followup_started" if any(action.follow_up for action in state.decision.actions) else "planning"
        self._emit(state, event_type, "finished", summary=state.decision.rationale)
        return self._dump(state)

    def _dispatch(self, value: GraphState) -> GraphState:
        state = ResearchState.model_validate(value["data"])
        assert state.decision
        actions = state.decision.actions
        if state.decision.parallel:
            self._emit(state, "parallel_started", "running", summary=f"agents={len(actions)}")
            with ThreadPoolExecutor(max_workers=min(len(actions), state.limits.max_parallel_agents)) as pool:
                results = list(pool.map(lambda action: self._run_action(state, action), actions))
        else:
            results = [self._run_action(state, action) for action in actions]
        state.pending_results = results
        state.total_agents += len(actions)
        for action in actions:
            state.agent_calls[action.agent] = state.agent_calls.get(action.agent, 0) + 1
            if action.agent not in state.used_agents:
                state.used_agents.append(action.agent)
        return self._dump(state)

    def _run_action(self, state: ResearchState, action: Action) -> AgentResult:
        self._emit(state, "agent_started", "running", agent=action.agent, task=action.task)
        findings, errors, calls = [], [], 0
        remaining = state.limits.max_runtime_seconds - (datetime.now(UTC) - state.started_at).total_seconds()
        tools = ResearchTools(state.mode, state.fixture_path, timeout=max(0.001, min(10, remaining)))
        try:
            used = state.agent_tool_calls.get(action.agent, 0)
            for tool_name in action.tools:
                if (datetime.now(UTC) - state.started_at).total_seconds() >= state.limits.max_runtime_seconds:
                    errors.append("runtime limit reached")
                    break
                if used + calls >= state.limits.max_tool_calls_per_agent:
                    errors.append("agent tool-call limit reached")
                    break
                self._emit(state, "tool_started", "running", agent=action.agent, task=tool_name)
                try:
                    records = tools.invoke(tool_name, action.task, state.documents)
                    findings.extend(self._findings(records))
                    self._emit(state, "tool_finished", "finished", agent=action.agent, task=tool_name, summary=f"records={len(records)}")
                except ToolError as exc:
                    errors.append(str(exc))
                    self._emit(state, "tool_finished", "failed", agent=action.agent, task=tool_name, summary=str(exc))
                calls += 1
        finally:
            tools.close()
        self._emit(state, "agent_finished", "finished" if findings else "partial", agent=action.agent, task=action.task, summary=f"findings={len(findings)} errors={len(errors)}")
        return AgentResult(agent=action.agent, task=action.task, findings=findings, tool_calls=calls, errors=errors)

    @staticmethod
    def _findings(records: list[SourceRecord]) -> list[Finding]:
        findings = []
        for record in records:
            evidence = " ".join(record.content.split())[:1_000]
            claim = record.claim or re.split(r"(?<=[.!?])\s+", evidence, maxsplit=1)[0]
            if claim:
                findings.append(Finding(claim=claim, source=record.title, source_type=record.source_type, evidence=evidence, confidence=record.confidence, citation=record.url))
        return findings

    def _collect(self, value: GraphState) -> GraphState:
        state = ResearchState.model_validate(value["data"])
        added = 0
        for result in state.pending_results:
            state.tool_calls += result.tool_calls
            state.agent_tool_calls[result.agent] = state.agent_tool_calls.get(result.agent, 0) + result.tool_calls
            for finding in result.findings:
                key = (finding.citation, finding.claim)
                if any((item.citation, item.claim) == key for item in state.evidence.findings):
                    continue
                token_cost = max(1, (len(finding.claim) + len(finding.evidence)) // 4)
                if state.approximate_tokens + token_cost > state.limits.max_tokens_per_run:
                    state.approximate_tokens = state.limits.max_tokens_per_run
                    continue
                state.evidence.findings.append(finding)
                state.evidence.sources = list(dict.fromkeys([*state.evidence.sources, finding.source]))
                state.approximate_tokens += token_cost
                added += 1
        state.pending_results = []
        state.depth += 1
        self._emit(state, "evidence_added", "finished", summary=f"added={added} total={len(state.evidence.findings)}")
        return self._dump(state)

    def _critique(self, value: GraphState) -> GraphState:
        state = ResearchState.model_validate(value["data"])
        state.evidence.claims = self._claim_index(state.evidence.findings)
        state.evidence.contradictions = self._contradictions(state.evidence.findings)
        state.evidence.unanswered_questions = self._gaps(state)
        for contradiction in state.evidence.contradictions:
            self._emit(state, "conflict_detected", "finished", summary=contradiction.reason)
        if state.pause_after_turn:
            state.status = "paused"
        return self._dump(state)

    def _synthesize(self, value: GraphState) -> GraphState:
        state = ResearchState.model_validate(value["data"])
        self._emit(state, "synthesis_started", "running", agent="synthesis_critic", task=state.question)
        state.final_answer = self._markdown(state)
        state.status = "finished"
        self._emit(state, "run_finished", "finished", agent="synthesis_critic", summary=f"findings={len(state.evidence.findings)}")
        return self._dump(state)

    @staticmethod
    def _claim_index(findings: list[Finding]) -> dict[str, list[int]]:
        claims: dict[str, list[int]] = {}
        for index, finding in enumerate(findings):
            key = " ".join(sorted(_words(finding.claim) - {"a", "an", "the", "because", "primarily", "did", "not", "no", "never"}))
            claims.setdefault(key, []).append(index)
        return claims

    @staticmethod
    def _contradictions(findings: list[Finding]) -> list[Contradiction]:
        contradictions = []
        negatives = {"not", "no", "never"}
        for left in range(len(findings)):
            for right in range(left + 1, len(findings)):
                one, two = findings[left], findings[right]
                if one.source == two.source:
                    continue
                words_one, words_two = _words(one.claim), _words(two.claim)
                base_one, base_two = words_one - negatives - {"did", "primarily"}, words_two - negatives - {"did", "primarily"}
                overlap = len(base_one & base_two) / max(1, min(len(base_one), len(base_two)))
                polarity_differs = bool(words_one & negatives) != bool(words_two & negatives)
                nums_one, nums_two = set(re.findall(r"\d+(?:\.\d+)?", one.claim)), set(re.findall(r"\d+(?:\.\d+)?", two.claim))
                number_differs = bool(nums_one and nums_two and nums_one != nums_two)
                if overlap >= 0.7 and (polarity_differs or number_differs):
                    contradictions.append(Contradiction(claim=one.claim, finding_indexes=[left, right], reason=f"{one.source} conflicts with {two.source}"))
        return contradictions

    @staticmethod
    def _gaps(state: ResearchState) -> list[str]:
        if not state.evidence.findings:
            return [state.question]
        gaps = []
        if DynamicOrchestrator._multi_terms & _words(state.question):
            source_types = {finding.source_type for finding in state.evidence.findings}
            if len(source_types) < 2:
                gaps.append("Need a second source category")
        if state.evidence.contradictions:
            gaps.append("Conflicting claims need verification")
        return gaps

    @staticmethod
    def _markdown(state: ResearchState) -> str:
        lines = ["# Research answer", ""]
        if not state.evidence.findings:
            lines.extend(["No supported answer found within configured bounds.", ""])
        else:
            for index, finding in enumerate(state.evidence.findings, 1):
                lines.append(f"- {finding.claim} [{index}]")
            lines.append("")
        if state.evidence.contradictions:
            lines.extend(["## Conflicts", ""])
            lines.extend(f"- {item.reason}" for item in state.evidence.contradictions)
            lines.append("")
        if state.evidence.unanswered_questions:
            lines.extend(["## Unanswered questions", ""])
            lines.extend(f"- {item}" for item in state.evidence.unanswered_questions)
            lines.append("")
        lines.extend(["## Sources", ""])
        for index, finding in enumerate(state.evidence.findings, 1):
            target = finding.citation or finding.source
            lines.append(f"[{index}] [{finding.source}]({target}) — {finding.source_type}, confidence {finding.confidence:.2f}")
        return "\n".join(lines).rstrip() + "\n"

    def _emit(self, state: ResearchState, event_type: str, status: str, *, agent: str | None = None, task: str | None = None, summary: str | None = None) -> None:
        event = RunEvent(run_id=state.run_id, event_type=event_type, agent=agent, task=task, status=status, result_summary=summary)
        with self._event_lock:
            state.events.append(event)
            if self.event_sink:
                self.event_sink(event)

    @staticmethod
    def _dump(state: ResearchState) -> GraphState:
        return {"data": state.model_dump(mode="json")}
