from __future__ import annotations

import json
import re
import os
import sqlite3
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .llm import LLMError, OpenAIClient
from .models import Action, AgentResult, Contradiction, Decision, Finding, ResearchState, RunEvent, RuntimeLimits
from .registry import CapabilityRegistry
from .tools import ResearchTools, SourceRecord, ToolError

EventSink = Callable[[RunEvent], None]


class RunLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, kind: str, **details: Any) -> None:
        entry = {"timestamp": datetime.now(UTC).isoformat(), "kind": kind, **details}
        with self.lock:
            self.stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self.stream.flush()

    def close(self) -> None:
        with self.lock:
            self.stream.close()


class GraphState(TypedDict):
    data: dict[str, Any]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_BROAD_COVERAGE = {
    "purpose": {"provide", "provides", "enable", "enables", "help", "helps", "let", "lets", "designed", "purpose"},
    "components": {"plugin", "plugins", "tool", "tools", "model", "models", "session", "sessions", "sandbox", "sandboxes", "filesystem", "file", "files", "ui", "interface", "architecture"},
    "operation": {"execute", "executes", "execution", "orchestrate", "orchestrates", "orchestration", "loop", "loops", "workflow", "workflows", "manage", "manages"},
    "use_cases": {"use", "used", "automate", "automates", "automation"},
    "status": {"preview", "beta", "alpha", "released", "release", "available", "availability", "experimental"},
    "limitations": {"limit", "limits", "limitation", "limitations", "only", "requires", "require", "unsupported", "not", "cannot", "doesnt", "doesn"},
}
_BROAD_PROMPT = re.compile(r"\b(?:what\s+does\b.*\bdo|what\s+is|tell\s+me\s+(?:more|about))\b", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"\b(?:advertisement|advertising|subscribe|newsletter|cookie|privacy policy|all rights reserved|skip to|read more|sign up|written by|image courtesy|follow\s+@?|(?:the|this) source (?:discusses|covers|describes)|world's most powerful|try\s+\S+(?:\s+\S+){0,2}\s+today|\[\.\.\.\])\b",
    re.IGNORECASE,
)
_MAX_FINDINGS_PER_RECORD = 3
_MAX_FINDINGS_PER_AGENT_RESULT = 24
_MAX_FINDINGS_PER_RUN = 120
_MAX_SYNTHESIS_RESERVE = 16_384


def _broad_subject(question: str) -> str:
    text = re.sub(r"\b(?:what\s+does|what\s+is|tell\s+me\s+(?:more\s+)?about|tell\s+me\s+more|the|a|an|do|does|is|it)\b", " ", question, flags=re.IGNORECASE)
    return " ".join(re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", text))


def _topic_plan(question: str) -> list[tuple[str, str]]:
    terms = _words(question)
    subject = "TikTok" if "tiktok" in terms else "recommendation system"
    plan = []
    if terms & {"architecture", "implemented", "implementation", "pipeline", "ranking", "retrieval", "stack", "system"}:
        plan.append(("architecture", f"{subject} recommendation system architecture implementation candidate generation retrieval ranking training serving"))
    if terms & {"companies", "company", "competitor", "competitors", "compare", "comparison"}:
        plan.append(("competitors", f"{subject} recommendation systems competitors YouTube Instagram Netflix Meta implementation comparison candidate generation ranking retrieval"))
    if plan or not _BROAD_PROMPT.search(question):
        return plan
    subject = _broad_subject(question)
    return [(area, f"{subject} {' '.join(sorted(keywords))}") for area, keywords in _BROAD_COVERAGE.items()] if subject else []


class DynamicOrchestrator:
    _multi_terms = {"compare", "multiple", "sources", "evidence", "research", "investigate", "report"}

    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def decide(self, state: ResearchState) -> Decision:
        reason = self._limit_reason(state)
        if reason:
            return Decision(rationale=reason, finish=True)
        if state.depth == 0:
            topics = _topic_plan(state.question) if not state.documents else []
            if topics:
                return self._topic_actions(state, topics, "Planned research topics")
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
        if any(gap.startswith("Need evidence for:") for gap in state.evidence.unanswered_questions):
            return "web_researcher"
        candidates = ["web_researcher", "academic_researcher", "company_researcher"]
        if state.documents:
            candidates.insert(0, "document_researcher")
        return next((name for name in candidates if name not in state.used_agents), None)

    @staticmethod
    def _follow_up_topic(state: ResearchState) -> str | None:
        for gap in state.evidence.unanswered_questions:
            match = re.match(r"Need evidence for:\s*(\w+)", gap)
            if match:
                return match.group(1)
        return None

    def _actions(self, state: ResearchState, agents: list[str], follow_up: bool, rationale: str) -> Decision:
        remaining = state.limits.max_total_agents - state.total_agents
        agents = agents[:remaining]
        if not agents:
            return Decision(rationale="Agent budget exhausted", finish=True)
        actions = []
        topic = self._follow_up_topic(state) if follow_up else None
        for name in agents:
            spec = self.registry.agents[name]
            tools = list(spec.core_tools)
            if name == "web_researcher":
                tools = ["web_search"]
            if name == "document_researcher":
                tools = ["search_document"]
            task = state.question
            if follow_up and state.evidence.unanswered_questions:
                gaps = [
                    re.sub(r"^Need evidence for:\s*", "", gap)
                    for gap in state.evidence.unanswered_questions
                    if topic is None or gap == f"Need evidence for: {topic}"
                ]
                task += "\n" + "\n".join(gaps)
            actions.append(Action(agent=name, task=task, topic=topic, skills=spec.skills, tools=tools, follow_up=follow_up))
        return Decision(rationale=rationale, actions=actions, parallel=len(actions) > 1)

    def _topic_actions(self, state: ResearchState, topics: list[tuple[str, str]], rationale: str) -> Decision:
        remaining = state.limits.max_total_agents - state.total_agents
        spec = self.registry.agents["web_researcher"]
        actions = [
            Action(agent=spec.name, topic=topic, task=task, skills=spec.skills, tools=["web_search"])
            for topic, task in topics[: min(remaining, state.limits.max_parallel_agents)]
        ]
        if not actions:
            return Decision(rationale="Agent budget exhausted", finish=True)
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
    def __init__(
        self,
        checkpoint_path: str | Path = ".research-assistant/checkpoints.sqlite",
        event_sink: EventSink | None = None,
        log_dir: str | Path = ".research-assistant/logs",
        llm_client: OpenAIClient | None = None,
    ) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.registry = CapabilityRegistry()
        self.orchestrator = DynamicOrchestrator(self.registry)
        self.llm = llm_client
        self._owns_llm = False
        self.event_sink = event_sink
        self.log_dir = Path(log_dir)
        self._loggers: dict[str, RunLogger] = {}
        self._event_lock = threading.Lock()
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def close(self) -> None:
        for logger in self._loggers.values():
            logger.close()
        self._loggers.clear()
        if self._owns_llm and self.llm:
            self.llm.close()
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
        state = ResearchState(thread_id=thread_id, question=question.strip(), mode=mode, documents=documents or [], fixture_path=fixture_path, limits=limits or RuntimeLimits(), pause_after_turn=pause_after_turn)
        self._start_log(state, "session_start")
        try:
            self._require_live_config(mode)
        except ValueError as exc:
            self._log(state, "session_error", error=str(exc))
            self._close_log(state)
            raise
        self._emit(state, "run_started", "running", summary=f"mode={mode}")
        return self._invoke(state)

    def resume(self, thread_id: str, *, pause_after_turn: bool = False) -> ResearchState:
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.values or "data" not in snapshot.values:
            raise ValueError(f"thread not found: {thread_id}")
        state = ResearchState.model_validate(snapshot.values["data"])
        self._require_live_config(state.mode)
        self._ensure_log(state)
        self._log(state, "session_resume", pause_after_turn=pause_after_turn, status=state.status)
        if state.status == "finished":
            self._close_log(state)
            return state
        state.status = "running"
        state.pause_after_turn = pause_after_turn
        return self._invoke(state)

    def _invoke(self, state: ResearchState) -> ResearchState:
        config = self._config(state.thread_id)
        config["recursion_limit"] = max(25, state.limits.max_research_depth * 4 + 4)
        self._log(state, "graph_invoke", depth=state.depth, status=state.status)
        try:
            result = self.graph.invoke({"data": state.model_dump(mode="json")}, config)
            finished = ResearchState.model_validate(result["data"])
            self._log(finished, "session_end", status=finished.status, depth=finished.depth, used_agents=finished.used_agents, total_agents=finished.total_agents, tool_calls=finished.tool_calls, final_answer=finished.final_answer)
            return finished
        except Exception as exc:
            self._log(state, "session_error", error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            self._close_log(state)

    def _start_log(self, state: ResearchState, kind: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        while True:
            path = self.log_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}.log"
            try:
                path.open("x").close()
                break
            except FileExistsError:
                continue
        state.log_path = str(path.resolve())
        self._loggers[state.run_id] = RunLogger(state.log_path)
        self._log(state, kind, question=state.question, mode=state.mode, documents=state.documents, fixture_path=state.fixture_path, limits=state.limits.model_dump(mode="json"))

    def _ensure_log(self, state: ResearchState) -> None:
        if state.run_id not in self._loggers:
            if state.log_path is None:
                self._start_log(state, "session_start")
            else:
                self._loggers[state.run_id] = RunLogger(state.log_path)

    def _log(self, state: ResearchState, kind: str, **details: Any) -> None:
        self._ensure_log(state)
        self._loggers[state.run_id].write(kind, run_id=state.run_id, thread_id=state.thread_id, **details)

    def _close_log(self, state: ResearchState) -> None:
        logger = self._loggers.pop(state.run_id, None)
        if logger:
            logger.close()

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
        state.decision = self._llm_decide(state) if state.mode == "live" else self.orchestrator.decide(state)
        self._log(state, "decision", decision=state.decision.model_dump(mode="json"))
        event_type = "followup_started" if any(action.follow_up for action in state.decision.actions) else "planning"
        self._emit(state, event_type, "finished", summary=state.decision.rationale)
        return self._dump(state)

    def _dispatch(self, value: GraphState) -> GraphState:
        state = ResearchState.model_validate(value["data"])
        assert state.decision
        actions = state.decision.actions
        for action in actions:
            self._log(state, "agent_spawn", action=action.model_dump(mode="json"))

        grouped: dict[str, list[Action]] = {}
        for action in actions:
            grouped.setdefault(action.agent, []).append(action)

        def run_group(group: list[Action]) -> list[AgentResult]:
            used = state.agent_tool_calls.get(group[0].agent, 0)
            results = []
            for action in group:
                result = self._run_action(state, action, used_override=used)
                results.append(result)
                used += result.tool_calls
            return results

        groups = list(grouped.values())
        if state.decision.parallel:
            self._emit(state, "parallel_started", "running", summary=f"agents={len(actions)}")
            with ThreadPoolExecutor(max_workers=min(len(groups), state.limits.max_parallel_agents)) as pool:
                grouped_results = list(pool.map(run_group, groups))
        else:
            grouped_results = [run_group(group) for group in groups]
        results = [result for group in grouped_results for result in group]
        state.pending_results = results
        state.total_agents += len(actions)
        for action in actions:
            state.agent_calls[action.agent] = state.agent_calls.get(action.agent, 0) + 1
            if action.agent not in state.used_agents:
                state.used_agents.append(action.agent)
        return self._dump(state)

    def _run_action(self, state: ResearchState, action: Action, *, used_override: int | None = None) -> AgentResult:
        self._emit(state, "agent_started", "running", agent=action.agent, task=action.task)
        self._log(state, "task_started", agent=action.agent, task=action.task, skills=action.skills, tools=action.tools, follow_up=action.follow_up)
        findings, errors, calls = [], [], 0
        remaining = state.limits.max_runtime_seconds - (datetime.now(UTC) - state.started_at).total_seconds()
        tools = ResearchTools(state.mode, state.fixture_path, timeout=max(0.001, min(10, remaining)))
        try:
            used = state.agent_tool_calls.get(action.agent, 0) if used_override is None else used_override
            allowed_tools = action.tools
            if state.mode == "live" and (spec := self.registry.agents.get(action.agent)):
                allowed_tools = spec.core_tools
            planned_tools = iter(action.tools)
            tool_name = next(planned_tools, None)
            query = action.task
            seen_queries: set[str] = set()
            while tool_name:
                if (datetime.now(UTC) - state.started_at).total_seconds() >= state.limits.max_runtime_seconds:
                    errors.append("runtime limit reached")
                    break
                if used + calls >= state.limits.max_tool_calls_per_agent:
                    errors.append("agent tool-call limit reached")
                    break
                normalized_query = " ".join(query.lower().split())
                if normalized_query in seen_queries:
                    errors.append("duplicate tool query skipped")
                    break
                seen_queries.add(normalized_query)
                self._emit(state, "tool_started", "running", agent=action.agent, task=tool_name)
                call_details = {"agent": action.agent, "tool": tool_name, "query": query, "documents": state.documents, "call_number": calls + 1}
                self._log(state, "subtask_started", **call_details)
                self._log(state, "tool_call", **call_details)
                try:
                    records = tools.invoke(tool_name, query, state.documents)
                    findings.extend(self._findings(records, action.topic, state.question))
                    findings = findings[:_MAX_FINDINGS_PER_AGENT_RESULT]
                    logged_records = [
                        {
                            "title": record.title,
                            "url": record.url,
                            "source_type": record.source_type,
                            "claim": record.claim,
                            "confidence": record.confidence,
                            "content_length": len(record.content),
                        }
                        for record in records
                    ]
                    self._log(state, "tool_result", agent=action.agent, tool=tool_name, records=logged_records, record_count=len(records))
                    self._emit(state, "tool_finished", "finished", agent=action.agent, task=tool_name, summary=f"records={len(records)}")
                    calls += 1
                    if len(findings) >= _MAX_FINDINGS_PER_AGENT_RESULT:
                        break
                    if state.mode == "live":
                        next_call = self._llm_next_tool(state, action, allowed_tools, tool_name, query, records)
                        if next_call is None:
                            break
                        tool_name, query = next_call
                        continue
                    tool_name = next(planned_tools, None)
                except ToolError as exc:
                    errors.append(str(exc))
                    self._log(state, "tool_error", agent=action.agent, tool=tool_name, error=str(exc), traceback=traceback.format_exc())
                    self._emit(state, "tool_finished", "failed", agent=action.agent, task=tool_name, summary=str(exc))
                    calls += 1
                    if state.mode == "live":
                        break
                    tool_name = next(planned_tools, None)
        finally:
            tools.close()
        result = AgentResult(agent=action.agent, task=action.task, topic=action.topic, findings=findings, tool_calls=calls, errors=errors)
        self._log(
            state,
            "agent_result",
            agent=result.agent,
            task=result.task,
            finding_count=len(result.findings),
            tool_calls=result.tool_calls,
            error_count=len(result.errors),
            claim_sample=[finding.claim[:240] for finding in result.findings[:3]],
        )
        self._emit(state, "agent_finished", "finished" if findings else "partial", agent=action.agent, task=action.task, summary=f"findings={len(findings)} errors={len(errors)}")
        return result

    def _llm_next_tool(
        self,
        state: ResearchState,
        action: Action,
        allowed_tools: list[str],
        tool_name: str,
        query: str,
        records: list[SourceRecord],
    ) -> tuple[str, str] | None:
        research_tokens = self._remaining_research_tokens(state)
        if research_tokens <= 0:
            return None
        prompt = json.dumps(
            {
                "question": state.question,
                "specialist": action.agent,
                "task": action.task,
                "allowed_tools": allowed_tools,
                "last_call": {"tool": tool_name, "query": query},
                "results": [
                    {"title": record.title, "url": record.url, "source_type": record.source_type, "content": record.content[:1_000]}
                    for record in records
                ],
                "response_schema": {"finish": "boolean", "tool": "one allowed tool when finish is false", "query": "specific query or absolute URL when finish is false"},
            },
            ensure_ascii=False,
        )
        system = (
            "Choose one next bounded Python tool call or finish. Use only allowed_tools. "
            "Use fetch_url only with an absolute HTTP(S) URL from results. Tool execution remains in Python. "
            "Return only the requested JSON object."
        )
        max_tokens = self._completion_budget(research_tokens, system, prompt)
        if max_tokens <= 0:
            return None
        try:
            raw, tokens = self._get_llm().complete_json(
                system,
                prompt,
                max_tokens=min(512, max_tokens),
                timeout=self._remaining_time(state),
            )
            self._consume_tokens(state, tokens)
            finish = raw.get("finish")
            if not isinstance(finish, bool) or finish:
                return None
            next_tool, next_query = raw.get("tool"), raw.get("query")
            if not isinstance(next_tool, str) or not isinstance(next_query, str) or next_tool not in allowed_tools:
                return None
            next_query = next_query.strip()
            if not next_query or (next_tool == "fetch_url" and not self._absolute_http_url(next_query)):
                return None
            return next_tool, next_query
        except (LLMError, KeyError, TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _findings(records: list[SourceRecord], topic: str | None = None, question: str = "") -> list[Finding]:
        findings = []
        for record in records:
            content = " ".join(record.content.split())
            candidates = [record.claim] if record.claim else []
            candidates.extend(re.split(r"(?<=[.!?])\s+", content))
            accepted = []
            for candidate in candidates:
                claim = ResearchRuntime._clean_claim(candidate or "")
                if not claim or not ResearchRuntime._relevant(record, claim, topic, question):
                    continue
                claim_terms = _words(claim)
                if any(len(claim_terms & _words(item)) / min(len(claim_terms), len(_words(item))) >= 0.8 for item in accepted):
                    continue
                accepted.append(claim)
                findings.append(Finding(claim=claim, source=record.title, source_type=record.source_type, evidence=claim, confidence=record.confidence, citation=record.url, topic=topic))
                if len(findings) >= _MAX_FINDINGS_PER_RECORD:
                    break
        return findings

    @staticmethod
    def _clean_claim(claim: str) -> str:
        claim = re.sub(r"^#+\s*", "", claim)
        claim = re.sub(r"\s+\[\d+\]\s*$", "", claim)
        claim = " ".join(claim.split()).strip()
        if not claim or claim[0].islower():
            return ""
        if _BOILERPLATE.search(claim):
            return ""
        words = _words(claim)
        if len(words) < 6 or not re.search(r"[.!?]$", claim):
            return ""
        # ponytail: repeated-phrase filter catches obvious navigation junk; use a content-quality classifier if this ceiling matters.
        tokens = re.findall(r"[a-z0-9]+", claim.lower())
        if len(tokens) >= 6 and len(set(tokens)) / len(tokens) < 0.65:
            return ""
        if re.search(r"\b((?:[a-z0-9]+\s+){0,4}[a-z0-9]+)\s+\1\b", claim.lower()):
            return ""
        return claim

    @staticmethod
    def _relevant(record: SourceRecord, claim: str, topic: str | None, question: str) -> bool:
        if not topic:
            return True
        claim_terms = _words(claim)
        if topic in _BROAD_COVERAGE:
            subject_terms = _words(_broad_subject(question))
            source_terms = _words(f"{record.title} {claim}")
            if topic == "status" and claim_terms & _BROAD_COVERAGE["limitations"]:
                return False
            return bool(subject_terms & source_terms) and bool(claim_terms & _BROAD_COVERAGE[topic])
        terms = _words(f"{record.title} {claim} {record.content}")
        if topic == "architecture" and "tiktok" in _words(question):
            architecture_terms = {"architecture", "system", "component", "pipeline", "retrieval", "ranking", "candidate", "serving", "training", "feature", "lakehouse", "recommendation"}
            return bool(terms & {"tiktok", "bytedance"}) and bool(claim_terms & architecture_terms)
        if topic == "competitors":
            competitor_terms = {"youtube", "meta", "instagram", "netflix", "amazon", "spotify", "google"}
            implementation_terms = {"recommendation", "system", "ranking", "retrieval", "candidate", "algorithm", "architecture", "serving", "model"}
            subject_terms = _words(f"{record.title} {claim}")
            return bool(subject_terms & competitor_terms) and bool(claim_terms & implementation_terms)
        return True

    def _collect(self, value: GraphState) -> GraphState:
        state = ResearchState.model_validate(value["data"])
        added = 0
        for result in state.pending_results:
            self._log(
                state,
                "result_collected",
                agent=result.agent,
                task=result.task,
                finding_count=len(result.findings),
                tool_calls=result.tool_calls,
                error_count=len(result.errors),
            )
            state.tool_calls += result.tool_calls
            state.agent_tool_calls[result.agent] = state.agent_tool_calls.get(result.agent, 0) + result.tool_calls
            for finding in result.findings:
                key = (finding.citation, finding.claim)
                if any((item.citation, item.claim) == key for item in state.evidence.findings):
                    continue
                if len(state.evidence.findings) >= _MAX_FINDINGS_PER_RUN:
                    break
                if state.mode != "live":
                    token_cost = max(1, (len(finding.claim) + len(finding.evidence)) // 4)
                    if state.approximate_tokens + token_cost > state.limits.max_tokens_per_run:
                        state.approximate_tokens = state.limits.max_tokens_per_run
                        continue
                state.evidence.findings.append(finding)
                state.evidence.sources = list(dict.fromkeys([*state.evidence.sources, finding.source]))
                if state.mode != "live":
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
        state.final_answer = self._llm_synthesis(state) if state.mode == "live" else self._markdown(state)
        self._log(state, "final_answer", answer=state.final_answer)
        state.status = "finished"
        self._emit(state, "run_finished", "finished", agent="synthesis_critic", summary=f"findings={len(state.evidence.findings)}")
        return self._dump(state)

    @staticmethod
    def _require_live_config(mode: str) -> None:
        if mode != "live":
            return
        missing = [name for name in ("TAVILY_API_KEY", "OPENAI_API_KEY") if not os.getenv(name)]
        if missing:
            raise ValueError(f"live mode requires {' and '.join(missing)}")

    def _get_llm(self) -> OpenAIClient:
        if self.llm is None:
            self.llm = OpenAIClient.from_env()
            self._owns_llm = True
        return self.llm

    def _llm_decide(self, state: ResearchState) -> Decision:
        if reason := self.orchestrator._limit_reason(state):
            return Decision(rationale=reason, finish=True)
        research_tokens = self._remaining_research_tokens(state)
        if research_tokens <= 0:
            return Decision(rationale="Synthesis token reserve reached", finish=True)
        remaining_agents = min(
            state.limits.max_parallel_agents,
            state.limits.max_total_agents - state.total_agents,
        )
        capabilities = {
            name: {"description": spec.description, "tools": spec.core_tools}
            for name, spec in self.registry.agents.items()
            if name != "synthesis_critic" and (name != "document_researcher" or state.documents)
        }
        prompt = json.dumps(
            {
                "question": state.question,
                "research_depth": state.depth,
                "maximum_actions": remaining_agents,
                "capabilities": capabilities,
                "evidence": self._evidence_prompt(state),
                "gaps": state.evidence.unanswered_questions,
                "conflicts": [item.reason for item in state.evidence.contradictions],
                "response_schema": {
                    "rationale": "string",
                    "actions": [{"agent": "capability name", "task": "specific retrieval query", "tools": ["allowed tool"]}],
                    "parallel": "boolean",
                    "finish": "boolean",
                },
            },
            ensure_ascii=False,
        )
        system = (
            "Plan bounded evidence research. Choose only listed agents and their tools. "
            "Choose fetch_url only when the action task is exactly an absolute HTTP(S) URL; otherwise choose web_search. "
            "Tool execution remains in Python; never claim to call HTTP yourself. "
            "Finish only when evidence answers the question or bounds make more research impossible. "
            "Return only the requested JSON object."
        )
        max_tokens = self._completion_budget(research_tokens, system, prompt)
        if max_tokens <= 0:
            return Decision(rationale="Research prompt exceeds remaining token budget", finish=True)
        raw, tokens = self._get_llm().complete_json(
            system,
            prompt,
            max_tokens=max_tokens,
            timeout=self._remaining_time(state),
        )
        self._consume_tokens(state, tokens)
        try:
            decision = Decision.model_validate(raw)
        except ValidationError as exc:
            raise LLMError("model returned invalid research plan") from exc
        if decision.finish:
            return decision
        actions = []
        for action in decision.actions[:remaining_agents]:
            spec = self.registry.agents.get(action.agent)
            if action.agent not in capabilities or spec is None:
                raise LLMError(f"model selected unknown agent: {action.agent}")
            available_calls = state.limits.max_tool_calls_per_agent - state.agent_tool_calls.get(action.agent, 0)
            tools = list(dict.fromkeys(action.tools))[:available_calls]
            if "fetch_url" in tools and not self._absolute_http_url(action.task):
                tools.remove("fetch_url")
                if "web_search" in spec.core_tools and "web_search" not in tools and available_calls:
                    tools.insert(0, "web_search")
            if not tools or any(name not in spec.core_tools for name in tools):
                raise LLMError(f"model selected invalid tools for {action.agent}")
            actions.append(
                Action(
                    agent=action.agent,
                    task=action.task.strip() or state.question,
                    topic=action.topic,
                    skills=spec.skills,
                    tools=tools,
                    follow_up=state.depth > 0,
                )
            )
        if not actions:
            return Decision(rationale="Agent or tool-call budget exhausted", finish=True)
        return Decision(rationale=decision.rationale, actions=actions, parallel=decision.parallel and len(actions) > 1)

    def _llm_synthesis(self, state: ResearchState) -> str:
        if not state.evidence.findings:
            return self._markdown(state)
        remaining_tokens = self._remaining_tokens(state)
        if remaining_tokens <= 0:
            return "# Research answer\n\nToken budget reached before synthesis.\n"
        system = (
            "Write a concise evidence-grounded Markdown answer. Every factual claim line must end with one or more "
            "provided source IDs such as [S1]. Use # headings rather than standalone bold labels when possible. "
            "Use only supplied evidence. Do not add a Sources section."
        )
        prompt = json.dumps({"question": state.question, "evidence": self._evidence_prompt(state)}, ensure_ascii=False)
        max_tokens = self._completion_budget(remaining_tokens, system, prompt)
        if max_tokens <= 0:
            return self._markdown(state)
        answer, tokens = self._get_llm().complete_text(
            system,
            prompt,
            max_tokens=max_tokens,
            timeout=self._remaining_time(state),
        )
        self._consume_tokens(state, tokens)
        return self._ground_synthesis(answer, state.evidence.findings)

    @staticmethod
    def _evidence_prompt(state: ResearchState) -> list[dict[str, str]]:
        source_ids: dict[str, str] = {}
        evidence = []
        for finding in state.evidence.findings:
            target = finding.citation or finding.source
            source_id = source_ids.setdefault(target, f"S{len(source_ids) + 1}")
            evidence.append(
                {
                    "source_id": source_id,
                    "title": finding.source,
                    "url": target,
                    "excerpt": finding.evidence[:2_000],
                    "claim": finding.claim,
                }
            )
        return evidence

    @staticmethod
    def _ground_synthesis(answer: str, findings: list[Finding]) -> str:
        sources: dict[str, Finding] = {}
        targets: dict[str, str] = {}
        for finding in findings:
            target = finding.citation or finding.source
            if target not in targets:
                source_id = f"S{len(targets) + 1}"
                targets[target] = source_id
                sources[source_id] = finding
        lines = []
        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ResearchRuntime._formatting_only(stripped):
                lines.append(line)
                continue
            cited = re.findall(r"\[S(\d+)\]", line)
            if not cited:
                raise LLMError("model synthesis contains an unsupported claim")
            if any(f"S{number}" not in sources for number in cited) or re.search(r"\[S\d+\]\(", line):
                raise LLMError("model synthesis contains an unknown source citation")
            lines.append(re.sub(r"\[S(\d+)\]", r"[\1]", line))
        lines.extend(["", "## Sources", ""])
        for source_id, finding in sources.items():
            number = source_id[1:]
            target = finding.citation or finding.source
            lines.append(f"[{number}] [{finding.source}]({target}) — {finding.source_type}, confidence {finding.confidence:.2f}")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _absolute_http_url(value: str) -> bool:
        parsed = urlparse(value.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _formatting_only(line: str) -> bool:
        if re.fullmatch(r"(?:[-*_]\s*){3,}", line):
            return True
        match = re.fullmatch(r"(?:[-*]\s+)?(?:\*\*|__)(.+?)(?:\*\*|__):?", line)
        return bool(match and len(_words(match.group(1))) <= 10 and not re.search(r"[.!?]", match.group(1)))

    @staticmethod
    def _remaining_tokens(state: ResearchState) -> int:
        return max(0, state.limits.max_tokens_per_run - state.approximate_tokens)

    @staticmethod
    def _synthesis_reserve(state: ResearchState) -> int:
        return min(_MAX_SYNTHESIS_RESERVE, max(1, state.limits.max_tokens_per_run // 5))

    @classmethod
    def _remaining_research_tokens(cls, state: ResearchState) -> int:
        return max(0, cls._remaining_tokens(state) - cls._synthesis_reserve(state))

    @staticmethod
    def _completion_budget(budget: int, system: str, prompt: str) -> int:
        estimated_input = max(1, (len(system) + len(prompt) + 3) // 4)
        return max(0, budget - estimated_input)

    @staticmethod
    def _consume_tokens(state: ResearchState, tokens: int) -> None:
        state.approximate_tokens = min(state.limits.max_tokens_per_run, state.approximate_tokens + tokens)

    @staticmethod
    def _remaining_time(state: ResearchState) -> float:
        elapsed = (datetime.now(UTC) - state.started_at).total_seconds()
        return max(0.001, state.limits.max_runtime_seconds - elapsed)

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
                if overlap >= 0.7 and "filed" not in (base_one & base_two) and (polarity_differs or number_differs):
                    contradictions.append(Contradiction(claim=one.claim, finding_indexes=[left, right], reason=f"{one.source} conflicts with {two.source}"))
        return contradictions

    @staticmethod
    def _gaps(state: ResearchState) -> list[str]:
        if not state.evidence.findings:
            return [state.question]
        gaps = []
        topics = _topic_plan(state.question)
        if topics:
            covered = {finding.topic for finding in state.evidence.findings}
            gaps.extend(f"Need evidence for: {topic}" for topic, _ in topics if topic not in covered)
        if not topics and DynamicOrchestrator._multi_terms & _words(state.question):
            source_types = {finding.source_type for finding in state.evidence.findings}
            if len(source_types) < 2:
                gaps.append("Need a second source category")
        if state.evidence.contradictions:
            gaps.append("Conflicting claims need verification")
        return gaps

    @staticmethod
    def _markdown(state: ResearchState) -> str:
        lines = ["# Research answer", ""]
        source_ids, sources = {}, []
        for finding in state.evidence.findings:
            target = finding.citation or finding.source
            if target not in source_ids:
                source_ids[target] = len(sources) + 1
                sources.append(finding)

        def bullets(findings: list[Finding]) -> list[str]:
            grouped = {}
            for finding in findings:
                grouped.setdefault(" ".join(finding.claim.lower().split()), []).append(finding)
            return [
                f"- {group[0].claim} [{', '.join(str(source_ids[target]) for target in dict.fromkeys(item.citation or item.source for item in group))}]"
                for group in grouped.values()
            ]

        if not state.evidence.findings:
            lines.extend(["No supported answer found within configured bounds.", ""])
        elif topics := _topic_plan(state.question):
            lines.extend(["## Summary", "", "Evidence is grouped by requested topic. Unsupported gaps remain explicit.", ""])
            for topic, _ in topics:
                lines.extend([f"## {topic.replace('_', ' ').title()}", ""])
                findings = [finding for finding in state.evidence.findings if finding.topic == topic]
                if findings:
                    lines.extend(bullets(findings))
                else:
                    lines.append("No supported evidence found.")
                lines.append("")
            additional = [finding for finding in state.evidence.findings if finding.topic is None]
            if additional:
                lines.extend(["## Additional evidence", ""])
                lines.extend(bullets(additional))
                lines.append("")
        else:
            lines.extend(bullets(state.evidence.findings))
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
        for index, finding in enumerate(sources, 1):
            target = finding.citation or finding.source
            lines.append(f"[{index}] [{finding.source}]({target}) — {finding.source_type}, confidence {finding.confidence:.2f}")
        return "\n".join(lines).rstrip() + "\n"

    def _emit(self, state: ResearchState, event_type: str, status: str, *, agent: str | None = None, task: str | None = None, summary: str | None = None) -> None:
        event = RunEvent(run_id=state.run_id, event_type=event_type, agent=agent, task=task, status=status, result_summary=summary)
        with self._event_lock:
            state.events.append(event)
            self._log(state, "event", event=event.model_dump(mode="json"))
            if self.event_sink:
                self.event_sink(event)

    @staticmethod
    def _dump(state: ResearchState) -> GraphState:
        return {"data": state.model_dump(mode="json")}
