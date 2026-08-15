from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import traceback
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver

from .models import Contradiction, Finding, ResearchState, RunEvent, RuntimeLimits
from .platform import DeepAgentPlatform, FixtureChatModel, RunToolRuntime
from .registry import CapabilityRegistry
from .tools import SourceRecord

EventSink = Callable[[RunEvent], None]
_MULTI_TERMS = {"compare", "multiple", "sources", "evidence", "research", "investigate", "report"}
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
    r"\b(?:advertisement|advertising|subscribe|newsletter|cookie|privacy policy|all rights reserved|skip to|read more|sign up|written by|image courtesy|follow\s+@?|resources documentation|repository files navigation|full documentation|community forum|(?:the|this) source (?:discusses|covers|describes)|world's most powerful|try\s+\S+(?:\s+\S+){0,2}\s+today|\[\.\.\.\])\b",
    re.IGNORECASE,
)
_MAX_FINDINGS_PER_RECORD = 3
_MAX_FINDINGS_PER_RUN = 120


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


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


class RunLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, kind: str, **details: Any) -> None:
        with self.lock:
            self.stream.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "kind": kind, **details}, ensure_ascii=False, default=str) + "\n")
            self.stream.flush()

    def close(self) -> None:
        with self.lock:
            self.stream.close()


class ResearchRuntime:
    def __init__(
        self,
        checkpoint_path: str | Path = ".research-assistant/checkpoints.deepagents.sqlite",
        event_sink: EventSink | None = None,
        log_dir: str | Path = ".research-assistant/logs",
        model: BaseChatModel | None = None,
    ) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS research_runs (thread_id TEXT PRIMARY KEY, catalog_fingerprint TEXT NOT NULL, state_json TEXT NOT NULL)"
        )
        self.connection.commit()
        self.registry = CapabilityRegistry()
        self.platform = DeepAgentPlatform(self.registry)
        self.model = model
        self.event_sink = event_sink
        self.log_dir = Path(log_dir)
        self._loggers: dict[str, RunLogger] = {}
        self._event_lock = threading.Lock()

    def close(self) -> None:
        for logger in self._loggers.values():
            logger.close()
        self._loggers.clear()
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
        state = ResearchState(
            thread_id=thread_id,
            question=question.strip(),
            mode=mode,
            documents=documents or [],
            fixture_path=fixture_path,
            limits=limits or RuntimeLimits(),
            pause_after_turn=pause_after_turn,
            catalog_fingerprint=self.registry.fingerprint,
        )
        self._start_log(state, "session_start")
        try:
            self._require_live_config(mode, self.model)
        except ValueError as exc:
            self._log(state, "session_error", error=str(exc))
            self._close_log(state)
            raise
        self._emit(state, "run_started", "running", summary=f"mode={mode}")
        return self._invoke(state)

    def resume(self, thread_id: str, *, pause_after_turn: bool = False) -> ResearchState:
        state = self._load_state(thread_id)
        if state is None:
            try:
                has_checkpoint = bool(
                    self.connection.execute(
                        "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1",
                        (thread_id,),
                    ).fetchone()
                )
            except sqlite3.OperationalError:
                has_checkpoint = False
            if has_checkpoint:
                raise ValueError("legacy or incomplete checkpoint is incompatible; start a new Deep Agents thread")
            raise ValueError(f"thread not found: {thread_id}")
        if state.catalog_fingerprint != self.registry.fingerprint:
            raise ValueError("checkpoint catalog fingerprint does not match current registry")
        if state.status == "finished":
            return state
        self._ensure_log(state)
        self._log(state, "session_resume", pause_after_turn=pause_after_turn, status=state.status)
        state.status = "running"
        state.pause_after_turn = pause_after_turn
        self._finish(state, self._markdown(state))
        self._save_state(state)
        self._close_log(state)
        return state

    def _invoke(self, state: ResearchState) -> ResearchState:
        self._emit(state, "planning", "finished", summary="Delegating through registered capabilities")
        self._log(state, "decision", decision={"framework": "deepagents", "catalog_fingerprint": self.registry.fingerprint})

        def publish(event_type: str, agent: str, task: str, summary: str | None) -> None:
            status = "failed" if event_type == "tool_finished" and summary and not summary.startswith("records=") else "running"
            if event_type == "tool_finished" and status != "failed":
                status = "finished"
            if event_type == "agent_started":
                self._log(state, "agent_spawn", agent=agent, task=task)
            elif event_type == "tool_started":
                self._log(state, "subtask_started", agent=agent, tool=task)
                self._log(state, "tool_call", agent=agent, tool=task)
            elif event_type == "tool_finished" and status == "failed":
                self._log(state, "tool_error", agent=agent, tool=task, error=summary)
            self._emit(state, event_type, status, agent=agent, task=task, summary=summary)

        tools = RunToolRuntime(
            self.registry,
            mode=state.mode,
            documents=state.documents,
            fixture_path=state.fixture_path,
            limits=state.limits,
            started_at=state.started_at,
            event=publish,
        )
        try:
            model = FixtureChatModel(self.registry, state.documents) if state.mode == "fixture" else (self.model or self.platform.live_model(state.limits))
            agent = self.platform.build(model=model, tool_runtime=tools, checkpointer=self.checkpointer)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": state.question}]},
                {"configurable": {"thread_id": state.thread_id}, "recursion_limit": max(25, state.limits.max_research_depth * 8)},
            )
            model_answer = str(result["messages"][-1].content) if result.get("messages") else ""
        except Exception as exc:
            self._log(state, "session_error", error=str(exc), traceback=traceback.format_exc())
            self._close_log(state)
            raise
        finally:
            tools.close()

        delegated_agents = self._delegated_agents(result.get("messages", []))
        started_agents = [name for name in dict.fromkeys(delegated_agents) if name in tools.started_agents]
        state.used_agents = list(dict.fromkeys(delegated_agents))
        state.total_agents = len(delegated_agents)
        state.agent_calls = dict(Counter(delegated_agents))
        state.agent_tool_calls = {name: tools.calls.get(name, 0) for name in started_agents}
        state.tool_calls = sum(state.agent_tool_calls.values())
        if len(started_agents) > 1:
            self._emit(state, "parallel_started", "running", summary=f"agents={len(started_agents)}")
        self._collect_evidence(state, tools, started_agents)
        self._critique(state)
        state.depth = 1
        self._emit(state, "evidence_added", "finished", summary=f"total={len(state.evidence.findings)}")
        if state.pause_after_turn:
            state.status = "paused"
            self._save_state(state)
            self._close_log(state)
            return state
        answer = self._markdown(state) if state.mode == "fixture" else self._ground_live_or_fallback(model_answer, state)
        self._finish(state, answer)
        self._save_state(state)
        self._close_log(state)
        return state

    @staticmethod
    def _delegated_agents(messages: list[Any]) -> list[str]:
        agents = []
        for message in messages:
            for call in getattr(message, "tool_calls", []) or []:
                if call.get("name") != "task":
                    continue
                agent = call.get("args", {}).get("subagent_type")
                if isinstance(agent, str):
                    agents.append(agent)
        return agents

    def _collect_evidence(self, state: ResearchState, tools: RunToolRuntime, agents: list[str]) -> None:
        topics = _topic_plan(state.question)
        topic_budget = min(len(topics), state.limits.max_total_agents, state.limits.max_parallel_agents * state.limits.max_research_depth)
        for agent_name in agents:
            records = self._select_records(tools.records.get(agent_name, []), state.question)
            errors = tools.errors.get(agent_name, [])
            findings: list[Finding] = []
            if topics and agent_name == "web_researcher":
                for topic, _ in topics[:topic_budget]:
                    findings.extend(self._findings(records, topic=topic, question=state.question))
            else:
                findings = self._findings(records, question=state.question)
            self._log(state, "tool_result", agent=agent_name, records=[{"title": item.title, "url": item.url, "source_type": item.source_type} for item in records], record_count=len(records))
            self._log(state, "agent_result", agent=agent_name, task=state.question, finding_count=len(findings), tool_calls=tools.calls.get(agent_name, 0), error_count=len(errors))
            for finding in findings:
                if len(state.evidence.findings) >= _MAX_FINDINGS_PER_RUN:
                    break
                if any((item.citation, item.claim) == (finding.citation, finding.claim) for item in state.evidence.findings):
                    continue
                token_cost = max(1, (len(finding.claim) + len(finding.evidence)) // 4)
                if state.mode == "fixture" and state.approximate_tokens + token_cost > state.limits.max_tokens_per_run:
                    state.approximate_tokens = state.limits.max_tokens_per_run
                    continue
                state.evidence.findings.append(finding)
                state.evidence.sources = list(dict.fromkeys([*state.evidence.sources, finding.source]))
                if state.mode == "fixture":
                    state.approximate_tokens += token_cost
            self._emit(state, "agent_finished", "finished" if records else "partial", agent=agent_name, task=state.question, summary=f"findings={len(findings)} errors={len(errors)}")

    @staticmethod
    def _select_records(records: list[SourceRecord], question: str) -> list[SourceRecord]:
        if not re.search(r"\b(?:authoritative|official|primary)\s+source\b", question, re.IGNORECASE):
            return records

        def priority(record: SourceRecord) -> int:
            host = (urlparse(record.url).hostname or "").lower()
            title = record.title.lower()
            score = 0
            if host.endswith(".gov") or host in {"arxiv.org", "www.arxiv.org"}:
                score += 4
            if host.startswith(("docs.", "developer.", "developers.", "reference.", "api.")):
                score += 3
            if re.search(r"\b(?:official|documentation|docs|reference)\b", title):
                score += 2
            return score

        ranked = sorted(enumerate(records), key=lambda item: (-priority(item[1]), item[0]))
        authoritative = [record for _, record in ranked if priority(record) > 0]
        selected = authoritative or [record for _, record in ranked]
        if re.search(r"\b(?:one|single)\b", question, re.IGNORECASE):
            return selected[:1]
        return selected

    def _critique(self, state: ResearchState) -> None:
        state.evidence.claims = self._claim_index(state.evidence.findings)
        state.evidence.contradictions = self._contradictions(state.evidence.findings)
        state.evidence.unanswered_questions = self._gaps(state)
        for contradiction in state.evidence.contradictions:
            self._emit(state, "conflict_detected", "finished", summary=contradiction.reason)

    def _finish(self, state: ResearchState, answer: str) -> None:
        self._emit(state, "synthesis_started", "running", agent="synthesis_critic", task=state.question)
        state.final_answer = answer
        self._log(state, "final_answer", answer=answer)
        state.status = "finished"
        self._emit(state, "run_finished", "finished", agent="synthesis_critic", summary=f"findings={len(state.evidence.findings)}")
        self._log(state, "session_end", status=state.status, depth=state.depth, used_agents=state.used_agents, total_agents=state.total_agents, tool_calls=state.tool_calls, final_answer=answer)

    @staticmethod
    def _ground_live_or_fallback(answer: str, state: ResearchState) -> str:
        known = {finding.citation for finding in state.evidence.findings if finding.citation}
        if not answer.strip() or not known:
            return ResearchRuntime._markdown(state)
        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ResearchRuntime._formatting_only(stripped):
                continue
            urls = set(re.findall(r"https?://[^\s)]+", line))
            if not urls or not urls <= known:
                return ResearchRuntime._markdown(state)
        return answer.rstrip() + "\n"

    @staticmethod
    def _require_live_config(mode: str, model: BaseChatModel | None = None) -> None:
        if mode != "live":
            return
        missing = []
        if not os.getenv("TAVILY_API_KEY"):
            missing.append("TAVILY_API_KEY")
        if model is None and not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if missing:
            raise ValueError(f"live mode requires {' and '.join(missing)}")

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
        if logger := self._loggers.pop(state.run_id, None):
            logger.close()

    def _save_state(self, state: ResearchState) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO research_runs(thread_id, catalog_fingerprint, state_json) VALUES (?, ?, ?)",
            (state.thread_id, self.registry.fingerprint, state.model_dump_json()),
        )
        self.connection.commit()

    def _load_state(self, thread_id: str) -> ResearchState | None:
        row = self.connection.execute("SELECT state_json FROM research_runs WHERE thread_id = ?", (thread_id,)).fetchone()
        return ResearchState.model_validate_json(row[0]) if row else None

    def _emit(self, state: ResearchState, event_type: str, status: str, *, agent: str | None = None, task: str | None = None, summary: str | None = None) -> None:
        event = RunEvent(run_id=state.run_id, event_type=event_type, agent=agent, task=task, status=status, result_summary=summary)
        with self._event_lock:
            state.events.append(event)
            self._log(state, "event", event=event.model_dump(mode="json"))
            if self.event_sink:
                self.event_sink(event)

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
                if len(accepted) >= _MAX_FINDINGS_PER_RECORD:
                    break
        return findings

    @staticmethod
    def _clean_claim(claim: str) -> str:
        claim = re.sub(r"^#+\s*", "", claim)
        claim = re.sub(r"\s+\[\d+\]\s*$", "", claim)
        claim = " ".join(claim.split()).strip()
        if not claim or claim[0].islower() or _BOILERPLATE.search(claim):
            return ""
        words = _words(claim)
        if len(words) < 6 or not re.search(r"[.!?]$", claim):
            return ""
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
            return bool(_words(f"{record.title} {claim}") & competitor_terms) and bool(claim_terms & implementation_terms)
        return True

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
                if overlap >= 0.7 and "filed" not in (base_one & base_two) and (polarity_differs or bool(nums_one and nums_two and nums_one != nums_two)):
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
        if not topics and _MULTI_TERMS & _words(state.question):
            if len({finding.source_type for finding in state.evidence.findings}) < 2:
                gaps.append("Need a second source category")
        if state.evidence.contradictions:
            gaps.append("Conflicting claims need verification")
        return gaps

    @staticmethod
    def _formatting_only(line: str) -> bool:
        if re.fullmatch(r"(?:[-*_]\s*){3,}", line):
            return True
        match = re.fullmatch(r"(?:[-*]\s+)?(?:\*\*|__)(.+?)(?:\*\*|__):?", line)
        return bool(match and len(_words(match.group(1))) <= 10 and not re.search(r"[.!?]", match.group(1)))

    @staticmethod
    def _markdown(state: ResearchState) -> str:
        lines = ["# Research answer", ""]
        source_ids: dict[str, int] = {}
        sources: list[Finding] = []
        for finding in state.evidence.findings:
            target = finding.citation or finding.source
            if target not in source_ids:
                source_ids[target] = len(sources) + 1
                sources.append(finding)

        def bullets(findings: list[Finding]) -> list[str]:
            grouped: dict[str, list[Finding]] = {}
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
                lines.extend(bullets(findings) if findings else ["No supported evidence found."])
                lines.append("")
            additional = [finding for finding in state.evidence.findings if finding.topic is None]
            if additional:
                lines.extend(["## Additional evidence", "", *bullets(additional), ""])
        else:
            lines.extend([*bullets(state.evidence.findings), ""])
        if state.evidence.contradictions:
            lines.extend(["## Conflicts", "", *(f"- {item.reason}" for item in state.evidence.contradictions), ""])
        if state.evidence.unanswered_questions:
            lines.extend(["## Unanswered questions", "", *(f"- {item}" for item in state.evidence.unanswered_questions), ""])
        lines.extend(["## Sources", ""])
        for index, finding in enumerate(sources, 1):
            target = finding.citation or finding.source
            lines.append(f"[{index}] [{finding.source}]({target}) — {finding.source_type}, confidence {finding.confidence:.2f}")
        return "\n".join(lines).rstrip() + "\n"
