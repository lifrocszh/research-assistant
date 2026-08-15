from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import Field

from .models import AgentSpec, RuntimeLimits, SourceRecords
from .registry import CapabilityRegistry
from .tools import ResearchTools, SourceRecord, ToolError

ToolEvent = Callable[[str, str, str, str | None], None]
_HIDDEN_HARNESS_TOOLS = {"write_todos", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}


def _register_profiles() -> None:
    profile = HarnessProfile(
        excluded_tools=frozenset(_HIDDEN_HARNESS_TOOLS),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    register_harness_profile("openai", profile)
    register_harness_profile("research_fixture", profile)
    register_harness_profile("fixturechatmodel", profile)


_register_profiles()


class FixtureChatModel(BaseChatModel):
    """Offline tool-calling model. It exercises the same Deep Agents graph."""

    registry: CapabilityRegistry
    documents: list[str]
    selected_agents: list[str] = Field(default_factory=list)

    def __init__(self, registry: CapabilityRegistry, documents: list[str]) -> None:
        super().__init__(registry=registry, documents=documents)

    @property
    def _llm_type(self) -> str:
        return "research_fixture"

    def bind_tools(self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any) -> Runnable:
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools") or []
        names = {
            item.get("function", {}).get("name") if isinstance(item, dict) else getattr(item, "name", None)
            for item in tools
        }
        names.discard(None)
        if "task" in names:
            message = self._supervisor(messages)
        else:
            message = self._specialist(messages, names & self.registry.tools.keys())
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _supervisor(self, messages: list[BaseMessage]) -> AIMessage:
        task_results = [item for item in messages if isinstance(item, ToolMessage) and item.name == "task"]
        if task_results:
            return AIMessage(content="Research delegation complete.")
        question = next((str(item.content) for item in reversed(messages) if isinstance(item, HumanMessage)), "")
        agents = [spec.id for spec in self.registry.discover_agents(question) if spec.id != "synthesis_critic"]
        if self.documents:
            agents.insert(0, "document_researcher")
        if not agents:
            agents = ["web_researcher"]
        multi = bool({"compare", "multiple", "sources", "evidence", "research", "report"} & set(question.lower().split()))
        agents = list(dict.fromkeys(agents))[: 3 if multi else 1]
        self.selected_agents = agents
        calls = [
            {
                "name": "task",
                "args": {"description": question, "subagent_type": agent},
                "id": f"fixture-{uuid4()}",
                "type": "tool_call",
            }
            for agent in agents
        ]
        return AIMessage(content="", tool_calls=calls)

    @staticmethod
    def _specialist(messages: list[BaseMessage], names: set[str]) -> AIMessage:
        results = [item for item in messages if isinstance(item, ToolMessage) and item.name in names]
        if results:
            return AIMessage(content="\n".join(str(item.content) for item in results))
        question = next((str(item.content) for item in reversed(messages) if isinstance(item, HumanMessage)), "")
        preferred = ["search_document", "fetch_sec", "fetch_arxiv", "web_search", "read_document", "fetch_url"]
        chosen = next((name for name in preferred if name in names), next(iter(names), None))
        if chosen is None:
            return AIMessage(content=json.dumps({"records": []}))
        key = "url" if chosen == "fetch_url" else "path" if chosen == "read_document" else "query"
        value = question
        if chosen == "read_document":
            value = "all-provided-documents"
        return AIMessage(
            content="",
            tool_calls=[{"name": chosen, "args": {key: value}, "id": f"fixture-{uuid4()}", "type": "tool_call"}],
        )


class RunToolRuntime:
    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        mode: str,
        documents: list[str],
        fixture_path: str | None,
        limits: RuntimeLimits,
        started_at: datetime,
        event: ToolEvent,
    ) -> None:
        self.registry = registry
        self.documents = documents
        self.limits = limits
        self.started_at = started_at
        self.event = event
        self.tools = ResearchTools(mode, fixture_path, timeout=min(10, limits.max_runtime_seconds))
        self.records: dict[str, list[SourceRecord]] = {}
        self.errors: dict[str, list[str]] = {}
        self.calls: dict[str, int] = {}
        self.started_agents: list[str] = []
        self._lock = threading.Lock()
        self._parallel = threading.BoundedSemaphore(limits.max_parallel_agents)

    def close(self) -> None:
        self.tools.close()

    def build(self, agent: AgentSpec) -> list[BaseTool]:
        return [self._build_one(agent.id, tool_id) for tool_id in agent.tools]

    def _build_one(self, agent_id: str, tool_id: str) -> BaseTool:
        manifest = self.registry.tools[tool_id]
        if manifest.execution_type != "python":
            raise NotImplementedError(f"MCP tool execution is not supported in v1: {tool_id}")

        def invoke(**values: str) -> str:
            query = next(iter(values.values()), "")
            with self._lock:
                if agent_id not in self.started_agents:
                    if len(self.started_agents) >= self.limits.max_total_agents:
                        return json.dumps({"records": [], "error": "total agent limit reached"})
                    self.started_agents.append(agent_id)
                    self.event("agent_started", agent_id, query, None)
                used = self.calls.get(agent_id, 0)
                if used >= self.limits.max_tool_calls_per_agent:
                    return json.dumps({"records": [], "error": "agent tool-call limit reached"})
                if (datetime.now(UTC) - self.started_at).total_seconds() >= self.limits.max_runtime_seconds:
                    return json.dumps({"records": [], "error": "runtime limit reached"})
                self.calls[agent_id] = used + 1
            self.event("tool_started", agent_id, tool_id, None)
            self._parallel.acquire()
            try:
                records = self.tools.invoke(tool_id, query, self.documents)
            except ToolError as exc:
                with self._lock:
                    self.errors.setdefault(agent_id, []).append(str(exc))
                self.event("tool_finished", agent_id, tool_id, str(exc))
                return json.dumps({"records": [], "error": str(exc)})
            finally:
                self._parallel.release()
            with self._lock:
                self.records.setdefault(agent_id, []).extend(records)
            self.event("tool_finished", agent_id, tool_id, f"records={len(records)}")
            return SourceRecords(records=[record.__dict__ for record in records]).model_dump_json()

        return StructuredTool.from_function(
            func=invoke,
            name=tool_id,
            description=manifest.description,
            args_schema=self.registry.schema(manifest.input_schema),
        )


class DeepAgentPlatform:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    @staticmethod
    def live_model(limits: RuntimeLimits) -> ChatOpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("live mode requires OPENAI_API_KEY")
        return ChatOpenAI(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-5.6-terra"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=0,
            timeout=limits.max_runtime_seconds,
            max_tokens=min(16_384, limits.max_tokens_per_run),
            use_responses_api=False,
        )

    def build(
        self,
        *,
        model: BaseChatModel,
        tool_runtime: RunToolRuntime,
        checkpointer: Any,
    ) -> Any:
        subagents = []
        for spec in self.registry.agents.values():
            subagent = {
                "name": spec.id,
                "description": spec.description,
                "system_prompt": spec.system_prompt,
                "model": model if spec.model == "inherit" else spec.model,
                "tools": tool_runtime.build(spec),
                "skills": [f"/skills/{skill_id}/" for skill_id in spec.skills],
            }
            subagents.append(subagent)
        capabilities = [item.model_dump() for item in self.registry.capabilities()]
        prompt = (
            "Coordinate research using only the task tool and registered subagents. "
            "Delegate independent tasks together. Never perform research directly. "
            "Synthesize a concise Markdown answer from returned findings. Every factual claim must cite a returned source URL. "
            f"Bounds: at most {tool_runtime.limits.max_total_agents} delegated tasks, "
            f"{tool_runtime.limits.max_parallel_agents} concurrent tasks, and {tool_runtime.limits.max_research_depth} research rounds. "
            f"Registered capabilities: {json.dumps(capabilities, sort_keys=True)}"
        )
        return create_deep_agent(
            model=model,
            tools=[],
            system_prompt=prompt,
            subagents=subagents,
            backend=FilesystemBackend(root_dir=str(self.registry.root)),
            checkpointer=checkpointer,
            name="research_orchestrator",
        )
