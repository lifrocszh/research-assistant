from __future__ import annotations

import shutil
from datetime import UTC, datetime

import pytest

from research_assistant.models import RuntimeLimits
from research_assistant.platform import DeepAgentPlatform, FixtureChatModel, RunToolRuntime
from research_assistant.registry import CatalogError, CapabilityRegistry


def test_catalog_is_valid_and_orchestrator_projection_hides_implementation() -> None:
    registry = CapabilityRegistry()
    assert (len(registry.agents), len(registry.skills), len(registry.tools)) == (5, 6, 6)
    capability = registry.capability("web_researcher")
    assert capability.tools == ["web_search", "fetch_url"]
    assert not hasattr(capability, "system_prompt")
    assert not hasattr(capability, "model")


def test_catalog_rejects_skill_without_required_tool(tmp_path) -> None:
    source = CapabilityRegistry().root
    target = tmp_path / "catalog"
    shutil.copytree(source, target)
    manifest = target / "agents" / "web_researcher.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("    - fetch_url\n", "", 1), encoding="utf-8")
    with pytest.raises(CatalogError, match="lacks tools required by skills"):
        CapabilityRegistry(target)


def test_new_agent_manifest_is_exposed_without_orchestrator_changes(tmp_path, monkeypatch) -> None:
    target = tmp_path / "catalog"
    shutil.copytree(CapabilityRegistry().root, target)
    (target / "agents" / "reviewer.yaml").write_text(
        """agent:
  id: reviewer
  version: 1.0.0
  description: Review grounded research results
  model: inherit
  system_prompt: Review evidence and report gaps.
  skills: [synthesis]
  tools: []
  input_schema: ResearchTask
  output_schema: ResearchResult
  permissions: []
  sources: [evidence]
  keywords: [review]
""",
        encoding="utf-8",
    )
    registry = CapabilityRegistry(target)
    captured = {}
    monkeypatch.setattr("research_assistant.platform.create_deep_agent", lambda **kwargs: captured.update(kwargs))
    runtime = RunToolRuntime(
        registry,
        mode="fixture",
        documents=[],
        fixture_path=None,
        limits=RuntimeLimits(),
        started_at=datetime.now(UTC),
        event=lambda *_: None,
    )
    try:
        DeepAgentPlatform(registry).build(
            model=FixtureChatModel(registry, []),
            tool_runtime=runtime,
            checkpointer=object(),
        )
    finally:
        runtime.close()

    assert "reviewer" in {item["name"] for item in captured["subagents"]}


def test_deep_agent_assembly_gives_tools_only_to_owning_subagents(monkeypatch) -> None:
    registry = CapabilityRegistry()
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("research_assistant.platform.create_deep_agent", fake_create_deep_agent)
    runtime = RunToolRuntime(
        registry,
        mode="fixture",
        documents=[],
        fixture_path=None,
        limits=RuntimeLimits(),
        started_at=datetime.now(UTC),
        event=lambda *_: None,
    )
    try:
        result = DeepAgentPlatform(registry).build(
            model=FixtureChatModel(registry, []),
            tool_runtime=runtime,
            checkpointer=object(),
        )
    finally:
        runtime.close()

    assert result is not None
    assert captured["tools"] == []
    subagents = {item["name"]: item for item in captured["subagents"]}
    assert {tool.name for tool in subagents["web_researcher"]["tools"]} == {"web_search", "fetch_url"}
    assert {tool.name for tool in subagents["academic_researcher"]["tools"]} == {"fetch_arxiv"}
    assert subagents["synthesis_critic"]["tools"] == []
    assert subagents["web_researcher"]["skills"] == ["/skills/web_research/"]
    assert all("response_format" not in item for item in subagents.values())
    assert subagents["web_researcher"]["system_prompt"] == registry.agents["web_researcher"].system_prompt
