from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .models import (
    AgentCapability,
    AgentSpec,
    DocumentInput,
    DocumentSearchInput,
    ResearchResult,
    ResearchTask,
    SearchInput,
    SkillSpec,
    SourceRecords,
    ToolSpec,
    UrlInput,
)

Spec = TypeVar("Spec", AgentSpec, SkillSpec, ToolSpec)
SCHEMAS: dict[str, type[BaseModel]] = {
    item.__name__: item
    for item in (ResearchTask, ResearchResult, SearchInput, UrlInput, DocumentInput, DocumentSearchInput, SourceRecords)
}
PERMISSIONS = {"network", "document_read"}


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {
        "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    }


class CatalogError(ValueError):
    pass


class CapabilityRegistry:
    """Validated catalog. Orchestrators consume capability projections only."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).with_name("catalog")
        self.agents = self._load_yaml_dir("agents", "agent", AgentSpec)
        self.tools = self._load_yaml_dir("tools", "tool", ToolSpec)
        self.skills = self._load_skills()
        self._validate_links()
        self.fingerprint = self._fingerprint()

    def capability(self, agent_id: str) -> AgentCapability:
        spec = self.agents[agent_id]
        return AgentCapability.model_validate(spec.model_dump(include=set(AgentCapability.model_fields)))

    def capabilities(self) -> list[AgentCapability]:
        return [self.capability(agent_id) for agent_id in sorted(self.agents)]

    def discover_agents(self, query: str) -> list[AgentSpec]:
        query_terms = _terms(query)
        return [
            spec
            for spec in self._discover(query, self.agents.values())
            if spec.id != "company_researcher" or query_terms & set(spec.keywords)
        ]

    def discover_skills(self, query: str) -> list[SkillSpec]:
        return self._discover(query, self.skills.values())

    def discover_tools(self, query: str) -> list[ToolSpec]:
        return self._discover(query, self.tools.values())

    def skill_path(self, skill_id: str) -> Path:
        return self.root / "skills" / skill_id

    def schema(self, schema_id: str) -> type[BaseModel]:
        return SCHEMAS[schema_id]

    @staticmethod
    def _discover(query: str, specs: Iterable[Spec]) -> list[Spec]:
        wanted = _terms(query)
        ranked = []
        for spec in specs:
            text = " ".join([spec.id, spec.description, *spec.keywords])
            score = len(wanted & _terms(text))
            if score:
                ranked.append((score, spec.id, spec))
        return [spec for _, _, spec in sorted(ranked, reverse=True)]

    def _load_yaml_dir(self, directory: str, wrapper: str, model: type[Spec]) -> dict[str, Spec]:
        target = self.root / directory
        if not target.is_dir():
            raise CatalogError(f"catalog directory not found: {target}")
        loaded: dict[str, Spec] = {}
        for path in sorted(target.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                data = raw[wrapper]
                if model is ToolSpec and "execution" in data:
                    data["execution_type"] = data.pop("execution")
                item = model.model_validate(data)
            except (OSError, KeyError, TypeError, ValidationError, yaml.YAMLError) as exc:
                raise CatalogError(f"invalid catalog manifest: {path}") from exc
            if item.id in loaded:
                raise CatalogError(f"duplicate {wrapper} id: {item.id}")
            if path.stem != item.id:
                raise CatalogError(f"{wrapper} filename must match id: {path}")
            loaded[item.id] = item
        if not loaded:
            raise CatalogError(f"no {wrapper} manifests found in {target}")
        return loaded

    def _load_skills(self) -> dict[str, SkillSpec]:
        loaded: dict[str, SkillSpec] = {}
        for path in sorted((self.root / "skills").glob("*/skill.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                item = SkillSpec.model_validate(raw["skill"])
            except (OSError, KeyError, TypeError, ValidationError, yaml.YAMLError) as exc:
                raise CatalogError(f"invalid catalog manifest: {path}") from exc
            if item.id in loaded:
                raise CatalogError(f"duplicate skill id: {item.id}")
            if path.parent.name != item.id:
                raise CatalogError(f"skill directory must match id: {path.parent}")
            entrypoint = path.parent / item.entrypoint
            if not entrypoint.is_file():
                raise CatalogError(f"skill entrypoint not found: {entrypoint}")
            loaded[item.id] = item
        if not loaded:
            raise CatalogError(f"no skill manifests found in {self.root / 'skills'}")
        return loaded

    def _validate_links(self) -> None:
        for tool in self.tools.values():
            for schema_id in (tool.input_schema, tool.output_schema):
                if schema_id not in SCHEMAS:
                    raise CatalogError(f"tool {tool.id} references unknown schema: {schema_id}")
        for agent in self.agents.values():
            for schema_id in (agent.input_schema, agent.output_schema):
                if schema_id not in SCHEMAS:
                    raise CatalogError(f"agent {agent.id} references unknown schema: {schema_id}")
            if unknown := set(agent.skills) - self.skills.keys():
                raise CatalogError(f"agent {agent.id} references unknown skills: {sorted(unknown)}")
            if unknown := set(agent.tools) - self.tools.keys():
                raise CatalogError(f"agent {agent.id} references unknown tools: {sorted(unknown)}")
            required = {tool for skill in agent.skills for tool in self.skills[skill].required_tools}
            if missing := required - set(agent.tools):
                raise CatalogError(f"agent {agent.id} lacks tools required by skills: {sorted(missing)}")
            grants = set(agent.permissions)
            if unknown := grants - PERMISSIONS:
                raise CatalogError(f"agent {agent.id} has unknown permissions: {sorted(unknown)}")
            for tool_id in agent.tools:
                if unknown := set(self.tools[tool_id].permissions) - PERMISSIONS:
                    raise CatalogError(f"tool {tool_id} has unknown permissions: {sorted(unknown)}")
                if missing := set(self.tools[tool_id].permissions) - grants:
                    raise CatalogError(f"agent {agent.id} lacks permissions for {tool_id}: {sorted(missing)}")

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
            digest.update(path.relative_to(self.root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()
