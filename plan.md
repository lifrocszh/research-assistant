# Registry-first Deep Agents migration

## Goal

Use one Deep Agents supervisor backed by a versioned capability registry.
Supervisor knows agent contracts and descriptions, not tool implementations.
Each specialist owns its skills and tools.

## Phase 1 — Catalog and contracts

- Package YAML manifests for five agents, six skills, and six tools.
- Validate IDs, SemVer, schemas, references, skill-required tools, and
  permissions at startup.
- Keep `ResearchTask`, `ResearchResult`, source records, and existing evidence
  models as typed Pydantic contracts.
- Load each skill from its `SKILL.md`; optional references, scripts, and assets
  stay inside that skill directory.

Acceptance:

- Invalid or unresolved catalog entries fail during registry construction.
- `discover_agents`, `discover_skills`, and `discover_tools` use loaded
  manifests.
- Adding an agent or skill does not require supervisor changes.

## Phase 2 — Deep Agents runtime

- Replace custom planner, dispatcher, and specialist tool loop with
  `create_deep_agent`.
- Build custom subagents from registry projections.
- Give supervisor no application research tools; it delegates through the
  Deep Agents task tool.
- Give each subagent only its manifest-declared tools and skill instructions.
- Keep one delegation level; specialists cannot invoke peers.
- Run independent task calls concurrently up to `max_parallel_agents`.
- Bind `execution: python` manifests to existing fixture/live adapters.
- Reject `execution: mcp` clearly in v1.

Acceptance:

- Supervisor cannot call web, SEC, arXiv, URL, or document tools directly.
- Cross-agent and undeclared tool calls fail closed.
- Company, academic, web, and document research use only owned tools.
- Structured tool records reach evidence validation and synthesis.

## Phase 3 — Durability, policy, compatibility

- Retain SQLite checkpointing through the Deep Agents/LangGraph runtime.
- Default to `.research-assistant/checkpoints.deepagents.sqlite`.
- Store a resolved catalog fingerprint with durable run state.
- Reject legacy custom-graph checkpoints and changed catalog fingerprints;
  do not migrate old state.
- Preserve runtime limits for parallelism, total delegations, depth, elapsed
  time, per-agent tool calls, and approximate tokens.
- Map `--pause-after-turn` to a post-research boundary: finish one Deep Agents
  delegated turn, save validated evidence, and defer synthesis. Resume
  synthesizes saved evidence without another supervisor invocation.

Acceptance:

- Pause/resume works in the new checkpoint store.
- Legacy resume returns a clear incompatibility error.
- Bounds and partial specialist failures preserve successful evidence.

## Phase 4 — Compatibility surfaces

- Preserve `ResearchRuntime.run/resume` and normalized `RunEvent` objects.
- Translate Deep Agents stream events into existing CLI/UI event types.
- Keep first event `run_started` and final event `run_finished`.
- Preserve Markdown citations, JSONL output, WebSocket behavior, logs, and CLI
  flags unless the architecture requires a documented compatibility break.
- Use a scripted LangChain tool-calling model for deterministic fixture mode;
  run the same supervisor/subagent path without network or provider calls.

Acceptance:

- Fixture runs remain deterministic and offline.
- CLI JSONL and UI WebSocket consumers need no orchestration changes.
- Live mocked runs delegate through Deep Agents and produce grounded output.

## Phase 5 — Evaluation

Compare registry-first Deep Agents execution against the prior fixed baseline
on ten versioned cases:

1. Simple factual question.
2. Multi-source research.
3. Company research.
4. Academic research.
5. User document plus web research.
6. Contradictory sources.
7. Multi-hop investigation.
8. Parallel investigation.
9. Evidence-gap follow-up.
10. Report generation.

Measure accuracy, citation/source quality, useful hops, unnecessary calls,
parallelism, conflict detection, unanswered questions, latency, and tokens.

## Required checks

```powershell
uv run python -m compileall -q src tests
uv run pytest
uv lock --check
git diff --check
uv run research-assistant run "What does LangGraph provide?" --mode fixture
```

## Non-goals

- Recursive subagent hierarchies.
- MCP execution in v1.
- Large agent/tool marketplace.
- Long-term memory or filesystem middleware.
- Browser computer use or autonomous external actions.
- Migration of legacy custom-graph checkpoints.
