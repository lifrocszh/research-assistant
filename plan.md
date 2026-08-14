Autonomous Research Agent Prototype

## Objective

Build a Python 3.12 prototype where LangGraph manages durable state and a custom orchestration layer dynamically selects agents, skills, tools, execution order, parallelism, and follow-up research.

Full project specs in `spec.md`.

Use `uv` with `pyproject.toml`.

## Phase 1 — CLI Vertical Slice

### Goal

Prove dynamic research execution end to end without a UI.

### Build

- Capability registry:
  - `AgentSpec`
  - `SkillSpec`
  - `ToolSpec`
- Five fixed agents:
  - Web researcher
  - Academic researcher
  - Company/financial researcher
  - Document researcher
  - Synthesis/critic
- Core tools:
  - `web_search`
  - `fetch_url`
  - `fetch_arxiv`
  - `fetch_sec`
  - `read_document`
  - `search_document`
- Structured models:
  - `Finding`
  - `EvidenceStore`
  - `ResearchState`
  - `Decision`
  - `Action`
  - `AgentResult`
  - `RunEvent`
- Dynamic orchestrator.
- Sequential and parallel dispatcher.
- Evidence aggregation.
- Gap and conflict detection.
- Follow-up delegation.
- Markdown final answers with citations.
- JSONL event output.

### LangGraph design

Use one dynamic turn:

```text
decide → dispatch → collect evidence → critique → continue or finish
```

Checkpoint after every turn with local SQLite.

Support:

```text
max_parallel_agents = 3
max_total_agents = 12
max_research_depth = 5
max_runtime_seconds = 180
max_tool_calls_per_agent = 8
max_tokens_per_run = 32000
```

Support resume by thread ID and optional pause after a turn.

### Research adapters

- Tavily-compatible live search adapter.
- Deterministic fixture search adapter.
- Bounded HTTP URL fetching.
- Public arXiv adapter.
- Public SEC adapter.
- Markdown/text document loading and search.

Defer PDF, DOCX, and spreadsheet extraction.

### CLI

Support:

- New research run.
- Fixture or live mode.
- JSONL event streaming.
- Final answer output.
- Thread resume.
- Optional pause after turn.

## Phase 2 — WebSocket UI

### Goal

Visualize the existing event stream after CLI behavior is stable.

### Build

Add minimal FastAPI/WebSocket layer.

Display:

```text
planning
agent started/finished
tool started/finished
parallel execution
evidence added
conflict detected
follow-up started
synthesis
run finished
```

Reuse `RunEvent`. Add no UI-specific orchestration logic.

## Phase 3 — Evaluation

### Goal

Compare dynamic orchestration against a fixed workflow.

### Test cases

Create versioned YAML cases covering:

1. Simple factual question.
2. Multi-source research.
3. Company research.
4. Academic research.
5. User document plus web research.
6. Contradictory sources.
7. Multi-hop investigation.
8. Parallel research.
9. Evidence-gap follow-up.
10. Report generation.

Each case defines:

- Question.
- Fixture sources.
- Expected capability categories.
- Expected follow-up or conflict behavior.
- Evaluation notes.

### Baseline

Implement:

```text
retrieve → research → synthesize
```

### Metrics

Compare both systems on:

- Accuracy.
- Citation quality.
- Source quality.
- Useful research hops.
- Unnecessary tool calls.
- Parallelization quality.
- Conflict detection.
- Unanswered questions.
- Latency.
- Token usage.

Store execution traces as JSONL.

## Phase 4 — Handoff

### Documentation

Create:

- `README.md`
  - Setup.
  - Environment variables.
  - CLI commands.
  - Fixture/live usage.
- `HANDOFF.md`
  - Verified architecture.
  - Current phase.
  - Passing checks.
  - Known limitations.
  - Design decisions.
  - Next-agent tasks.
- Evaluation report with fixed-vs-dynamic results.

### Handoff acceptance

- Fixture run passes end to end.
- SQLite resume works.
- Missing live credentials fail clearly.
- WebSocket event contract passes.
- Evaluation traces are generated.
- Tests pass.
- `python -m compileall` passes.
- `pytest` passes.
- `git diff --check` passes.
- No unexplained generated artifacts remain.

## Testing Strategy

Use focused pytest tests plus one CLI smoke test.

Cover:

- Registry discovery.
- Decision validation.
- Sequential dispatch.
- Parallel dispatch.
- Partial parallel failure.
- Runtime limits.
- Evidence conflict detection.
- Critic gap detection.
- SQLite resume.
- Fixture determinism.
- HTTP timeout and error handling.
- Missing credentials.
- JSONL event validity.
- Dynamic path variation.
- Fixed-vs-dynamic evaluation.

## Explicit Non-Goals

Defer:

- Large agent marketplace.
- Hundreds of tools.
- Complex memory architecture.
- Browser computer-use.
- Production authentication.
- External production database.
- Reinforcement learning.
- Sophisticated frontend.
- Autonomous external actions.

## Success Criterion

The prototype succeeds when the same runtime produces meaningfully different execution paths for different questions, while remaining bounded, observable, recoverable, and evidence-grounded.
