# Registry-first Deep Agents research assistant

## 1. Objective

Build a bounded research runtime where a Deep Agents supervisor delegates to
fixed specialist agents discovered from a versioned catalog. Specialists own
procedural skills and executable tools. Orchestrator consumes capability
contracts only.

Core rule:

> Add an agent, skill, or tool manifest without editing orchestrator code.

A new Python tool still needs one runtime adapter. MCP declarations are
schema-valid but execution remains unsupported in v1.

## 2. Architecture

```text
Agent Platform
  Agents       Skills       Tools
      \          |          /
       Versioned YAML catalog
                |
         Capability Registry
                |
      Deep Agents Supervisor
        task delegation only
                |
  +-------------+-------------+-------------+
  |             |             |             |
 Web         Academic      Company       Document
 agent        agent         agent          agent
  |             |             |             |
owned tools   owned tool    owned tools   owned tools
  +-------------+-------------+-------------+
                |
      evidence validation
                |
       Synthesis/Critic
                |
          cited answer
```

Responsibilities:

- Catalog defines identities, contracts, permissions, and ownership.
- Registry loads, validates, discovers, and projects capabilities.
- Supervisor selects named specialists and composes results.
- Deep Agents provides delegation and subagent execution contexts.
- Tool runtime binds declared tools to fixture/live Python adapters.
- Existing evidence code validates claims, citations, gaps, and conflicts.
- LangGraph/SQLite provides durable thread state beneath Deep Agents.
- CLI and UI consume normalized `RunEvent` objects.

## 3. Catalog

Catalog is packaged under `src/research_assistant/catalog/`.

```text
catalog/
  agents/<agent-id>.yaml
  skills/<skill-id>/skill.yaml
  skills/<skill-id>/SKILL.md
  tools/<tool-id>.yaml
```

### Agent schema

```text
Agent
  id, version, description
  model, system_prompt
  skills[], tools[]
  input_schema, output_schema
  permissions[]
  sources[], keywords[]
```

`model: inherit` uses the runtime-selected model. Agent IDs and versions are
stable durable-state inputs.

### Skill schema

```text
Skill
  id, version, description
  entrypoint
  required_tools[]
  keywords[]
```

`entrypoint` is `SKILL.md`. Optional `references/`, `scripts/`, and `assets/`
belong to the skill and are loaded only when needed.

### Tool schema

```text
Tool
  id, version, description
  execution
  input_schema, output_schema
  permissions[]
  source_types[], keywords[]
```

Allowed execution declarations are `python` and `mcp`. Only `python` binds in
v1. Attempting to bind `mcp` fails with an explicit unsupported-execution
error.

### Registered contracts

- `ResearchTask`
- `ResearchResult`
- `SearchInput`
- `UrlInput`
- `DocumentInput`
- `DocumentSearchInput`
- `SourceRecords`

Schema names resolve through an explicit Pydantic contract map. Arbitrary
imports from manifests are forbidden.

### Startup validation

Registry rejects:

- malformed YAML or wrong top-level wrapper;
- duplicate or mismatched file/manifest IDs;
- invalid SemVer;
- unknown agent, skill, tool, or schema references;
- missing skill entrypoints;
- a skill-required tool absent from its agent;
- tool permissions absent from the owning agent;
- unsupported permission names.

Registry exposes descriptions and contracts. It never exposes Python callables
to supervisor prompts.

## 4. Agents and ownership

### `web_researcher`

- Skill: `web_research`
- Tools: `web_search`, `fetch_url`
- Permission: `network`
- Purpose: current public-web evidence.

### `academic_researcher`

- Skill: `academic_research`
- Tool: `fetch_arxiv`
- Permission: `network`
- Purpose: scholarly sources and reported research.

### `company_researcher`

- Skill: `company_research`
- Tools: `fetch_sec`, `web_search`
- Permission: `network`
- Purpose: company, financial, and regulatory evidence.

### `document_researcher`

- Skill: `document_research`
- Tools: `read_document`, `search_document`
- Permission: `document_read`
- Purpose: supplied Markdown/plain-text evidence.

### `synthesis_critic`

- Skills: `evidence_critique`, `synthesis`
- Tools: none
- Permissions: none
- Purpose: check support, detect gaps/conflicts, and write cited output.

Custom subagents receive explicit tool lists. They do not inherit supervisor
tools. Supervisor has no application research tools and cannot call specialist
tools by name.

## 5. Tool runtime

Python adapters preserve existing behavior:

```text
web_search(query)
fetch_url(url)
fetch_arxiv(query)
fetch_sec(query)
read_document(paths)
search_document(query, paths)
```

Rules:

- Fixture mode uses deterministic source records and no network.
- Live web search uses a Tavily-compatible endpoint.
- URL fetch accepts absolute HTTP(S) URLs and bounded responses.
- arXiv and SEC use public endpoints with timeouts.
- SEC requests identify the application through `SEC_USER_AGENT`.
- Documents are bounded Markdown or plain text.
- PDF, DOCX, and spreadsheet extraction is deferred.
- Tool output is compact structured source records, not unbounded page text.
- Existing query cleanup, extraction, validation, and timeout behavior remains.

## 6. Deep Agents orchestration

Runtime constructs one supervisor with registry-generated custom subagent
specifications. The supervisor prompt contains agent ID, description, and
input/output contract. Tool implementation details stay outside its context.

Supervisor may:

- delegate one task to one specialist;
- issue independent task calls concurrently;
- inspect specialist summaries while collecting validated structured tool records;
- request bounded follow-up research;
- delegate evidence review/synthesis.

Supervisor may not:

- invoke research tools directly;
- invent agent or tool IDs;
- grant a subagent undeclared tools or permissions;
- create recursive subagent hierarchies;
- bypass runtime policy.

One delegation level is sufficient for v1. Deep Agents filesystem, long-term
memory, recursive delegation, and human-approval features are deferred.

## 7. Fixture and live models

Fixture mode uses a scripted LangChain tool-calling model. It drives the same
Deep Agents supervisor/subagent path, returns deterministic delegations, makes
no provider calls, and preserves fixture-specific citations.

Live mode uses a LangChain-compatible OpenAI chat model selected through
`OPENAI_MODEL` and `OPENAI_BASE_URL`. Model output never authorizes undeclared
tools. Runtime validates delegation targets, tool inputs, results, and limits.

## 8. Evidence and output

Specialists return `ResearchResult` containing findings and errors. Accepted
findings retain:

```text
claim, source, source_type, evidence, confidence,
citation, topic, entities, time_period
```

Post-processing preserves:

- incomplete/noisy claim rejection;
- topic and implementation relevance checks;
- duplicate claim/source merging;
- conflict and evidence-gap detection;
- source IDs and Markdown citation formatting;
- grounded live synthesis validation;
- deterministic fixture Markdown.

Partial specialist failure does not discard successful independent findings.

## 9. Runtime policy

Defaults:

```text
max_parallel_agents = 3
max_total_agents = 12
max_research_depth = 5
max_runtime_seconds = 180
max_tool_calls_per_agent = 8
max_tokens_per_run = 32000
```

Policy covers supervisor task calls and specialist tool calls. Exhausted or
invalid operations fail closed and emit normalized errors.

## 10. Durability

New checkpoint default:

```text
.research-assistant/checkpoints.deepagents.sqlite
```

Durable state includes a persisted run-state record plus a fingerprint of
resolved manifest IDs and versions. Deep Agents checkpoints share the same
SQLite file; resume requires the run-state record and matching catalog
fingerprint.

Legacy `.research-assistant/checkpoints.sqlite` custom-graph state is not
migrated. Resume returns a clear incompatibility error. `--pause-after-turn`
lets the current Deep Agents delegated research turn complete, persists its
validated evidence, and defers synthesis. Resume synthesizes saved evidence;
it does not continue or invoke another supervisor turn.

## 11. Events and interfaces

`ResearchRuntime.run/resume`, CLI flags, Markdown output, JSONL, and WebSocket
consumers remain compatibility surfaces. Deep Agents stream data maps to:

```text
run_started
planning
parallel_started
agent_started
tool_started
tool_finished
agent_finished
evidence_added
conflict_detected
followup_started
synthesis_started
run_finished
error
```

`run_started` is first. Successful runs end with `run_finished`, followed by
the UI's final `answer` message. UI contains no orchestration logic.

## 12. Testing

Required offline tests cover:

- valid and invalid catalog loading;
- unresolved references, permissions, schemas, and missing `SKILL.md`;
- capability discovery from manifests;
- supervisor isolation from research tools;
- specialist tool and skill ownership;
- adding an agent manifest without supervisor edits;
- deterministic fixture execution through Deep Agents;
- mocked live delegation and structured tool results;
- parallelism, limits, timeouts, and partial failures;
- evidence validation and grounded citations;
- normalized CLI/UI event ordering;
- SQLite pause/resume and catalog fingerprint checks;
- clear legacy checkpoint rejection.

Unit tests never call live services. Use `tmp_path`, `monkeypatch`,
`httpx.MockTransport`, and FastAPI `TestClient`.

## 13. Non-goals

- MCP execution in v1.
- Recursive agent trees.
- Large marketplace or remote catalog.
- Arbitrary manifest imports.
- Complex memory architecture.
- Production auth or external database.
- Browser computer use.
- Autonomous external actions.
- Legacy checkpoint migration.
