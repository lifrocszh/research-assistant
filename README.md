# Research Assistant

Python 3.12 prototype for bounded, durable, evidence-grounded research. A
registry builds a Deep Agents supervisor from versioned agent, skill, and tool
manifests. The supervisor sees specialist capabilities through Deep Agents'
delegation tool. Research tools remain private to their owning specialists.

```text
question -> registry -> Deep Agents supervisor -> specialist -> owned tools
                                      |              |
                                      +----------- evidence -> synthesis
```

## Setup

```powershell
uv sync
```

Create `.env` for live runs. Do not commit it.

```dotenv
TAVILY_API_KEY=tvly-...
OPENAI_API_KEY=...
# Optional OpenAI-compatible model settings.
# OPENAI_MODEL=gpt-5.6-terra
# OPENAI_BASE_URL=https://api.openai.com/v1
# TAVILY_API_URL=https://api.tavily.com/search
# SEC_USER_AGENT=YourName Research Assistant you@example.com
```

## Run

Fixture mode is the CLI default. It uses deterministic sources plus a scripted
LangChain tool-calling model. It executes the same registry, supervisor, and
subagent path as live mode without network or model-provider calls.

```powershell
uv run research-assistant run "What does LangGraph provide?"
uv run research-assistant run "Compare Acme sources" --fixtures fixtures.json
```

`fixtures.json` is a JSON array. Each record needs `title`, `url`, `content`,
and `source_type`; `claim` and `confidence` are optional.

Live mode uses the configured OpenAI-compatible model and current public
sources. It requires `OPENAI_API_KEY` and `TAVILY_API_KEY`.

```powershell
uv run --env-file .env research-assistant run "Latest frontier model updates" --mode live
```

Live search uses a Tavily-compatible endpoint. Public arXiv and SEC access use
Python tool adapters. Set `SEC_USER_AGENT` to a real product/contact identity
before filing research.

### Local documents

Markdown and plain text are supported. PDF, DOCX, and spreadsheets remain
unsupported.

```powershell
uv run research-assistant run "Summarize launch risks" --document notes.md
uv run --env-file .env research-assistant run "Compare notes with current reporting" --mode live --document notes.md
```

### Pause and resume

Use a stable thread ID. New Deep Agents checkpoints default to
`.research-assistant/checkpoints.deepagents.sqlite`.

```powershell
uv run research-assistant run "Research dynamic delegation" --thread demo --pause-after-turn
uv run research-assistant resume demo
```

`--pause-after-turn` lets one Deep Agents delegated research turn finish, saves
its validated evidence, and defers synthesis. `resume` synthesizes that saved
evidence; it does not invoke another supervisor turn.

Legacy custom-graph checkpoints in `.research-assistant/checkpoints.sqlite`
are not migrated. Attempting to resume one returns a clear incompatibility
error. Start a new thread in the Deep Agents checkpoint store.

### Output and events

```powershell
uv run research-assistant run "Research Acme margin decline" --jsonl events.jsonl --output answer.md
```

`--jsonl -` streams normalized `RunEvent` JSON to stdout. `--output` writes the
final Markdown answer. Timestamped logs default to `.research-assistant/logs/`.
The CLI and WebSocket UI share the same event contract.

## Architecture

```text
CLI / WebSocket UI
        |
        v
ResearchRuntime (run state, limits, events, evidence, SQLite checkpoints)
        |
        +---- CapabilityRegistry <---- versioned agent / skill / tool manifests
        |
        v
DeepAgentPlatform
        |
        v
Deep Agents supervisor (no application research tools)
        |
        +---- task(name=..., task=...) ---- registered subagent
                                              |-- SKILL.md instructions
                                              `-- explicitly owned tools
                                                       |
                                                       v
                                              fixture / live source adapters
```

Main components:

- `catalog/` is source of truth for agent capabilities, skill instructions,
  tool schemas, permissions, and ownership.
- `registry.py` loads and validates catalog, then exposes capability projections.
  Supervisor sees descriptions and names, not tool implementations.
- `platform.py` builds Deep Agents supervisor and every registered subagent.
  Supervisor receives `tools=[]`; Deep Agents supplies its `task` delegation tool.
  Each subagent receives only tools and skill paths declared by its manifest.
- `tools.py` implements deterministic fixture and live source adapters.
  `RunToolRuntime` in `platform.py` wraps them as run-scoped LangChain tools and
  enforces agent, tool-call, and runtime limits.
- `engine.py` owns run/resume lifecycle, evidence normalization, conflict checks,
  cited synthesis fallback, `RunEvent` emission, and SQLite persistence.
- `models.py` defines manifests, runtime state, evidence, limits, and event schemas.
- `cli.py` and `ui.py` are transport layers over `ResearchRuntime`; neither
  contains orchestration logic.

Fixture and live modes use the same Deep Agents graph. Fixture mode swaps in a
deterministic chat model and offline source adapter; live mode uses the configured
OpenAI-compatible model and network adapters.

## Agent catalog

Packaged source-of-truth lives under `src/research_assistant/catalog/`:

```text
catalog/
  agents/*.yaml
  skills/<skill-id>/skill.yaml
  skills/<skill-id>/SKILL.md
  tools/*.yaml
```

The registry validates SemVer, unique IDs, schema references, required tools,
and permissions at startup. It exposes capability descriptions to the
supervisor, while the runtime keeps Python implementations private.

| Subagent | Responsibility | Skills | Owned tools | Permission / source |
| --- | --- | --- | --- | --- |
| `web_researcher` | Current public-web research and source verification | `web_research` | `web_search`, `fetch_url` | network / web |
| `academic_researcher` | Scholarly-paper discovery and assessment | `academic_research` | `fetch_arxiv` | network / arXiv |
| `company_researcher` | Company research led by primary regulatory evidence | `company_research` | `fetch_sec`, `web_search` | network / SEC filings |
| `document_researcher` | Search user-provided Markdown and text files | `document_research` | `read_document`, `search_document` | document read / local files |
| `synthesis_critic` | Detect evidence gaps and conflicts; produce cited synthesis | `evidence_critique`, `synthesis` | none | none / evidence store |

All subagents inherit configured run model and use `ResearchTask` input plus
`ResearchResult` output schemas. Skills map to packaged `SKILL.md` files:

- `web_research`: search and verify public web evidence.
- `academic_research`: find and assess scholarly sources.
- `company_research`: prioritize filings and primary company evidence.
- `document_research`: find evidence in user-provided local text.
- `evidence_critique`: detect evidence gaps and conflicts; requires no tool.
- `synthesis`: write an evidence-grounded cited answer; requires no tool.

The supervisor owns no application research tools. It selects named subagents
from registry-provided descriptions. Independent delegations may run in
parallel within runtime limits.

Adding an agent or skill requires catalog content, not supervisor changes.
Adding a Python tool requires its adapter plus manifest, still no supervisor
change. Tool manifests accept `execution: mcp`, but MCP execution is explicitly
unsupported in v1 and fails clearly at binding time.

## Web UI

```powershell
uv run --env-file .env uvicorn research_assistant.ui:app --reload
```

Open `http://127.0.0.1:8000`. The UI defaults to live mode. One conversation
surface streams concise planning, delegation, tool, evidence, and synthesis
updates while research runs. Full normalized events stay in a collapsed
`Research activity` trace. Final cited answer appears after evidence validation.
Use the CLI for custom fixtures, documents, pause, and resume workflows.

## Runtime settings

- `OPENAI_MODEL`: OpenAI-compatible chat model. Default `gpt-5.6-terra`.
- `OPENAI_BASE_URL`: compatible API root. Default `https://api.openai.com/v1`.
- `TAVILY_API_URL`: Tavily-compatible search endpoint.
- `SEC_USER_AGENT`: SEC product/contact identity.

Default bounds: 3 parallel agents, 12 total delegations, depth 5, 180 seconds,
8 tool calls per specialist, and 32,000 approximate tokens. Policy enforcement
applies to supervisor and subagent execution and fails closed.

## Research quality

- Tools return compact source records; runtime validates findings while
  specialists return provider-compatible summaries.
- Python validates evidence, merges duplicates, and detects conflicts and gaps.
- Live factual output must cite collected source IDs.
- Fixture synthesis remains deterministic and extractive.
- Live HTML extraction is plain text.
- Token accounting is approximate and can differ from provider billing.
- SEC matching is limited to company names or tickers present in the query.

## Checks

```powershell
uv run python -m compileall -q src tests
uv run pytest
uv lock --check
git diff --check
```
