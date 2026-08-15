# Research Assistant

Python 3.12 prototype for bounded, durable, evidence-grounded research. LangGraph checkpoints generic research turns; a capability registry and custom orchestrator choose specialist agents and execution order at runtime.

## Setup

```powershell
uv sync
```

Create `.env` for live runs. Do not commit it.

```dotenv
TAVILY_API_KEY=tvly-...
OPENAI_API_KEY=...
# Optional model and OpenAI-compatible API root.
# OPENAI_MODEL=gpt-5.6-terra
# OPENAI_BASE_URL=https://api.openai.com/v1
# Optional: use a compatible search endpoint instead of Tavily.
# TAVILY_API_URL=https://api.tavily.com/search
# Recommended for SEC requests: real product/contact identity.
# SEC_USER_AGENT=YourName Research Assistant you@example.com
```

## Run types

### Fixture: offline, deterministic

Use this for development, tests, and repeatable demonstrations. It uses the built-in sample sources; it cannot answer current-news questions.

```powershell
# --mode fixture is the CLI default.
uv run research-assistant run "What does LangGraph provide?"
```

Replace built-in records with a JSON fixture file:

```powershell
uv run research-assistant run "Compare Acme sources" --fixtures fixtures.json
```

`fixtures.json` must be a JSON array. Each record needs `title`, `url`, `content`, and `source_type`; `claim` and `confidence` are optional.

### Live: current public-web research

Use this for questions such as current news, recent model releases, or current company information. This is the adaptive path: model plans, reviews evidence, chooses follow-up work, and synthesizes. It requires both `TAVILY_API_KEY` for retrieval and `OPENAI_API_KEY` for planning, follow-up decisions, and synthesis.

```powershell
uv run --env-file .env research-assistant run "What are the latest AI model updates from frontier labs?" --mode live
```

Live search uses Tavily-compatible web search. Model calls use `OPENAI_MODEL` (default `gpt-5.6-terra`) through `OPENAI_BASE_URL` (default `https://api.openai.com/v1`). OpenAI-compatible providers must support `/chat/completions` and JSON response mode. Public arXiv and SEC lookups remain Python tools selected by the model. Set `SEC_USER_AGENT` to a real product/contact identity before company or filing research.

Live runs incur OpenAI-compatible provider and Tavily usage costs. Runtime bounds cap agents, depth, tool calls, elapsed time, and approximate stored/model tokens; provider billing may count input tokens differently.

### Local documents: add to either mode

Documents provide extra evidence. The selected sources still follow the chosen mode. Supported files are Markdown and plain text.

```powershell
# Offline document research
uv run research-assistant run "Summarize the launch risks" --document notes.md

# Document plus current web research
uv run --env-file .env research-assistant run "Compare these notes with current reporting" --mode live --document notes.md
```

PDF, DOCX, and spreadsheets are not supported yet.

### Pause and resume

Use a stable thread ID. The checkpoint database defaults to `.research-assistant/checkpoints.sqlite`.

```powershell
uv run research-assistant run "Research dynamic delegation" --thread demo --pause-after-turn
uv run research-assistant resume demo
```

### Save output and execution events

```powershell
uv run research-assistant run "Research Acme's margin decline" --jsonl events.jsonl --output answer.md
```

`--jsonl -` writes events to stdout. `--output` writes the final Markdown answer. Each run also writes a detailed timestamped log to `.research-assistant/logs/`; use `--log-dir PATH` to change that directory. Resuming a paused thread appends to its original log. Logs include the question, decisions, agent/task delegation, subtasks, tool inputs/results, findings, errors, and final answer. `--document` may be repeated. Run `research-assistant run --help` for bounds such as `--max-parallel-agents` and `--max-research-depth`.

## CLI

The CLI defaults to fixture mode: fully offline, deterministic, and zero LLM calls. Use `--mode live` for adaptive model-guided research.

```powershell
# Fixture
uv run research-assistant run "What does LangGraph provide?"

# Live
uv run --env-file .env research-assistant run "Current research question" --mode live
```

## Web UI

```powershell
uv run --env-file .env uvicorn research_assistant.ui:app --reload
```

Open `http://127.0.0.1:8000`. Choose **Live: adaptive model research** for model-guided research, or **Fixture: offline, no LLM calls** for deterministic local runs. The page defaults to live mode and presents an orchestration timeline beside the rendered final answer; both panes scroll independently. Live mode needs `TAVILY_API_KEY` and `OPENAI_API_KEY`; without them, the run reports a clear error. Use the CLI for custom fixtures, documents, and resume runs.

## Runtime settings

Optional environment variables:

- `OPENAI_MODEL`: OpenAI-compatible chat model. Defaults to `gpt-5.6-terra`.
- `OPENAI_BASE_URL`: OpenAI-compatible API root. Defaults to `https://api.openai.com/v1`.
- `TAVILY_API_URL`: Tavily-compatible search endpoint. Defaults to Tavily.
- `SEC_USER_AGENT`: SEC-compliant product/contact user agent.

Checkpoints default to `.research-assistant/checkpoints.sqlite`. Limits have CLI flags and default to 3 parallel agents, 12 total agents, depth 5, 180 seconds, 8 tool calls per agent, and 32,000 approximate tokens.

## Research quality behavior

- Fixture mode uses deterministic capability matching and extractive Markdown. It makes no LLM calls and stays fully offline.
- Live mode uses the configured model to plan specialist/tool actions, decide follow-up research after each evidence turn, and synthesize the answer.
- Python executes every web, arXiv, SEC, URL, and document tool call; the model never performs HTTP requests directly.
- Live synthesis receives source excerpts and URLs. Every factual output line must cite a collected source ID; missing or unknown citations reject the model output.
- Live Tavily and arXiv queries remove instruction prose and follow-up metadata before sending the query. Fixture search remains unchanged and deterministic.
- Architecture findings need a subject plus implementation terms such as candidate generation, retrieval, ranking, training, or serving.
- Competitor findings need a named competitor plus recommendation-system implementation evidence.
- Incomplete, fragmentary, boilerplate, promotional, social-card, or obviously noisy claims are dropped. Relevant pages yield multiple distinct full-content claims.
- Duplicate claims merge their source IDs; each URL appears once in the source list with its source type.
- Broad “what does it do” and “tell me more” questions group extractive claims into purpose, components, operation, use cases, status, and limitations; uncovered areas remain explicit.
- Fixture synthesis remains deterministic extractive Markdown.

## Checks

```powershell
uv run python -m compileall -q src tests
uv run pytest
uv lock --check
git diff --check
```

## Remaining limits

- Multi-topic questions are split into architecture and competitor sections when detected; unsupported topics and low-quality claims are reported or dropped.
- Topic follow-ups remain scoped to the missing topic. Generic multi-source questions may still use a second source category.
- Live HTML extraction is plain text. PDF, DOCX, and spreadsheets are deferred.
- Token accounting uses provider-reported totals when available plus the existing character estimate for stored evidence; exact billing can differ by provider.
- SEC lookup matches company names/tickers present in the query and returns recent filing metadata.
