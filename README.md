# Research Assistant

Python 3.12 prototype for bounded, durable, evidence-grounded research. LangGraph checkpoints generic research turns; a capability registry and custom orchestrator choose specialist agents and execution order at runtime.

## Setup

```powershell
uv sync
```

Fixture mode needs no network or credentials. Live web search needs `TAVILY_API_KEY`. Optional settings:

- `TAVILY_API_URL`: Tavily-compatible search endpoint. Defaults to Tavily.
- `SEC_USER_AGENT`: SEC-compliant product/contact user agent.

## CLI

```powershell
# Deterministic offline run
uv run research-assistant run "What does LangGraph provide?" --mode fixture

# Stream one JSON object per event; write answer separately
uv run research-assistant run "Research Acme's margin decline" --jsonl events.jsonl --output answer.md

# Include local Markdown/text evidence
uv run research-assistant run "Compare these notes with web evidence" --document notes.md

# Stop after one research turn, then continue same checkpoint
uv run research-assistant run "Research dynamic delegation" --thread demo --pause-after-turn
uv run research-assistant resume demo

# Live mode
$env:TAVILY_API_KEY = "..."
uv run research-assistant run "Current research question" --mode live
```

Use `--fixtures path.json` to replace built-in fixture records. File must contain a JSON array with `title`, `url`, `content`, and `source_type`; `claim` and `confidence` are optional.

Checkpoints default to `.research-assistant/checkpoints.sqlite`. Limits have CLI flags and default to 3 parallel agents, 12 total agents, depth 5, 180 seconds, 8 tool calls per agent, and 32,000 approximate tokens.

## Checks

```powershell
uv run python -m compileall -q src tests
uv run pytest
git diff --check
```

## Phase 1 limits

- Synthesis is deterministic extractive Markdown, not model-generated prose.
- Live HTML extraction is plain text. PDF, DOCX, and spreadsheets are deferred.
- Token usage is a conservative character-based estimate because no model is required.
- SEC lookup matches company names/tickers present in the query and returns recent filing metadata.
