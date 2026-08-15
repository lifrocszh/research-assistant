# Project instructions

## Required skills

Use these skills for every task in this repository:

- `caveman` — terse, high-density communication. Default: `full`.
- `ponytail` — minimal implementation and YAGNI. Default: `full`.

Load both at session start. Keep them active for every response and code
change.

## Repository facts

- Python `>=3.12`, managed with `uv`; source uses the `src/` layout.
- CLI entry point is `research-assistant = research_assistant.cli:main`.
- LangGraph graph/checkpointer owns durable turn state; custom runtime code
  handles bounded dynamic delegation. SQLite checkpoints default to
  `.research-assistant/checkpoints.sqlite`.
- `registry.py` discovers capabilities; `engine.py` orchestrates bounded
  research; `tools.py` provides fixture/live adapters and extraction.
- `ui.py` provides the Phase 2 FastAPI/WebSocket UI. It consumes existing
  `RunEvent` objects and contains no orchestration logic. The browser page uses
  vanilla HTML/CSS/JavaScript, defaults to live mode, and renders a live event
  timeline beside the final answer.
- Phase 1 synthesis is deterministic extractive Markdown. Live HTML becomes
  plain text. PDF, DOCX, and spreadsheet extraction are not implemented.
- `plan.md` and `spec.md` describe intended scope. Verify source behavior
  before treating them as implemented.

## Development workflow

Run from repository root. On first setup or after dependency changes:

```powershell
uv sync
```

Before completion:

```powershell
uv run python -m compileall -q src tests
uv run pytest
uv lock --check
git diff --check
```

Use fixture mode for deterministic, offline checks:

```powershell
uv run research-assistant run "What does LangGraph provide?" --mode fixture
```

Live mode requires `TAVILY_API_KEY`; do not call live services from unit
tests. Use `tmp_path` for SQLite checkpoints and temporary fixture/document
files. Treat `.research-assistant/` as local runtime state, not source.

Run the Phase 2 UI:

```powershell
uv run --env-file .env uvicorn research_assistant.ui:app --reload
```

The UI defaults to live mode. Use the CLI for fixture, custom fixture,
document, pause, and resume workflows.

## Change rules

- Make the smallest change that fixes the behavior. Reuse existing registry,
  runtime, adapter, and model patterns before adding abstractions or
  dependencies.
- Add a focused pytest for every behavior change. Documentation-only changes
  need no test, but still require `uv run pytest` before completion.
- Keep tests offline and isolated: use pytest functions, `tmp_path` for files
  and SQLite, `monkeypatch` for environment, and `httpx.MockTransport` for
  HTTP behavior. Use FastAPI `TestClient` for UI/WebSocket tests and send
  `mode="fixture"` explicitly. Assert errors and emitted events explicitly.
- Keep `RunEvent` as the shared CLI/UI event contract. Do not add UI-specific
  orchestration or duplicate runtime state in the frontend.
- Update `README.md` when CLI flags, environment variables, or user-visible
  behavior changes. Update `plan.md` or `spec.md` only when project scope or
  intended architecture changes.
- Do not commit secrets, live credentials, checkpoint databases, caches, or
  generated files. Change `uv.lock` only when dependencies change.
- Before completion, inspect the diff and run the checks above. Preserve
  unrelated working-tree changes.

## Agent coordination

- Delegate only independent, read-only scopes unless edit ownership is
  explicit.
- Never let agents edit the same file concurrently. Parent agent reconciles
  findings and reviews all changes.
