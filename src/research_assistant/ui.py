from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from .engine import ResearchRuntime
from .models import RunEvent


class RunRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["fixture", "live"] = "live"
    documents: list[str] = Field(default_factory=list)
    fixture_path: str | None = None
    thread_id: str | None = None
    pause_after_turn: bool = False


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Assistant</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      --bg-primary: #000;
      --bg-sidebar: #050505;
      --bg-surface: #212121;
      --bg-selected: #1f1f1f;
      --text-primary: #f2f2f2;
      --text-secondary: #b4b4b4;
      --text-muted: #8e8e8e;
      --border: #2f2f2f;
      --accent: #2f6fd6;
      --accent-hover: #3b7be5;
      background: var(--bg-primary);
      color: var(--text-primary);
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; background: var(--bg-primary); }
    .shell { max-width: 1440px; min-height: 100vh; margin: auto; padding: 28px; }
    .masthead, .question-form, .panel { border: 1px solid var(--border); background: var(--bg-sidebar); }
    .masthead { display: flex; justify-content: space-between; align-items: center; padding: 18px 20px; border-radius: 16px 16px 0 0; }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 1.1rem; letter-spacing: -.02em; }
    .eyebrow, .meta, .event-type { color: var(--text-muted); font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; }
    .eyebrow { margin-bottom: 5px; }
    .run-state { display: flex; align-items: center; gap: 10px; color: var(--text-secondary); font-size: .82rem; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); }
    .dot.running { background: var(--accent); box-shadow: 0 0 8px rgb(47 111 214 / .65); }
    .dot.ready { background: #34d399; }
    .dot.failed { background: #fb7185; }
    .question-form { display: flex; gap: 10px; padding: 14px; border-top: 0; }
    input, select { min-width: 0; border: 1px solid var(--border); border-radius: 26px; padding: 12px 18px; background: var(--bg-surface); color: inherit; font: inherit; }
    input { width: 100%; }
    select { flex: 0 0 auto; max-width: 260px; }
    input::placeholder { color: var(--text-muted); }
    input:focus { outline: 2px solid var(--accent); border-color: transparent; }
    button { border: 0; border-radius: 22px; padding: 0 20px; background: var(--accent); color: #fff; font: inherit; font-weight: 800; cursor: pointer; transition: background .15s ease; }
    button:hover:not(:disabled) { background: var(--accent-hover); }
    button:disabled { cursor: wait; opacity: .55; }
    .workspace { display: grid; grid-template-columns: minmax(300px, .85fr) minmax(0, 1.5fr); gap: 16px; margin-top: 16px; }
    .panel { height: 580px; min-height: 0; display: flex; flex-direction: column; border-radius: 16px; overflow: hidden; }
    .panel-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 18px; border-bottom: 1px solid var(--border); }
    h2 { font-size: .92rem; }
    #events { flex: 1; min-height: 0; margin: 0; padding: 8px 16px 16px; overflow-y: auto; list-style: none; }
    .event { display: grid; grid-template-columns: 12px 1fr; gap: 12px; padding: 13px 8px; border-bottom: 1px solid var(--border); border-radius: 8px; }
    .event:hover { background: var(--bg-selected); }
    .event:last-child { border-bottom: 0; }
    .event-mark { width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; background: var(--text-muted); }
    .event.planning .event-mark, .event.parallel_started .event-mark { background: #a78bfa; }
    .event.agent_started .event-mark, .event.tool_started .event-mark { background: var(--accent-hover); }
    .event.agent_finished .event-mark, .event.tool_finished .event-mark, .event.evidence_added .event-mark { background: #34d399; }
    .event.conflict_detected .event-mark, .event.error .event-mark { background: #fb7185; }
    .event.followup_started .event-mark, .event.synthesis_started .event-mark { background: #fbbf24; }
    .event-title { display: flex; justify-content: space-between; gap: 12px; color: var(--text-primary); font-size: .86rem; font-weight: 700; }
    .event-summary { margin-top: 4px; color: var(--text-secondary); font-size: .82rem; line-height: 1.45; }
    .empty { padding: 28px 2px; color: var(--text-muted); font-size: .88rem; }
    .answer { flex: 1; min-height: 0; padding: 24px; overflow-y: auto; color: var(--text-primary); line-height: 1.7; }
    .answer h1 { margin: 0 0 18px; font-size: 1.45rem; }
    .answer h2 { margin: 22px 0 9px; font-size: 1rem; }
    .answer h3 { margin: 16px 0 7px; font-size: .92rem; }
    .answer p { margin: 0 0 12px; white-space: pre-wrap; }
    .answer ul { margin: 0 0 12px; padding-left: 20px; }
    .answer a { color: var(--accent-hover); }
    .answer .placeholder { color: var(--text-muted); }
    @media (max-width: 780px) { .shell { padding: 14px; } .masthead { border-radius: 12px 12px 0 0; } .workspace { grid-template-columns: 1fr; } .panel { min-height: 0; } .answer { min-height: 320px; } }
  </style>
</head>
<body>
  <main class="shell">
    <header class="masthead">
      <div><p class="eyebrow">Dynamic research runtime</p><h1>Research Assistant</h1></div>
      <div class="run-state"><span id="state-dot" class="dot"></span><span id="run-state">Ready</span><span id="elapsed">00:00</span></div>
    </header>
    <form id="run" class="question-form">
      <input name="question" required placeholder="Ask a research question" autocomplete="off">
      <select id="mode" name="mode" aria-label="Research mode">
        <option value="live" selected>Live: adaptive model research</option>
        <option value="fixture">Fixture: offline, no LLM calls</option>
      </select>
      <button id="submit" type="submit">Run</button>
    </form>
    <section class="workspace">
      <aside class="panel" aria-label="Live orchestration activity">
        <div class="panel-header"><div><p class="eyebrow">Live run</p><h2>Orchestration</h2></div><span id="event-count" class="meta">0 events</span></div>
        <ol id="events"><li class="empty">Run a question to see delegation, tool use, and follow-up research.</li></ol>
      </aside>
      <article class="panel" aria-label="Research answer">
        <div class="panel-header"><div><p class="eyebrow">Grounded result</p><h2>Answer</h2></div><span id="answer-mode" class="meta">Live: adaptive</span></div>
        <div id="answer" class="answer"><p class="placeholder">Sources and cited findings appear here when the run completes.</p></div>
      </article>
    </section>
  </main>
  <script>
    const form = document.querySelector('#run');
    const events = document.querySelector('#events');
    const answer = document.querySelector('#answer');
    const mode = document.querySelector('#mode');
    const answerMode = document.querySelector('#answer-mode');
    const submit = document.querySelector('#submit');
    const state = document.querySelector('#run-state');
    const dot = document.querySelector('#state-dot');
    const elapsed = document.querySelector('#elapsed');
    const count = document.querySelector('#event-count');
    const labels = { run_started: 'Run started', planning: 'Orchestrator planning', parallel_started: 'Parallel research', agent_started: 'Agent started', tool_started: 'Tool started', tool_finished: 'Tool finished', agent_finished: 'Agent finished', evidence_added: 'Evidence added', conflict_detected: 'Conflict detected', followup_started: 'Follow-up research', synthesis_started: 'Synthesizing', run_finished: 'Run finished', error: 'Run failed' };
    let eventTotal = 0, startedAt, clock;

    function setState(text, tone = '') { state.textContent = text; dot.className = `dot ${tone}`; }
    function stopRun(text, tone) { clearInterval(clock); submit.disabled = false; setState(text, tone); }
    function updateClock() { const seconds = Math.floor((Date.now() - startedAt) / 1000); elapsed.textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; }

    function addEvent(message) {
      events.querySelector('.empty')?.remove();
      eventTotal += 1; count.textContent = `${eventTotal} event${eventTotal === 1 ? '' : 's'}`;
      const item = document.createElement('li'); item.className = `event ${labels[message.event_type] ? message.event_type : ''}`;
      const mark = document.createElement('span'); mark.className = 'event-mark';
      const body = document.createElement('div');
      const title = document.createElement('div'); title.className = 'event-title';
      const name = document.createElement('span'); name.textContent = labels[message.event_type] || message.event_type;
      const status = document.createElement('span'); status.className = 'event-type'; status.textContent = message.status;
      title.append(name, status);
      const detail = [message.agent, message.task, message.result_summary].filter(Boolean).join(' — ');
      if (detail) { const summary = document.createElement('p'); summary.className = 'event-summary'; summary.textContent = detail; body.append(title, summary); } else body.append(title);
      item.append(mark, body); events.append(item); item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    function inline(text) {
      const escaped = text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
      return escaped.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\s)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    }

    function renderMarkdown(markdown) {
      const output = []; let listOpen = false;
      for (const line of markdown.split('\\n')) {
        const heading = line.match(/^(#{1,3})\\s+(.+)$/);
        if (heading) { if (listOpen) { output.push('</ul>'); listOpen = false; } output.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`); continue; }
        if (line.startsWith('- ')) { if (!listOpen) { output.push('<ul>'); listOpen = true; } output.push(`<li>${inline(line.slice(2))}</li>`); continue; }
        if (listOpen) { output.push('</ul>'); listOpen = false; }
        if (line) output.push(`<p>${inline(line)}</p>`);
      }
      if (listOpen) output.push('</ul>');
      answer.innerHTML = output.join('') || '<p class="placeholder">No answer returned.</p>';
    }

    form.onsubmit = event => {
      event.preventDefault();
      eventTotal = 0; events.replaceChildren(); count.textContent = '0 events';
      answer.innerHTML = '<p class="placeholder">Researching sources and reconciling evidence...</p>';
      submit.disabled = true; startedAt = Date.now(); updateClock(); clearInterval(clock); clock = setInterval(updateClock, 1000); setState('Connecting', 'running');
      const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
      const isLive = mode.value === 'live';
      answerMode.textContent = isLive ? 'Live: adaptive' : 'Fixture: no LLM calls';
      socket.onopen = () => { setState(isLive ? 'Researching' : 'Running fixture', 'running'); socket.send(JSON.stringify({ question: form.question.value, mode: mode.value })); };
      socket.onmessage = event => {
        const message = JSON.parse(event.data);
        if (message.event_type === 'answer') { renderMarkdown(message.final_answer); stopRun(message.status === 'paused' ? 'Paused' : 'Complete', 'ready'); return; }
        addEvent(message);
        if (message.event_type === 'error') { answer.innerHTML = `<p class="placeholder">${inline(message.result_summary || 'Run failed.')}</p>`; stopRun('Failed', 'failed'); }
      };
      socket.onerror = () => { if (submit.disabled) { addEvent({ event_type: 'error', status: 'failed', result_summary: 'Could not connect to research service.' }); stopRun('Failed', 'failed'); } };
    };
  </script>
</body>
</html>"""


def create_app(checkpoint_path: str | Path = ".research-assistant/checkpoints.sqlite") -> FastAPI:
    app = FastAPI(title="Research Assistant")
    checkpoint = Path(checkpoint_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.websocket("/ws")
    async def run(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            request = RunRequest.model_validate(await websocket.receive_json())
        except ValidationError as exc:
            await websocket.send_json({"event_type": "error", "status": "failed", "result_summary": str(exc)})
            await websocket.close(code=1003)
            return

        loop = asyncio.get_running_loop()
        events: asyncio.Queue[RunEvent | dict[str, str] | None] = asyncio.Queue()

        def publish(event: RunEvent) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        def run_research() -> None:
            try:
                with ResearchRuntime(checkpoint, publish) as runtime:
                    state = runtime.run(
                        request.question,
                        thread_id=request.thread_id or str(uuid4()),
                        mode=request.mode,
                        documents=request.documents,
                        fixture_path=request.fixture_path,
                        pause_after_turn=request.pause_after_turn,
                    )
                loop.call_soon_threadsafe(events.put_nowait, {"event_type": "answer", "status": state.status, "final_answer": state.final_answer or ""})
            except Exception as exc:
                loop.call_soon_threadsafe(events.put_nowait, {"event_type": "error", "status": "failed", "result_summary": str(exc)})
            finally:
                loop.call_soon_threadsafe(events.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(run_research))
        while (event := await events.get()) is not None:
            if isinstance(event, RunEvent):
                await websocket.send_json(event.model_dump(mode="json"))
            else:
                await websocket.send_json(event)
        await task

    return app


app = create_app()
