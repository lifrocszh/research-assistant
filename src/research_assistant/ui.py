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
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      --bg-primary: #f7f7f5;
      --bg-surface: #fff;
      --bg-muted: #f0f0ed;
      --bg-selected: #f4f7fb;
      --text-primary: #202123;
      --text-secondary: #5f6368;
      --text-muted: #858585;
      --border: #e5e5e0;
      --accent: #2477d4;
      --accent-hover: #1b63b5;
      --success: #18864b;
      --danger: #c24150;
      --shadow: 0 12px 36px rgb(32 33 35 / .07);
      background: var(--bg-primary);
      color: var(--text-primary);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg-primary: #171717;
        --bg-surface: #212121;
        --bg-muted: #2b2b2b;
        --bg-selected: #292d33;
        --text-primary: #f2f2f2;
        --text-secondary: #b4b4b4;
        --text-muted: #8e8e8e;
        --border: #3a3a3a;
        --accent: #5b9bea;
        --accent-hover: #73aaf0;
        --success: #4fd38b;
        --danger: #fb7185;
        --shadow: 0 12px 36px rgb(0 0 0 / .18);
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; background: var(--bg-primary); }
    button, textarea, select { font: inherit; }
    .shell { max-width: 1480px; min-height: 100vh; margin: auto; padding: 22px clamp(16px, 4vw, 56px); }
    .topbar { display: flex; justify-content: space-between; align-items: center; gap: 18px; padding: 4px 0 24px; }
    .brand { display: flex; align-items: center; gap: 10px; color: var(--text-primary); font-size: .95rem; font-weight: 750; }
    .brand-mark { display: grid; width: 28px; height: 28px; place-items: center; border-radius: 9px; background: var(--text-primary); color: var(--bg-primary); font-size: .82rem; }
    .brand-badge { padding: 5px 9px; border: 1px solid var(--border); border-radius: 999px; color: var(--text-muted); font-size: .68rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }
    h1, h2, h3, p { margin: 0; }
    h1 { max-width: 760px; font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.03; letter-spacing: -.055em; }
    h2 { font-size: 1rem; letter-spacing: -.02em; }
    .eyebrow, .meta, .event-type { color: var(--text-muted); font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; }
    .eyebrow { margin-bottom: 8px; font-weight: 700; }
    .run-state { display: flex; align-items: center; gap: 9px; color: var(--text-secondary); font-size: .8rem; white-space: nowrap; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); }
    .dot.running { background: var(--accent); box-shadow: 0 0 9px rgb(36 119 212 / .55); }
    .dot.ready { background: var(--success); }
    .dot.failed { background: var(--danger); }
    .research-frame { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; gap: 22px; min-height: calc(100vh - 92px); }
    .research-intro { padding: 22px 2px 0; }
    .research-intro p { max-width: 650px; margin-top: 16px; color: var(--text-secondary); line-height: 1.6; }
    .workspace { display: grid; grid-template-columns: minmax(215px, 250px) minmax(0, 1fr); gap: 22px; height: min(720px, calc(100vh - 380px)); min-height: 520px; }
    .activity-rail, .report-panel, .question-form { border: 1px solid var(--border); background: var(--bg-surface); box-shadow: var(--shadow); }
    .activity-rail, .report-panel { min-height: 0; border-radius: 20px; overflow: hidden; }
    .activity-rail { display: flex; flex-direction: column; }
    .rail-header, .report-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; padding: 20px 20px 17px; border-bottom: 1px solid var(--border); }
    .progress { padding: 15px 20px 12px; border-bottom: 1px solid var(--border); }
    .progress-item { display: flex; align-items: center; gap: 9px; padding: 6px 0; color: var(--text-muted); font-size: .78rem; }
    .progress-item.active { color: var(--text-primary); }
    .progress-item.done { color: var(--success); }
    .progress-mark { width: 7px; height: 7px; border: 1px solid currentColor; border-radius: 50%; }
    .progress-item.active .progress-mark { background: var(--accent); border-color: var(--accent); box-shadow: 0 0 7px rgb(36 119 212 / .45); }
    .progress-item.done .progress-mark { background: currentColor; }
    .progress-value { margin-left: auto; color: var(--text-muted); font-size: .7rem; }
    #events { flex: 1; min-height: 0; margin: 0; padding: 9px 12px 14px; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; scrollbar-width: thin; list-style: none; }
    .event { position: relative; display: grid; grid-template-columns: 15px 1fr; gap: 9px; padding: 10px 8px; border-radius: 10px; }
    .event:hover { background: var(--bg-selected); }
    .event:not(:last-child)::before { position: absolute; top: 22px; bottom: -10px; left: 14px; width: 1px; background: var(--border); content: ""; }
    .event-mark { z-index: 1; width: 7px; height: 7px; margin-top: 5px; border: 2px solid var(--bg-surface); border-radius: 50%; background: var(--text-muted); box-shadow: 0 0 0 1px currentColor; }
    .event.planning .event-mark, .event.parallel_started .event-mark { color: #8b5cf6; background: #8b5cf6; }
    .event.agent_started .event-mark, .event.tool_started .event-mark { color: var(--accent); background: var(--accent); }
    .event.agent_finished .event-mark, .event.tool_finished .event-mark, .event.evidence_added .event-mark { color: var(--success); background: var(--success); }
    .event.conflict_detected .event-mark, .event.error .event-mark { color: var(--danger); background: var(--danger); }
    .event.followup_started .event-mark, .event.synthesis_started .event-mark { color: #d89008; background: #d89008; }
    .event-title { display: flex; justify-content: space-between; gap: 8px; color: var(--text-primary); font-size: .78rem; font-weight: 700; }
    .event-summary { margin-top: 3px; color: var(--text-secondary); font-size: .75rem; line-height: 1.4; }
    .event-time { display: block; margin-top: 4px; color: var(--text-muted); font-size: .68rem; }
    .empty { padding: 22px 2px; color: var(--text-muted); font-size: .8rem; line-height: 1.5; }
    .report-panel { display: flex; flex-direction: column; }
    .report-header { align-items: center; }
    .report-title { max-width: 70ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .answer { flex: 1; min-height: 0; max-width: 900px; padding: 34px clamp(22px, 5vw, 68px) 54px; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; scrollbar-width: thin; color: var(--text-primary); line-height: 1.72; }
    #events::-webkit-scrollbar, .answer::-webkit-scrollbar { width: 8px; }
    #events::-webkit-scrollbar-thumb, .answer::-webkit-scrollbar-thumb { border: 2px solid transparent; border-radius: 999px; background: var(--border); background-clip: padding-box; }
    .answer h1 { margin: 0 0 22px; font-size: 1.8rem; letter-spacing: -.035em; }
    .answer h2 { margin: 28px 0 10px; font-size: 1.08rem; }
    .answer h3 { margin: 20px 0 8px; font-size: .98rem; }
    .answer p { margin: 0 0 14px; white-space: pre-wrap; }
    .answer ul { margin: 0 0 14px; padding-left: 22px; }
    .answer li { padding-left: 4px; }
    .answer a { color: var(--accent-hover); text-underline-offset: 3px; }
    .answer .placeholder { max-width: 520px; color: var(--text-muted); }
    .question-form { position: sticky; bottom: 18px; display: grid; gap: 12px; padding: 14px; border-radius: 18px; }
    .prompt-label { display: block; padding: 0 5px; color: var(--text-muted); font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    textarea, select { min-width: 0; border: 1px solid var(--border); border-radius: 12px; background: var(--bg-muted); color: inherit; }
    textarea { width: 100%; min-height: 74px; resize: vertical; padding: 14px 15px; line-height: 1.45; }
    textarea::placeholder { color: var(--text-muted); }
    textarea:focus, select:focus, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .prompt-controls { display: flex; justify-content: flex-end; align-items: center; gap: 10px; }
    select { max-width: 280px; padding: 10px 12px; }
    button { border: 0; border-radius: 11px; padding: 11px 18px; background: var(--accent); color: #fff; font-weight: 800; cursor: pointer; transition: background .15s ease, transform .15s ease; }
    button:hover:not(:disabled) { background: var(--accent-hover); transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: .55; }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
    @media (max-width: 900px) {
      .shell { padding: 18px 14px; }
      .workspace { grid-template-columns: 1fr; height: auto; min-height: 0; }
      .activity-rail { max-height: 280px; }
      .report-panel { min-height: 520px; }
      .research-frame { min-height: 0; }
    }
    @media (max-width: 560px) {
      .topbar { align-items: flex-start; }
      .brand-badge { display: none; }
      .run-state { flex-wrap: wrap; justify-content: flex-end; }
      h1 { font-size: 2.2rem; }
      .prompt-controls { align-items: stretch; flex-direction: column; }
      select, button { max-width: none; width: 100%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark" aria-hidden="true">R</span><span>Research Assistant</span><span class="brand-badge">Deep research</span></div>
      <div class="run-state"><span id="state-dot" class="dot"></span><span id="run-state">Ready</span><span id="elapsed">00:00</span></div>
    </header>
    <section class="research-frame">
      <header class="research-intro">
        <p class="eyebrow">Research workspace</p>
        <h1>Ask a question. Get a grounded report.</h1>
        <p>Research runs stay visible while they work. Follow planning, delegated agents, tools, and evidence before reviewing the final answer.</p>
      </header>
      <section class="workspace">
        <aside class="activity-rail" aria-label="Live research activity">
          <div class="rail-header"><div><p class="eyebrow">Live run</p><h2>Research activity</h2></div><span id="event-count" class="meta">0 events</span></div>
          <div class="progress" aria-label="Research progress">
            <div class="progress-item" data-progress="plan"><span class="progress-mark"></span><span>Plan</span><span class="progress-value">—</span></div>
            <div class="progress-item" data-progress="agents"><span class="progress-mark"></span><span>Agents</span><span class="progress-value">—</span></div>
            <div class="progress-item" data-progress="tools"><span class="progress-mark"></span><span>Tools</span><span class="progress-value">—</span></div>
            <div class="progress-item" data-progress="evidence"><span class="progress-mark"></span><span>Evidence</span><span class="progress-value">—</span></div>
            <div class="progress-item" data-progress="synthesis"><span class="progress-mark"></span><span>Synthesis</span><span class="progress-value">—</span></div>
          </div>
          <ol id="events"><li class="empty">Run a question to see planning, delegation, tool use, and evidence.</li></ol>
        </aside>
        <article class="report-panel" aria-label="Research report">
          <div class="report-header"><div><p class="eyebrow">Grounded result</p><h2 id="report-title" class="report-title">Awaiting question</h2></div><span id="answer-mode" class="meta">Live: adaptive</span></div>
          <div id="answer" class="answer"><p class="placeholder">Your cited findings and synthesized report will appear here when research completes.</p></div>
        </article>
      </section>
      <form id="run" class="question-form">
        <label class="prompt-label" for="question">Research prompt</label>
        <textarea id="question" name="question" required rows="2" placeholder="What do you want to investigate?" autocomplete="off"></textarea>
        <div class="prompt-controls">
          <select id="mode" name="mode" aria-label="Research mode">
            <option value="live" selected>Live: adaptive model research</option>
            <option value="fixture">Fixture: offline, no LLM calls</option>
          </select>
          <button id="submit" type="submit">Start research</button>
        </div>
      </form>
    </section>
  </main>
  <script>
    const form = document.querySelector('#run');
    const events = document.querySelector('#events');
    const answer = document.querySelector('#answer');
    const reportTitle = document.querySelector('#report-title');
    const mode = document.querySelector('#mode');
    const answerMode = document.querySelector('#answer-mode');
    const submit = document.querySelector('#submit');
    const state = document.querySelector('#run-state');
    const dot = document.querySelector('#state-dot');
    const elapsed = document.querySelector('#elapsed');
    const count = document.querySelector('#event-count');
    const labels = { run_started: 'Run started', planning: 'Orchestrator planning', parallel_started: 'Parallel research', agent_started: 'Agent started', tool_started: 'Tool started', tool_finished: 'Tool finished', agent_finished: 'Agent finished', evidence_added: 'Evidence added', conflict_detected: 'Conflict detected', followup_started: 'Follow-up research', synthesis_started: 'Synthesizing', run_finished: 'Run finished', error: 'Run failed' };
    const progressMap = { planning: 'plan', parallel_started: 'plan', agent_started: 'agents', agent_finished: 'agents', tool_started: 'tools', tool_finished: 'tools', evidence_added: 'evidence', conflict_detected: 'evidence', followup_started: 'agents', synthesis_started: 'synthesis', run_finished: 'synthesis' };
    let eventTotal = 0, startedAt, clock;

    function setState(text, tone = '') { state.textContent = text; dot.className = `dot ${tone}`; }
    function stopRun(text, tone) { clearInterval(clock); submit.disabled = false; setState(text, tone); }
    function updateClock() { const seconds = Math.floor((Date.now() - startedAt) / 1000); elapsed.textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; }
    function resetProgress() { document.querySelectorAll('[data-progress]').forEach(item => { item.className = 'progress-item'; item.querySelector('.progress-value').textContent = '—'; }); }
    function updateProgress(eventType, status) {
      const phase = progressMap[eventType];
      if (!phase) return;
      const item = document.querySelector(`[data-progress="${phase}"]`);
      if (!item) return;
      item.classList.add(status === 'finished' ? 'done' : 'active');
      if (status === 'finished') item.classList.remove('active');
      item.querySelector('.progress-value').textContent = status === 'finished' ? 'done' : 'live';
    }

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
      if (message.timestamp) { const time = document.createElement('time'); time.className = 'event-time'; time.textContent = new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); body.append(time); }
      item.append(mark, body); events.append(item); item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      updateProgress(message.event_type, message.status);
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
      const question = form.elements.question.value.trim();
      if (!question) return;
      eventTotal = 0; events.replaceChildren(); count.textContent = '0 events';
      resetProgress(); reportTitle.textContent = question; answer.innerHTML = '<p class="placeholder">Researching sources and reconciling evidence...</p>';
      submit.disabled = true; startedAt = Date.now(); updateClock(); clearInterval(clock); clock = setInterval(updateClock, 1000); setState('Connecting', 'running');
      const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
      const isLive = mode.value === 'live';
      answerMode.textContent = isLive ? 'Live: adaptive' : 'Fixture: no LLM calls';
      socket.onopen = () => { setState(isLive ? 'Researching' : 'Running fixture', 'running'); socket.send(JSON.stringify({ question, mode: mode.value })); };
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
