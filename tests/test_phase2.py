from __future__ import annotations

from fastapi.testclient import TestClient

from research_assistant.ui import RunRequest, create_app


def test_websocket_streams_existing_events_and_answer(tmp_path) -> None:
    assert RunRequest(question="x").mode == "live"
    app = create_app(tmp_path / "checkpoints.sqlite")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'class="research-frame"' in page.text
        assert 'class="report-panel conversation-panel"' in page.text
        assert 'id="question" name="question"' in page.text
        assert 'id="events"' in page.text
        assert 'id="answer"' in page.text
        assert 'id="live-update"' in page.text
        assert 'id="activity-details"' in page.text
        assert "What can I research for you?" in page.text
        assert "Delegated to ${agent}." in page.text
        assert "is calling ${tool}" in page.text
        assert "Enough context gathered." in page.text
        assert "scrollIntoView" not in page.text
        assert "\x08" not in page.text
        assert r"replace(/\b\w/g" in page.text
        assert "String(value || '').replaceAll" in page.text
        assert "--bg-primary: #f7f7f5" in page.text
        assert "--accent: #2477d4" in page.text
        assert ".workspace { min-height: 420px; }" in page.text
        assert "#events { max-height: 320px;" in page.text
        assert "scrollbar-gutter: stable" in page.text
        assert page.text.count("overflow-y: auto") == 1
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"question": "What does LangGraph provide?", "mode": "fixture"})
            messages = []
            for _ in range(20):
                messages.append(websocket.receive_json())
                if messages[-1]["event_type"] == "answer":
                    break

    assert messages[0]["event_type"] == "run_started"
    assert any(message["event_type"] == "agent_started" for message in messages)
    assert any(message["event_type"] == "tool_started" for message in messages)
    assert any(message["event_type"] == "tool_finished" for message in messages)
    assert any(message["event_type"] == "evidence_added" for message in messages)
    assert any(message["event_type"] == "run_finished" for message in messages)
    assert messages[-1]["event_type"] == "answer"
    assert "LangGraph" in messages[-1]["final_answer"]


def test_page_exposes_live_and_fixture_modes(tmp_path) -> None:
    app = create_app(tmp_path / "checkpoints.sqlite")
    with TestClient(app) as client:
        page = client.get("/")

    assert '<textarea id="question" name="question"' in page.text
    assert '<select id="mode" name="mode" aria-label="Research mode">' in page.text
    assert 'value="live" selected>Live: adaptive model research' in page.text
    assert 'value="fixture">Fixture: offline, no LLM calls' in page.text
    assert "socket.send(JSON.stringify({ question, mode: mode.value }))" in page.text
