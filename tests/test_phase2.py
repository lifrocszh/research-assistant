from __future__ import annotations

from fastapi.testclient import TestClient

from research_assistant.ui import RunRequest, create_app


def test_websocket_streams_existing_events_and_answer(tmp_path) -> None:
    assert RunRequest(question="x").mode == "live"
    app = create_app(tmp_path / "checkpoints.sqlite")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'id="events"' in page.text
        assert 'id="answer"' in page.text
        assert "Dynamic research runtime" in page.text
        assert "--bg-primary: #000" in page.text
        assert "--accent: #2f6fd6" in page.text
        assert ".panel { height: 580px" in page.text
        assert "#events { flex: 1; min-height: 0;" in page.text
        assert ".answer { flex: 1; min-height: 0;" in page.text
        assert page.text.count("overflow-y: auto") == 2
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"question": "What does LangGraph provide?", "mode": "fixture"})
            messages = []
            for _ in range(20):
                messages.append(websocket.receive_json())
                if messages[-1]["event_type"] == "answer":
                    break

    assert messages[0]["event_type"] == "run_started"
    assert any(message["event_type"] == "run_finished" for message in messages)
    assert messages[-1]["event_type"] == "answer"
    assert "LangGraph" in messages[-1]["final_answer"]


def test_page_exposes_live_and_fixture_modes(tmp_path) -> None:
    app = create_app(tmp_path / "checkpoints.sqlite")
    with TestClient(app) as client:
        page = client.get("/")

    assert '<select id="mode" name="mode" aria-label="Research mode">' in page.text
    assert 'value="live" selected>Live: adaptive model research' in page.text
    assert 'value="fixture">Fixture: offline, no LLM calls' in page.text
    assert "mode: mode.value" in page.text
