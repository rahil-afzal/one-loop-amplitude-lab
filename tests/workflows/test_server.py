import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from amplitude_lab.workflows import server


def test_sse_frontend_streams_background_run_events(monkeypatch):
    def fake_run(run_id, task_name, max_retries):
        channel = server._channels[run_id]
        channel.publish({"type": "step.started", "state": "generating", "step_id": "rules"})
        channel.publish({"type": "run.done", "state": "completed_unverified"})

    monkeypatch.setattr(server, "_start_derivation", fake_run)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"
    try:
        page = urlopen(base_url, timeout=3).read().decode()
        assert "EventSource" in page
        assert "Problem interpretation &amp; approach" in page
        assert "startPlanningTimer" in page
        assert "Waiting for planner output" in page
        assert "Planner-proposed Lagrangian" in page
        assert "response.md" in page
        assert "Accepted derivation" in page
        assert "Deterministic verification" in page
        assert "Technical event log (raw SSE)" in page
        assert 'id="step-list"' in page
        assert "node.textContent=value" in page
        request = Request(
            base_url + "/api/derive",
            data=json.dumps({"task": "phi4_dim4"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = json.loads(urlopen(request, timeout=3).read())
        stream = urlopen(base_url + response["events"], timeout=3).read().decode()
        assert "event: update" in stream
        assert '"step_id": "rules"' in stream
        assert '"state": "completed_unverified"' in stream
    finally:
        httpd.shutdown()
        thread.join(timeout=3)
        httpd.server_close()
        server._channels.clear()


def test_sse_endpoint_rejects_task_paths():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{httpd.server_port}/api/derive",
        data=json.dumps({"task": "..\\secret"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        try:
            urlopen(request, timeout=3)
        except HTTPError as exc:
            assert exc.code == 400
            assert "not a path" in exc.read().decode()
        else:
            raise AssertionError("task path was accepted by the HTTP endpoint")
    finally:
        httpd.shutdown()
        thread.join(timeout=3)
        httpd.server_close()
