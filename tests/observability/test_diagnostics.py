import json
from uuid import uuid4

from finagents.observability.diagnostics import RunDiagnostics


def test_diagnostics_omit_prompts_responses_and_errors(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = RunDiagnostics(path, {"run_id": "aabbccdd"})
    for status in ("completed", "errored"):
        request = uuid4()
        trace.on_chat_model_start({}, ["SECRET prompt"], run_id=request)
        if status == "completed":
            trace.on_llm_end("SECRET response", run_id=request)
        else:
            trace.on_llm_error(RuntimeError("SECRET key"), run_id=request)
    content = path.read_text()
    assert "SECRET" not in content
    events = [json.loads(line) for line in content.splitlines()]
    assert [event["status"] for event in events[1:]] == ["completed", "errored"]
    assert all(event["elapsed_seconds"] >= 0 for event in events[1:])
