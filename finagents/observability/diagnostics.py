import json
import threading
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler


class RunDiagnostics(BaseCallbackHandler):

    def __init__(self, path: Path, metadata: dict):
        self.path = path
        self.started = {}
        self.lock = threading.Lock()
        self._write({"kind": "run", **metadata})

    def _write(self, event):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.lock:
            self.started[str(run_id)] = time.monotonic()

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._finish(run_id, "completed")

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, "errored")

    def _finish(self, run_id, status):
        with self.lock:
            start = self.started.pop(str(run_id), None)
        self._write(
            {
                "kind": "llm",
                "span_id": str(run_id),
                "status": status,
                "elapsed_seconds": None if start is None else time.monotonic() - start,
            }
        )
