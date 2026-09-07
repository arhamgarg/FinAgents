from __future__ import annotations

import signal
import subprocess
import sys
from typing import Any


def launch_dashboard(*, port: int = 8501, host: str = "localhost") -> int:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "finagents/dashboard/app.py",
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, **kwargs)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 130
