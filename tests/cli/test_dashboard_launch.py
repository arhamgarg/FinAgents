from __future__ import annotations

import signal
import sys
from unittest.mock import MagicMock, patch


class TestDashboardLaunch:
    def test_subprocess_argv_no_shell(self) -> None:
        with patch("financialagents.cli.dashboard_launch.subprocess.Popen") as popen:
            popen.return_value = MagicMock(wait=MagicMock(return_value=0))
            from financialagents.cli.dashboard_launch import launch_dashboard

            launch_dashboard(port=8501, host="localhost")
            args, kwargs = popen.call_args
            assert isinstance(args[0], list), "argv must be a list (T-5-04)"
            assert kwargs.get("shell", False) is False, (
                "shell=True is forbidden (T-5-04 code-injection mitigation)"
            )

    def test_argv_format(self) -> None:
        with patch("financialagents.cli.dashboard_launch.subprocess.Popen") as popen:
            popen.return_value = MagicMock(wait=MagicMock(return_value=0))
            from financialagents.cli.dashboard_launch import launch_dashboard

            launch_dashboard(port=8501, host="localhost")
            cmd = popen.call_args.args[0]
            assert cmd == [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "financialagents/dashboard/app.py",
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
            ]

    def test_returns_child_exit_code(self) -> None:
        with patch("financialagents.cli.dashboard_launch.subprocess.Popen") as popen:
            popen.return_value = MagicMock(wait=MagicMock(return_value=42))
            from financialagents.cli.dashboard_launch import launch_dashboard

            assert launch_dashboard(port=1234, host="0.0.0.0") == 42

    def test_keyboard_interrupt_forwards_signal(self) -> None:
        mock_proc = MagicMock()
        wait_calls = {"n": 0}

        def _wait(*args, **kwargs):
            wait_calls["n"] += 1
            if wait_calls["n"] == 1:
                raise KeyboardInterrupt()
            return 130

        mock_proc.wait.side_effect = _wait
        with patch(
            "financialagents.cli.dashboard_launch.subprocess.Popen",
            return_value=mock_proc,
        ):
            from financialagents.cli.dashboard_launch import launch_dashboard

            rc = launch_dashboard(port=8501, host="localhost")

        assert mock_proc.send_signal.called, (
            "KeyboardInterrupt must forward a signal to the child"
        )
        sent = mock_proc.send_signal.call_args.args[0]
        if sys.platform == "win32":
            assert sent == signal.CTRL_BREAK_EVENT
        else:
            assert sent == signal.SIGINT
        assert rc == 130

    def test_no_bootstrap_import(self) -> None:
        import importlib
        import sys as _sys

        for mod_name in [
            "financialagents.cli.dashboard_launch",
            "streamlit.web.bootstrap",
        ]:
            _sys.modules.pop(mod_name, None)

        importlib.import_module("financialagents.cli.dashboard_launch")
        assert "streamlit.web.bootstrap" not in _sys.modules, (
            "dashboard_launch must NOT import streamlit.web.bootstrap "
            " — main-thread signal handler requirement)"
        )
