from __future__ import annotations

import logging
import re



class TestCLIMain:
    def test_version(self, cli_runner) -> None:
        from financialagents.cli.main import app

        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert re.match(
            r"^financialagents \d+\.\d+\.\d+\s*$", result.stdout
        ), f"unexpected --version output: {result.stdout!r}"

    def test_subcommands_registered(self, cli_runner) -> None:
        from financialagents.cli.main import app

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = result.stdout
        for cmd in ("run", "history", "show", "dashboard", "export"):
            assert cmd in out, f"subcommand {cmd!r} missing from --help output:\n{out}"

    def test_bare_invocation_runs_wizard(self, cli_runner, monkeypatch) -> None:
        from financialagents.cli import main as main_module

        called = {"n": 0}

        def fake_wizard():
            called["n"] += 1

        monkeypatch.setattr(
            "financialagents.cli.wizard.run_wizard", fake_wizard, raising=False
        )

        result = cli_runner.invoke(main_module.app, [])
        assert result.exit_code == 0, (
            f"bare-fa exited {result.exit_code}; stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        assert called["n"] == 1, (
            f"wizard called {called['n']}× (expected 1); "
            f"stdout={result.stdout!r}"
        )

    def test_debug_flag_enables_logging(self, cli_runner, monkeypatch) -> None:
        from financialagents.cli import main as main_module

        levels = {"after_root": None}

        async def fake_run_pipeline(*a, **kw):
            levels["after_root"] = logging.getLogger().level
            return 0

        monkeypatch.setattr(
            "financialagents.cli.run_command.run_pipeline",
            fake_run_pipeline,
            raising=False,
        )
        logging.getLogger().setLevel(logging.WARNING)

        result = cli_runner.invoke(
            main_module.app, ["--debug", "run", "RELIANCE", "--provider", "ollama"]
        )
        assert result.exit_code == 0, (
            f"exited {result.exit_code}; stderr={result.stderr!r}"
        )
        assert levels["after_root"] == logging.DEBUG, (
            f"expected logging.DEBUG ({logging.DEBUG}), "
            f"got {levels['after_root']}"
        )

    def test_install_completion_help(self, cli_runner) -> None:
        from financialagents.cli.main import app

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "completion" in result.stdout.lower(), (
            f"completion options missing from --help; "
            f"add_completion=True should ship them. stdout={result.stdout!r}"
        )
