from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASH_RUN_PATH = (
    REPO_ROOT / "financialagents" / "dashboard" / "pages" / "run.py"
)


_VECTORS = (
    "connection failed: GROQ_API_KEY=secret123",
    "api_key=foo",
    "my_api_key_lookup=foo",
    "ANGELONE_TOTP_SECRET=xyz",
    "REDDIT_CLIENT_SECRET=abc",
    "OPENROUTER_API_KEY='sk-quoted'",
    "CEREBRAS_API_KEY: sk-colon-style",
    "boom: 260526-no0-marker",
    "",
)


class TestCliSharedRedactionParity:

    @pytest.mark.parametrize("vec", _VECTORS)
    def test_cli_and_shared_redact_identically(self, vec: str) -> None:
        from financialagents._redaction import redact_secrets as shared
        from financialagents.cli.run_command import (
            _redact_secrets as cli_redact,
        )

        cli_out = cli_redact(vec)
        shared_out = shared(vec)
        assert cli_out == shared_out, (
            f"redaction drift on vector {vec!r}: "
            f"cli={cli_out!r} shared={shared_out!r}"
        )

    def test_cli_helper_is_the_shared_callable(self) -> None:
        from financialagents._redaction import redact_secrets
        from financialagents.cli.run_command import _redact_secrets

        assert _redact_secrets is redact_secrets, (
            "CLI _redact_secrets is no longer the shared helper — a "
            "divergent local copy was reintroduced regression)"
        )

    def test_cli_pattern_is_the_shared_object(self) -> None:
        from financialagents._redaction import _SECRET_PATTERN as shared_pat
        from financialagents.cli.run_command import (
            _SECRET_PATTERN as cli_pat,
        )

        assert cli_pat is shared_pat, (
            "CLI _SECRET_PATTERN drifted from the shared module"
        )


class TestDashboardSharedRedactionParity:

    def _parse(self) -> ast.Module:
        return ast.parse(DASH_RUN_PATH.read_text(encoding="utf-8"))

    def test_dashboard_imports_shared_redaction_helpers(self) -> None:
        tree = self._parse()
        imported_names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (
                node.module == "financialagents._redaction"
            ):
                for alias in node.names:
                    imported_names.add(alias.name)
        assert {"_SECRET_PATTERN", "redact_secrets"}.issubset(
            imported_names
        ), (
            "Dashboard run.py must import _SECRET_PATTERN and "
            "redact_secrets from financialagents._redaction — found: "
            f"{imported_names!r}  regression)"
        )

    def test_dashboard_aliases_redact_secrets_to_shared(self) -> None:
        tree = self._parse()
        found_alias = False
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_redact_secrets"
                and isinstance(node.value, ast.Name)
                and node.value.id == "redact_secrets"
            ):
                found_alias = True
                break
        assert found_alias, (
            "Dashboard run.py must alias `_redact_secrets = "
            "redact_secrets` at module scope so the historic private "
            "name still resolves to the shared callable "
        )
