from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_comments import check

CLEAN = {
    "finagents/clean.py": (
        "from __future__ import annotations\n"
        "\n"
        "import os  # noqa: F401\n"
        "\n"
        "TEMPLATE = 'name = \"demo\"  # inside a string'\n"
        "\n"
        "\n"
        "def run() -> int:\n"
        "    value = os.environ.get('X')  # type: ignore[assignment]\n"
        "    'not a docstring' if value else None\n"
        "    return 0  # pragma: no cover\n"
    ),
    "scripts/run.sh": "#!/usr/bin/env bash\nset -euo pipefail\necho '# not a comment'\n",
    "configs/a.toml": 'name = "a#b"\n',
    ".github/workflows/ci.yml": "on: push\n",
    "finagents/storage/migrations/001.sql": "SELECT '-- not a comment';\n",
    "scripts/tool.py": "#!/usr/bin/env python3\nprint(1)\n",
    ".gitignore": "# ignored\n.venv/\n",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={"HOME": str(repo), "PATH": "/usr/bin:/bin"},
    )


def _add(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    for rel, text in CLEAN.items():
        _add(tmp_path, rel, text)
    _git(tmp_path, "add", ".")
    return tmp_path


def test_strings_bare_pragmas_and_shebangs_pass(repo: Path) -> None:
    assert check(repo) == []


@pytest.mark.parametrize(
    ("rel", "text", "line"),
    [
        ("finagents/full.py", "x = 1\n# why x is one\n", 2),
        ("finagents/trailing.py", "x = 1  # why x is one\n", 1),
        ("finagents/reason.py", "import os  # noqa: F401 - re-exported\n", 1),
        ("finagents/blanket.py", "import os  # noqa\n", 1),
        ("finagents/bare.py", "x: int = '1'  # type: ignore\n", 1),
        ("finagents/late_shebang.py", "x = 1\n#!/usr/bin/env python3\n", 2),
        ("finagents/broken.py", "def f(:\n", 1),
        ("finagents/module_doc.py", '"""Module."""\nx = 1\n', 1),
        ("tests/test_doc.py", "def test_x() -> None:\n    '''Doc.'''\n    assert True\n", 2),
        ("finagents/cls.py", "class A:\n    '''Doc.'''\n", 2),
        ("finagents/coro.py", "async def f() -> None:\n    '''Doc.'''\n", 2),
        ("scripts/hook.sh", "#!/usr/bin/env bash\n# explain\necho ok\n", 2),
        ("configs/b.toml", "# explain\nname = 'b'\n", 1),
        (".github/workflows/a.yml", "on: push\n  # explain\n", 2),
        (".github/workflows/b.yaml", "# explain\non: push\n", 1),
        ("finagents/migrations/002.sql", "SELECT 1;\n-- explain\n", 2),
        ("finagents/migrations/003.sql", "/* explain */\nSELECT 1;\n", 1),
    ],
)
def test_each_comment_or_docstring_is_reported_with_its_line(
    repo: Path, rel: str, text: str, line: int
) -> None:
    _add(repo, rel, text)
    (failure,) = check(repo)
    assert failure.startswith(f"{rel}:{line}: ")


def test_untracked_files_are_checked_and_ignored_files_are_not(repo: Path) -> None:
    _add(repo, ".venv/lib/site.py", "# vendored\n")
    _add(repo, "finagents/new.py", "x = 1  # new\n")
    (failure,) = check(repo)
    assert failure.startswith("finagents/new.py:1: ")


def test_failures_steer_the_constraint_into_code_and_the_reason_into_the_commit(
    repo: Path,
) -> None:
    _add(repo, "finagents/why.py", "x = 1\n# why x is one\n")
    (failure,) = check(repo)
    assert failure.endswith(
        "encode the constraint in a name, a type, a test, or a lint rule, "
        "and explain the reason in the commit message"
    )
