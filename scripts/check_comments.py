from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

PRAGMA = re.compile(
    r"#\s*(noqa: [A-Z]+[0-9]+(, ?[A-Z]+[0-9]+)*"
    r"|isort: (off|on|skip_file)"
    r"|pragma: no cover"
    r"|type: ?ignore\[[\w, -]+\]"
    r"|ty: ?ignore\[[\w, -]+\]"
    r"|fmt: (on|off|skip))"
)
HASH_FILES = re.compile(r"\.(sh|toml|ya?ml)$")
SQL_FILES = re.compile(r"\.sql$")
FIX = (
    "encode the constraint in a name, a type, a test, or a lint rule, "
    "and explain the reason in the commit message"
)
DOC_FIX = f"move command help to help=; {FIX}"


def _files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(p for p in set(out.split("\0")) if p and (repo / p).is_file())


def _python(path: str, src: str) -> list[str]:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
        tree = ast.parse(src, path)
    except tokenize.TokenError as exc:
        return [f"{path}:{exc.args[1][0]}: cannot parse; fix the syntax first"]
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: cannot parse; fix the syntax first"]
    failures = []
    for token in tokens:
        if token.type != tokenize.COMMENT or PRAGMA.fullmatch(token.string.strip()):
            continue
        if token.start[0] == 1 and token.string.startswith("#!"):
            continue
        failures.append(f"{path}:{token.start[0]}: comment; {FIX}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            failures.append(f"{path}:{first.lineno}: docstring; {DOC_FIX}")
    return failures


def _lines(path: str, src: str, markers: tuple[str, ...]) -> list[str]:
    failures = []
    for number, line in enumerate(src.splitlines(), 1):
        stripped = line.lstrip()
        if number == 1 and stripped.startswith("#!"):
            continue
        if stripped.startswith(markers):
            failures.append(f"{path}:{number}: comment; {FIX}")
    return failures


def check(repo: Path) -> list[str]:
    failures: list[str] = []
    for path in _files(repo):
        if path.endswith(".py"):
            failures += _python(path, (repo / path).read_text())
        elif HASH_FILES.search(path):
            failures += _lines(path, (repo / path).read_text(), ("#",))
        elif SQL_FILES.search(path):
            failures += _lines(path, (repo / path).read_text(), ("--", "/*"))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail on any comment or docstring; only shebangs and bare tool pragmas pass."
    )
    parser.parse_args(argv)
    failures = check(Path.cwd())
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
