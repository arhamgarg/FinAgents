from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

WITH_LINE = re.compile(r"^.+:\d+: (?!new blank line at EOF\.$).+\.$")
FIX = (
    "fix the whitespace errors listed above (trailing whitespace, a space before a tab in "
    "the indent, blank lines at the end of a file, or a conflict marker) and commit the fix"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def merge_base(repo: Path, base: str) -> str:
    return _git(repo, "merge-base", "HEAD", base).strip()


@contextmanager
def worktree_index(repo: Path) -> Iterator[dict[str, str]]:
    index = Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index").strip())
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch, "index"))}
        if index.is_file():
            shutil.copyfile(index, env["GIT_INDEX_FILE"])
        subprocess.run(["git", "add", "-A"], cwd=repo, env=env, check=True, capture_output=True)
        yield env


def _diff_check(repo: Path, *args: str, env: dict[str, str] | None = None) -> list[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", "diff", "--check", "--no-color", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 2):
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    lines = iter(result.stdout.split("\n"))
    failures: list[str] = []
    for header in lines:
        if header:
            failures.append(header)
        if WITH_LINE.match(header):
            next(lines, None)
    return failures


def check(repo: Path, base: str) -> list[str]:
    committed = _diff_check(repo, merge_base(repo, base), "HEAD")
    with worktree_index(repo) as env:
        uncommitted = _diff_check(repo, "--cached", "HEAD", env=env)
    return committed + uncommitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate: the branch's commits and the uncommitted working tree, untracked files "
            "included, may not add whitespace errors that git diff --check reports."
        )
    )
    parser.add_argument("--base", default="origin/main", help="ref to take the merge base with")
    args = parser.parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    failures = check(repo, args.base)
    for failure in failures:
        print(f"check_whitespace: {failure}", file=sys.stderr)
    if failures:
        print(f"check_whitespace: {FIX}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
