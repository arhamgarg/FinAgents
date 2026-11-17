from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BUDGET = 500
SOURCE_ROOTS = ("src",)
SOURCE_PREFIXES = tuple(f"{root}/" for root in SOURCE_ROOTS)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _changed_source_files(repo: Path, merge_base: str) -> dict[str, str | None]:
    files: dict[str, str | None] = {}
    fields = iter(
        _git(repo, "diff", "--name-status", "--find-renames", "-z", merge_base).split("\0")
    )
    for status in fields:
        if not status:
            continue
        base_path = path = next(fields)
        if status.startswith(("R", "C")):
            path = next(fields)
        if not path.endswith(".py") or status == "D" or not path.startswith(SOURCE_PREFIXES):
            continue
        files[path] = None if status == "A" else f"{merge_base}:{base_path}"
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_ROOTS)
    for path in filter(None, untracked.split("\0")):
        if path.endswith(".py"):
            files[path] = None
    return files


def check(repo: Path, base: str) -> list[str]:
    merge_base = _git(repo, "merge-base", "HEAD", base).strip()
    failures: list[str] = []
    for path, base_blob in sorted(_changed_source_files(repo, merge_base).items()):
        lines = _line_count((repo / path).read_text())
        if lines <= BUDGET:
            continue
        if base_blob is None:
            failures.append(
                f"{path}: new file has {lines} lines; the budget is {BUDGET}. "
                "Split it along the concepts it owns before committing."
            )
            continue
        was = _line_count(_git(repo, "show", base_blob))
        if was <= BUDGET:
            failures.append(
                f"{path}: {lines} lines (was {was}); the change takes it past the {BUDGET}-line "
                "budget. Split the file in its own commit before the behaviour change."
            )
        elif lines > was:
            failures.append(
                f"{path}: {lines} lines (was {was}); files over {BUDGET} lines may not grow. "
                "Split the file in its own commit before the behaviour change."
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate: a change may not leave a source file over the line budget "
            "or grow one already over it."
        )
    )
    parser.add_argument("--base", default="origin/main", help="ref to take the merge base with")
    args = parser.parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    failures = check(repo, args.base)
    for failure in failures:
        print(f"check_file_size: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
