from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

KEBAB = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
PYTHON = re.compile(r"[a-z_][a-z0-9_]*")
FOLDER = re.compile(r"\.?[a-z0-9]+(?:[-_][a-z0-9]+)*")
UPPERCASE_DOCUMENTS = {"README.md", "PROJECT.md"}


def words(stem: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", stem)
    return [word.lower() for word in re.split(r"[-_\s.]+", spaced) if word]


def kebab(stem: str) -> str:
    return "-".join(words(stem))


def snake(stem: str) -> str:
    return "_".join(words(stem))


def _split(name: str) -> tuple[str, str]:
    stem, dot, rest = name.partition(".")
    return stem, f"{dot}{rest}"


def _python_name(stem: str) -> str:
    name = snake(stem)
    return name if PYTHON.fullmatch(name) else f"m_{name}"


def _folder_problem(parts: tuple[str, ...], packages: frozenset[str]) -> str | None:
    for depth, folder in enumerate(parts[:-1]):
        prefix = "/".join(parts[: depth + 1])
        if prefix in packages:
            if not PYTHON.fullmatch(folder):
                return f"name Python package folders in snake_case: {_python_name(folder)}"
        elif not FOLDER.fullmatch(folder):
            return f"name folders in lowercase kebab-case or snake_case: {kebab(folder)}"
    return None


def _file_problem(parts: tuple[str, ...]) -> str | None:
    name = parts[-1]
    stem, suffix = _split(name)
    if name.endswith(".py"):
        if not PYTHON.fullmatch(stem):
            return f"name Python modules in snake_case: {_python_name(stem)}{suffix}"
    elif name.endswith(".md") and name not in UPPERCASE_DOCUMENTS and not KEBAB.fullmatch(stem):
        return f"name Markdown files in kebab-case: {kebab(stem)}{suffix}"
    return None


def problem(path: str, packages: frozenset[str] = frozenset()) -> str | None:
    parts = PurePosixPath(path).parts
    return _folder_problem(parts, packages) or _file_problem(parts)


def tracked() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted({path for path in listed.split("\0") if path and Path(path).is_file()})


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Gate: file and folder names follow their language's case. Python "
        "modules and package folders (those with __init__.py) use snake_case, Markdown "
        "uses kebab-case except README.md and PROJECT.md, and other folders "
        "are lowercase."
    ).parse_args(argv)
    paths = tracked()
    packages = frozenset(
        str(PurePosixPath(path).parent) for path in paths if path.endswith("/__init__.py")
    )
    failures = [f"{path}: {message}" for path in paths if (message := problem(path, packages))]
    for failure in failures:
        print(f"check_naming: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
