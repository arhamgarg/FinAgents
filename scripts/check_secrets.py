from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import tomli_w

from scripts.check_whitespace import merge_base, worktree_index

FLAGS = (
    "--redact",
    "--no-banner",
    "--no-color",
    "--log-level",
    "info",
    "--report-format",
    "json",
    "--report-path",
    "-",
)
LOG = re.compile(r"^\S+ (?P<level>INF|WRN) (?P<message>.*)$")
SCANNED = re.compile(r"^(\d+) commits scanned\.$")
FIX = (
    "remove the secret from every commit on the branch (rewrite the branch with git rebase "
    "so no commit contains it) and rotate it with its issuer, because a committed secret "
    "counts as exposed"
)
NO_RENAMES = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "diff.renames",
    "GIT_CONFIG_VALUE_0": "false",
}
MINIMUM_VERSION = "8.30.1"
HOOK = "#!/bin/sh\nexec uv run --locked python -m scripts.check_secrets --staged\n"
INCOMPLETE = (
    "gitleaks did not finish the scan, so the branch is unverified; repair what its output "
    "names (for a corrupt object, run git fsck and restore the object) and rerun"
)


def gitleaks_binary() -> Path:
    found = shutil.which("gitleaks")
    if found is None:
        raise SystemExit(
            f"check_secrets: gitleaks is not on PATH; install gitleaks {MINIMUM_VERSION} or "
            "later with `brew install gitleaks`"
        )
    reported = subprocess.run(
        [found, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    version = tuple(int(part) for part in re.findall(r"\d+", reported)[:3])
    if version < tuple(int(part) for part in MINIMUM_VERSION.split(".")):
        raise SystemExit(
            f"check_secrets: gitleaks {reported} is older than {MINIMUM_VERSION}; "
            "upgrade it with `brew upgrade gitleaks`"
        )
    return Path(found)


def gitleaks_config(repo: Path) -> str:
    tool = tomllib.loads((repo / "pyproject.toml").read_text()).get("tool", {})
    if "gitleaks" not in tool:
        raise SystemExit("check_secrets: pyproject.toml has no [tool.gitleaks] table")
    return tomli_w.dumps(tool["gitleaks"])


def _scan(
    repo: Path, gitleaks: Path, *args: str, env: dict[str, str] | None = None
) -> tuple[list[str], list[str]]:
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory, "gitleaks.toml")
        config.write_text(gitleaks_config(repo))
        result = subprocess.run(
            [str(gitleaks), "git", "--config", str(config), *FLAGS, *args, "."],
            cwd=repo,
            env={**(env or os.environ), **NO_RENAMES},
            capture_output=True,
            text=True,
            check=False,
        )
    messages: list[str] = []
    errors: list[str] = []
    for line in result.stderr.splitlines():
        match = LOG.match(line)
        if match and (match["level"] == "INF" or match["message"].startswith("leaks found")):
            messages.append(match["message"])
        elif line.strip():
            errors.append(line)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = None
    if errors or not isinstance(report, list) or result.returncode != (1 if report else 0):
        raise SystemExit(
            "\n".join(
                [
                    *(f"check_secrets: {error}" for error in errors),
                    f"check_secrets: gitleaks exited with status {result.returncode}",
                    f"check_secrets: {INCOMPLETE}",
                ]
            )
        )
    findings = [
        f"{finding['File']}:{finding['StartLine']}: {finding['RuleID']} in "
        + (f"commit {finding['Commit']}" if finding["Commit"] else "uncommitted changes")
        for finding in report
    ]
    return findings, messages


def _commits_with_text_changes(repo: Path, span: str) -> int:
    log = subprocess.run(
        [
            "git",
            "log",
            "--diff-merges=first-parent",
            "--no-renames",
            "--diff-filter=d",
            "--format=%x00",
            "--numstat",
            span,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sum(
        any(
            added != "-" and int(added) + int(deleted) > 0
            for added, deleted, _ in (line.split("\t", 2) for line in commit.splitlines() if line)
        )
        for commit in log.split("\0")[1:]
    )


def check(repo: Path, base: str, gitleaks: Path) -> list[str]:
    span = f"{merge_base(repo, base)}..HEAD"
    committed, messages = _scan(repo, gitleaks, f"--log-opts=--diff-merges=first-parent {span}")
    scanned = [int(match[1]) for message in messages if (match := SCANNED.match(message))]
    expected = _commits_with_text_changes(repo, span)
    if len(scanned) != 1 or scanned[0] < expected:
        raise SystemExit(
            f"check_secrets: gitleaks reported {scanned or 'no count of'} commits scanned in "
            f"{span}, but {expected} commits there change text; {INCOMPLETE}"
        )
    with worktree_index(repo) as env:
        uncommitted, _ = _scan(repo, gitleaks, "--pre-commit", "--staged", env=env)
    return committed + uncommitted


def check_staged(repo: Path, gitleaks: Path) -> list[str]:
    return _scan(repo, gitleaks, "--pre-commit", "--staged")[0]


def install_hook(repo: Path) -> None:
    path = (
        repo
        / subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks/pre-commit"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if path.is_file() and path.read_text() == HOOK and os.access(path, os.X_OK):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() != HOOK:
        path.replace(path.with_name(f"{path.name}.bak"))
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staged.write_text(HOOK)
    staged.chmod(0o755)
    staged.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate: gitleaks, with the [tool.gitleaks] table of pyproject.toml, finds no "
            "secret in the branch's commits, merge commits included, or in the uncommitted "
            "working tree, untracked files included. It also writes the pre-commit hook that "
            "runs the staged scan."
        )
    )
    parser.add_argument("--base", default="origin/main", help="ref to take the merge base with")
    parser.add_argument(
        "--staged", action="store_true", help="scan only staged changes, as the git hook does"
    )
    args = parser.parse_args(argv)
    repo = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    gitleaks = gitleaks_binary()
    if not args.staged:
        install_hook(repo)
    failures = check_staged(repo, gitleaks) if args.staged else check(repo, args.base, gitleaks)
    for failure in failures:
        print(f"check_secrets: {failure}", file=sys.stderr)
    if failures:
        print(f"check_secrets: {FIX}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
