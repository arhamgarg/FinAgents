from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_whitespace import check, main


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin",
        },
    )


def _commit(repo: Path, rel: str, text: str) -> None:
    (repo / rel).write_text(text)
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", rel)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _commit(tmp_path, "legacy.txt", "trailing on main \n")
    _git(tmp_path, "checkout", "-q", "-b", "task")
    return tmp_path


def test_whitespace_already_on_the_base_branch_passes(repo: Path) -> None:
    _commit(repo, "clean.txt", "clean\n")
    assert check(repo, "main") == []


def test_trailing_whitespace_in_a_branch_commit_fails_with_the_line(repo: Path) -> None:
    _commit(repo, "notes.txt", "fine\nbad line \n")
    assert check(repo, "main") == ["notes.txt:2: trailing whitespace."]


def test_every_kind_of_error_is_listed_without_its_line(repo: Path) -> None:
    _commit(repo, "+notes.txt", "a \n<<<<<<< ours\n \tz\n\n")
    assert check(repo, "main") == [
        "+notes.txt:1: trailing whitespace.",
        "+notes.txt:2: leftover conflict marker",
        "+notes.txt:3: space before tab in indent.",
        "+notes.txt:4: new blank line at EOF.",
    ]


def test_whitespace_that_main_removed_after_the_branch_point_passes(repo: Path) -> None:
    _commit(repo, "notes.txt", "clean\n")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "legacy.txt", "fixed on main\n")
    _git(repo, "checkout", "-q", "task")
    assert check(repo, "main") == []


def test_an_earlier_branch_commit_counts_when_head_adds_nothing(repo: Path) -> None:
    _commit(repo, "notes.txt", "bad \n")
    _commit(repo, "other.txt", "clean\n")
    assert check(repo, "main")


def test_unstaged_edit_to_a_tracked_file_fails(repo: Path) -> None:
    _commit(repo, "notes.txt", "fine\n")
    (repo / "notes.txt").write_text("fine\n\tbad\t\n")
    assert any(failure.startswith("notes.txt:2:") for failure in check(repo, "main"))


def test_untracked_file_fails_and_stays_untracked(repo: Path) -> None:
    (repo / "new.txt").write_text("x\n\n")
    assert any("new.txt:" in failure for failure in check(repo, "main"))
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert status == "?? new.txt\n"


def test_main_reports_the_supported_fix(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _commit(repo, "notes.txt", "bad \n")
    monkeypatch.chdir(repo)
    assert main(["--base", "main"]) == 1
    assert "commit the fix" in capsys.readouterr().err
