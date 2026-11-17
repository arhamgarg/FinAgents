from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_file_size import BUDGET, check


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


def _write(repo: Path, rel: str, lines: int) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"x = {i}\n" for i in range(lines)))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, "src/finagents/big.py", BUDGET + 100)
    _write(tmp_path, "src/finagents/small.py", 20)
    _write(tmp_path, "tests/test_big.py", BUDGET + 500)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "task")
    return tmp_path


def test_clean_tree_passes_even_with_oversized_existing_files(repo: Path) -> None:
    assert check(repo, "main") == []


def test_growing_an_oversized_file_fails_with_counts(repo: Path) -> None:
    _write(repo, "src/finagents/big.py", BUDGET + 101)
    (failure,) = check(repo, "main")
    assert "src/finagents/big.py" in failure
    assert f"{BUDGET + 101} lines" in failure
    assert f"was {BUDGET + 100}" in failure


def test_shrinking_an_oversized_file_passes_while_still_over_budget(repo: Path) -> None:
    _write(repo, "src/finagents/big.py", BUDGET + 50)
    assert check(repo, "main") == []


def test_new_tracked_file_over_budget_fails_and_at_budget_passes(repo: Path) -> None:
    _write(repo, "src/finagents/new_over.py", BUDGET + 1)
    _write(repo, "src/finagents/new_at.py", BUDGET)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add")
    failures = check(repo, "main")
    assert [f for f in failures if "new_over.py" in f]
    assert not [f for f in failures if "new_at.py" in f]


def test_untracked_new_file_over_budget_fails(repo: Path) -> None:
    _write(repo, "src/finagents/untracked.py", BUDGET + 1)
    (failure,) = check(repo, "main")
    assert "src/finagents/untracked.py" in failure


def test_files_outside_src_and_non_python_are_ignored(repo: Path) -> None:
    _write(repo, "tests/test_big.py", BUDGET + 600)
    (repo / "src/finagents/notes.md").write_text("line\n" * (BUDGET + 10))
    assert check(repo, "main") == []


def test_small_file_growing_within_budget_passes(repo: Path) -> None:
    _write(repo, "src/finagents/small.py", BUDGET)
    assert check(repo, "main") == []


def test_editing_an_oversized_file_without_changing_its_length_passes(repo: Path) -> None:
    path = repo / "src/finagents/big.py"
    path.write_text(path.read_text().replace("x = 7\n", "y = 7\n"))
    assert check(repo, "main") == []


def test_deleting_a_source_file_passes(repo: Path) -> None:
    (repo / "src/finagents/small.py").unlink()
    assert check(repo, "main") == []


def test_crossing_the_budget_names_the_crossing_not_growth(repo: Path) -> None:
    _write(repo, "src/finagents/small.py", BUDGET + 1)
    (failure,) = check(repo, "main")
    assert "takes it past" in failure
    assert "may not grow" not in failure


def test_renamed_oversized_file_is_measured_against_its_old_path(repo: Path) -> None:
    (repo / "src/app").mkdir()
    _git(repo, "mv", "src/finagents/big.py", "src/app/moved.py")
    assert check(repo, "main") == []
    _write(repo, "src/app/moved.py", BUDGET + 101)
    (failure,) = check(repo, "main")
    assert f"was {BUDGET + 100}" in failure
