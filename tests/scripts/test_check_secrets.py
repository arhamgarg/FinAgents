from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts.check_secrets import HOOK, check, check_staged, gitleaks_binary, main

ROOT = Path(__file__).resolve().parents[2]
KEY = "AKIA" + "QYLPMN5HHHFPZAM2"


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
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", rel)


@pytest.fixture(scope="module")
def gitleaks() -> Path:
    return gitleaks_binary()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    _commit(tmp_path, "readme.txt", "hello\n")
    _git(tmp_path, "checkout", "-q", "-b", "task")
    return tmp_path


def test_clean_branch_and_tree_pass(repo: Path, gitleaks: Path) -> None:
    _commit(repo, "config.txt", "region = us-east-1\n")
    assert check(repo, "main", gitleaks) == []


def test_secret_committed_then_deleted_on_the_branch_fails(repo: Path, gitleaks: Path) -> None:
    _commit(repo, "config.txt", f"aws_access_key_id = {KEY}\n")
    _commit(repo, "config.txt", "aws_access_key_id = redacted\n")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD~1"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert check(repo, "main", gitleaks) == [f"config.txt:1: aws-access-token in commit {head}"]


def test_findings_print_no_email_address_or_secret(
    repo: Path, gitleaks: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _commit(repo, "config.txt", f"aws_access_key_id = {KEY}\n")
    monkeypatch.chdir(repo)
    assert main(["--base", "main"]) == 1
    output = capsys.readouterr()
    printed = output.out + output.err
    assert "aws-access-token" in printed
    assert "rotate" in printed
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", printed)
    assert KEY not in printed


def test_staged_and_untracked_secrets_fail(repo: Path, gitleaks: Path) -> None:
    (repo / "staged.txt").write_text(f"key = {KEY}\n")
    _git(repo, "add", "staged.txt")
    (repo / "untracked.txt").write_text(f"key = {KEY}\n")
    failures = check(repo, "main", gitleaks)
    assert any("staged.txt" in failure for failure in failures)
    assert any("untracked.txt" in failure for failure in failures)


def test_unstaged_edit_to_a_tracked_file_fails(repo: Path, gitleaks: Path) -> None:
    (repo / "readme.txt").write_text(f"hello {KEY}\n")
    assert any("readme.txt" in failure for failure in check(repo, "main", gitleaks))


def _rev_parse(repo: Path, ref: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_secret_added_by_a_merge_commit_fails(repo: Path, gitleaks: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "side", "main")
    _commit(repo, "side.txt", "side\n")
    _git(repo, "checkout", "-q", "task")
    _commit(repo, "task.txt", "task\n")
    _git(repo, "merge", "-q", "--no-commit", "side")
    (repo / "merged.txt").write_text(f"key = {KEY}\n")
    _git(repo, "add", "merged.txt")
    _git(repo, "commit", "-q", "-m", "merge side")
    merge = _rev_parse(repo, "HEAD")
    assert check(repo, "main", gitleaks) == [f"merged.txt:1: aws-access-token in commit {merge}"]


def test_corrupt_object_in_the_branch_fails_the_scan(repo: Path, gitleaks: Path) -> None:
    _commit(repo, "notes.txt", "notes\n")
    blob = _rev_parse(repo, "HEAD:notes.txt")
    loose = repo / ".git" / "objects" / blob[:2] / blob[2:]
    loose.chmod(0o644)
    loose.write_bytes(b"corrupt")
    with pytest.raises(SystemExit, match="did not finish the scan"):
        check(repo, "main", gitleaks)


def test_scan_that_skips_commits_without_an_error_fails(repo: Path, tmp_path: Path) -> None:
    _commit(repo, "notes.txt", "notes\n")
    with pytest.raises(SystemExit, match="but 1 commits there change text"):
        check(repo, "main", _stub(tmp_path, "[]", "1:00AM INF 0 commits scanned.", 0))


def test_commit_past_the_rename_limit_is_scanned(repo: Path, gitleaks: Path) -> None:
    _git(repo, "config", "diff.renameLimit", "1")
    for number in range(3):
        (repo / f"old{number}.txt").write_text(f"old {number}\n")
    _commit(repo, "old0.txt", "old 0\n")
    for number in range(3):
        (repo / f"old{number}.txt").unlink()
        (repo / f"new{number}.txt").write_text(f"new {number}\n")
    _commit(repo, "new0.txt", f"key = {KEY}\n")
    commit = _rev_parse(repo, "HEAD")
    assert check(repo, "main", gitleaks) == [f"new0.txt:1: aws-access-token in commit {commit}"]


def _stub(tmp_path: Path, stdout: str, stderr: str, status: int) -> Path:
    stub = tmp_path / "stub-gitleaks"
    stub.write_text(f"#!/bin/sh\necho '{stdout}'\ncat >&2 <<'EOF'\n{stderr}\nEOF\nexit {status}\n")
    stub.chmod(0o755)
    return stub


@pytest.mark.parametrize(
    ("stdout", "stderr", "status"),
    [
        ("[]", "1:00AM INF 1 commits scanned.", 1),
        ("{}", "1:00AM INF 1 commits scanned.", 0),
        ("[]", "1:00AM WRN skipping a large file\n1:00AM INF 1 commits scanned.", 0),
    ],
    ids=["exit status without findings", "report that is not a list", "unexpected warning"],
)
def test_gitleaks_output_that_does_not_prove_a_clean_scan_fails(
    repo: Path, tmp_path: Path, stdout: str, stderr: str, status: int
) -> None:
    _commit(repo, "notes.txt", "notes\n")
    with pytest.raises(SystemExit, match="did not finish the scan"):
        check(repo, "main", _stub(tmp_path, stdout, stderr, status))


def test_allowlisted_fixture_path_passes(repo: Path, gitleaks: Path) -> None:
    (repo / "tests/fixtures/angelone").mkdir(parents=True)
    _commit(repo, "tests/fixtures/angelone/generateSession.json", f"key = {KEY}\n")
    assert check(repo, "main", gitleaks) == []


def test_staged_scan_reports_staged_secrets_only(repo: Path, gitleaks: Path) -> None:
    (repo / "staged.txt").write_text(f"key = {KEY}\n")
    _git(repo, "add", "staged.txt")
    (repo / "untracked.txt").write_text(f"key = {KEY}\n")
    assert check_staged(repo, gitleaks) == ["staged.txt:1: aws-access-token in uncommitted changes"]


def test_the_allowlist_comes_from_pyproject(repo: Path, gitleaks: Path) -> None:
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    with pytest.raises(SystemExit, match=r"no \[tool\.gitleaks\] table"):
        check(repo, "main", gitleaks)


def _path_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    directory = tmp_path / "bin"
    directory.mkdir()
    if version is not None:
        binary = directory / "gitleaks"
        binary.write_text(f"#!/bin/sh\necho {version}\n")
        binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory))


def test_missing_gitleaks_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path_with(tmp_path, monkeypatch, None)
    with pytest.raises(SystemExit, match="brew install gitleaks"):
        gitleaks_binary()


@pytest.mark.parametrize(
    ("version", "accepted"), [("8.29.9", False), ("8.30.1", True), ("v9.0.0", True)]
)
def test_gitleaks_older_than_the_minimum_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, accepted: bool
) -> None:
    _path_with(tmp_path, monkeypatch, version)
    if accepted:
        assert gitleaks_binary() == tmp_path / "bin" / "gitleaks"
    else:
        with pytest.raises(SystemExit, match="brew upgrade gitleaks"):
            gitleaks_binary()


def test_an_ambient_gitleaks_config_cannot_replace_the_project_config(
    repo: Path, gitleaks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    permissive = tmp_path / "permissive.toml"
    permissive.write_text("[[allowlists]]\npaths = ['.*']\n")
    monkeypatch.setenv("GITLEAKS_CONFIG", str(permissive))
    monkeypatch.setenv("GITLEAKS_CONFIG_TOML", permissive.read_text())
    _commit(repo, "config.txt", f"aws_access_key_id = {KEY}\n")
    (repo / "staged.txt").write_text(f"key = {KEY}\n")
    _git(repo, "add", "staged.txt")
    assert any("config.txt" in failure for failure in check(repo, "main", gitleaks))
    assert check_staged(repo, gitleaks) == ["staged.txt:1: aws-access-token in uncommitted changes"]


def test_the_staged_entry_point_fails_on_a_staged_secret(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(repo)
    assert main(["--staged"]) == 0
    (repo / "staged.txt").write_text(f"key = {KEY}\n")
    _git(repo, "add", "staged.txt")
    assert main(["--staged"]) == 1
    assert "staged.txt:1: aws-access-token" in capsys.readouterr().err


def test_the_gate_entry_point_installs_an_executable_staged_scan_hook(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = repo / ".git/hooks/pre-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.chdir(repo)
    assert main(["--base", "main"]) == 0
    assert main(["--base", "main"]) == 0
    assert hook.read_text() == HOOK
    assert hook.stat().st_mode & 0o111
    assert (hook.parent / "pre-commit.bak").read_text() == "#!/bin/sh\nexit 0\n"
    assert [path.name for path in hook.parent.iterdir() if path.name.endswith(".tmp")] == []
