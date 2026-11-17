import subprocess

import pytest

from scripts.check_naming import main, problem

PACKAGES = frozenset({"finagents", "finagents/DataFlows", "finagents/2fa", "scripts"})


@pytest.mark.parametrize(
    "path",
    [
        "finagents/dataflows/fundamentals_client.py",
        "finagents/__init__.py",
        "README.md",
        "PROJECT.md",
        "docs/guide/run-pipeline.md",
        "tests/fixtures/angelone/generateSession.json",
        "tests/fixtures/nse/allIndices.json",
        ".github/workflows/verify.yml",
        "scripts/uat-ticker-batch.txt",
        "LICENSE",
    ],
)
def test_names_in_their_language_case_pass(path):
    assert problem(path, PACKAGES) is None


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("scripts/check-naming.py", "snake_case: check_naming.py"),
        ("finagents/dataStore.py", "snake_case: data_store.py"),
        ("finagents/2fa.py", "snake_case: m_2fa.py"),
        ("finagents/DataFlows/cache.py", "snake_case: data_flows"),
        ("finagents/2fa/store.py", "snake_case: m_2fa"),
        ("docs/Run_Pipeline.md", "kebab-case: run-pipeline.md"),
        ("docs/ARCHIVED.md", "kebab-case: archived.md"),
        ("docs/Guide/a.md", "snake_case: guide"),
    ],
)
def test_a_misnamed_path_names_the_expected_name(path, expected):
    message = problem(path, PACKAGES)
    assert message is not None
    assert message.endswith(f" {expected}")


def test_untracked_files_are_checked(tmp_path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/mod.py").write_text("VALUE = 1\n")
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0

    (tmp_path / "pkg/badName.py").write_text("VALUE = 2\n")
    assert main([]) == 1
    assert "pkg/badName.py: name Python modules in snake_case: bad_name.py" in (
        capsys.readouterr().err
    )
