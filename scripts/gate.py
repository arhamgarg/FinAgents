import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic

UV = ("uv", "run", "--locked")
CHECKS: tuple[str, ...] = ("whitespace", "file_size", "comments", "naming", "secrets")


def run(*args: str) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, check=True)


@contextmanager
def stage(name: str) -> Iterator[None]:
    start = monotonic()
    try:
        yield
    finally:
        print(f"{name}: {monotonic() - start:.2f}s", flush=True)


def setup() -> None:
    with stage("setup"):
        run("uv", "sync", "--locked")


def checks() -> None:
    with stage("checks"):
        run(*UV, "ruff", "check", ".")
        run(*UV, "ruff", "format", "--check", ".")
        run(*UV, "ty", "check", ".", "--error-on-warning")
        run(*UV, "lint-imports", "--no-logo")
        for name in CHECKS:
            run(*UV, "python", "-m", f"scripts.check_{name}")


def tests() -> None:
    with stage("tests"):
        run(*UV, "pytest", "-q", "--tb=short", "--cov=finagents", "--cov-report=term-missing")


def main() -> None:
    setup()
    checks()
    tests()


def entry() -> None:
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)


if __name__ == "__main__":
    entry()
