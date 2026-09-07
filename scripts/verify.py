import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    for command in (
        ["uv", "sync", "--locked", "--all-extras"],
        ["uv", "run", "--locked", "ruff", "check", "."],
        ["uv", "run", "--locked", "ruff", "format", "--check", "."],
        ["uv", "run", "--locked", "ty", "check", ".", "--error-on-warning"],
        ["uv", "run", "--locked", "pytest", "-q", "--tb=short"],
    ):
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=root)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
