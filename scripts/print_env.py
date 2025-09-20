import json
import os
import platform
import subprocess
import sys


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"ERR: {exc}"


def main() -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git_commit": sh("git rev-parse --short HEAD"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
