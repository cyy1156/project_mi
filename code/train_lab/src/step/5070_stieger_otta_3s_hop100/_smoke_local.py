"""本地冒烟：单被试 · fold0 · v1.2 关键臂快速路径。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=HERE)


def main() -> None:
    subj = "S1"
    _run(
        [
            PY,
            "eval_ab.py",
            "--arms",
            "A0,A2,A3,B0,B2,B3",
            "--subjects",
            subj,
            "--smoke",
        ]
    )
    _run([PY, "eval_c1.py", "--subjects", subj, "--smoke", "--tasks", "three"])
    print("[smoke ok]", flush=True)


if __name__ == "__main__":
    main()
