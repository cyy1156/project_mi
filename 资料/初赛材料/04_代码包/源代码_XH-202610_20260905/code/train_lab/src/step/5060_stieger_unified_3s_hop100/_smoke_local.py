"""本地冒烟：3s 回归 + 2s 零样本 + OTTA S1。"""

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
    _run([PY, "eval_zeroshot.py", "--tw", "3s", "--subjects", subj, "--smoke"])
    _run([PY, "eval_zeroshot.py", "--tw", "2s", "--subjects", subj, "--smoke"])
    _run(
        [
            PY,
            "eval_otta.py",
            "--arms",
            "A0,A3,B0,B3",
            "--subjects",
            subj,
            "--smoke",
        ]
    )
    print("[smoke ok]", flush=True)


if __name__ == "__main__":
    main()
