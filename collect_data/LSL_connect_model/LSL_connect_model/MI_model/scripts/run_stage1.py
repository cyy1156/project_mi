"""一键运行 Stage 1（front / back 二分类）全流程。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STAGE = 1


def run(script: str, extra_args: list[str] | None = None) -> None:
    path = SCRIPTS / script
    cmd = [sys.executable, str(path), "--stage", str(STAGE)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'=' * 60}\n>>> {script}\n{'=' * 60}")
    subprocess.run(cmd, check=True)


def main() -> None:
    run("01_inspect_data.py")
    run("02_build_dataset.py")
    run("03_train_csp_svm.py")
    run("04_eval_offline.py")
    print("\nStage 1 全流程完成。")


if __name__ == "__main__":
    main()
