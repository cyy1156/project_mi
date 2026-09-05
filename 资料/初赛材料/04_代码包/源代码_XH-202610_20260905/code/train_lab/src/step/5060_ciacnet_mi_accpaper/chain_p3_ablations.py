"""Run P3a → P3b → P3c → P3d sequentially (paper config, 8ch)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = Path(r"D:\cyy\MI\.venv\Scripts\python.exe")
ARMS = ("P3a", "P3b", "P3c", "P3d")


def main() -> None:
    for arm in ARMS:
        print(f"\n======== launching {arm} ========", flush=True)
        rc = subprocess.call(
            [str(PY), "-u", str(HERE / "run_p_track.py"), "--arm", arm],
            cwd=str(HERE),
        )
        if rc != 0:
            print(f"FAIL {arm} rc={rc}", flush=True)
            sys.exit(rc)
        print(f"======== DONE {arm} ========", flush=True)
    print("ALL P3 ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
