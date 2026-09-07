"""L1e → L1c sequential (5-fold Acc_paper)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = Path(r"D:\cyy\MI\.venv\Scripts\python.exe")


def main() -> None:
    for arm in ("L1e", "L1c"):
        print(f"\n======== launching {arm} ========", flush=True)
        rc = subprocess.call(
            [str(PY), "-u", str(HERE / "run_l_track.py"), "--arm", arm, "--num-workers", "0"],
            cwd=str(HERE),
        )
        if rc != 0:
            print(f"FAIL {arm} rc={rc}", flush=True)
            sys.exit(rc)
        print(f"======== DONE {arm} ========", flush=True)
    print("ALL L1 DONE", flush=True)


if __name__ == "__main__":
    main()
