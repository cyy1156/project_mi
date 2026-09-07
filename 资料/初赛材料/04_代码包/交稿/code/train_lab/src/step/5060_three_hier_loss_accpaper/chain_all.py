"""Scheme 16 full chain: Three S0→H1→H2→H3 (5-fold) then Task T0 (CE, 5-fold).

H3 currently matches H2 terms (trial_cons not wired); still run for scheme completeness.
T1 (focal) not in this chain until implemented.

Usage:
  python chain_all.py
  python chain_all.py --from H2_three
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
_VENV_PY = Path(r"D:\cyy\MI\.venv\Scripts\python.exe")
PY = _VENV_PY if _VENV_PY.is_file() else Path(sys.executable)
STATE = HERE / "chain_state.json"
LOG = HERE / "chain_all_stdout.log"

# (name, argv after run_arm.py)
STEPS: list[tuple[str, list[str]]] = [
    ("S0_three", ["--arm", "S0", "--max-folds", "0", "--num-workers", "0"]),
    ("H1_three", ["--arm", "H1", "--max-folds", "0", "--num-workers", "0"]),
    ("H2_three", ["--arm", "H2", "--max-folds", "0", "--num-workers", "0"]),
    ("H3_three", ["--arm", "H3", "--max-folds", "0", "--num-workers", "0"]),
    (
        "T0_task",
        [
            "--arm",
            "S0",
            "--with-task",
            "--skip-three",
            "--max-folds",
            "0",
            "--num-workers",
            "0",
        ],
    ),
]


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _save(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _free_gb() -> float:
    try:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return float(m.ullAvailPhys) / (1024**3)
    except Exception:
        return -1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_step", default="", help="resume from step name")
    args = p.parse_args()

    names = [n for n, _ in STEPS]
    start_i = 0
    if args.from_step:
        if args.from_step not in names:
            raise SystemExit(f"--from must be one of {names}")
        start_i = names.index(args.from_step)

    state = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "steps": names,
        "from": args.from_step or names[start_i],
        "done": [],
        "failed": None,
        "current": None,
    }
    _save(state)
    _log(f"scheme16 chain start from={state['from']} steps={names[start_i:]}")
    _log(f"free_ram_gb={_free_gb():.2f}")

    for name, argv in STEPS[start_i:]:
        free = _free_gb()
        _log(f"before {name}: free_ram_gb={free:.2f}")
        if 0 <= free < 4.0:
            _log(
                f"WARN low RAM before {name}; sleeping 30s — close other apps if OOM recurs"
            )
            time.sleep(30)
        state["current"] = name
        _save(state)
        cmd = [str(PY), "-u", str(HERE / "run_arm.py"), *argv]
        _log(f"RUN {name}: {' '.join(cmd)}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(HERE))
        dt = time.time() - t0
        if r.returncode != 0:
            state["failed"] = {"step": name, "code": r.returncode}
            state["current"] = None
            _save(state)
            _log(f"FAIL {name} exit={r.returncode} after {dt/3600:.2f}h")
            raise SystemExit(r.returncode)
        state["done"].append(name)
        state["current"] = None
        _save(state)
        _log(f"OK {name} in {dt/3600:.2f}h")

    state["finished"] = datetime.now().isoformat(timespec="seconds")
    _save(state)
    _log("ALL DONE scheme16")


if __name__ == "__main__":
    main()
