"""Scheme 16 · 5090 full chain: Three S0→H1→H2→H3 (5-fold) then Task T0 (CE, 5-fold).

H3 currently matches H2 terms (trial_cons not wired); still run for scheme completeness.

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
# 5090 机可用系统/conda python；优先本仓库 .venv，否则当前解释器
_VENV_PY = Path(r"D:\cyy\MI\.venv\Scripts\python.exe")
PY = _VENV_PY if _VENV_PY.is_file() else Path(sys.executable)
STATE = HERE / "chain_state.json"
LOG = HERE / "chain_all_stdout.log"

# 全量五折；workers 交给 shared_hparams 默认（5090=4），此处不强制 0
STEPS: list[tuple[str, list[str]]] = [
    ("S0_three", ["--arm", "S0", "--max-folds", "0"]),
    ("H1_three", ["--arm", "H1", "--max-folds", "0"]),
    ("H2_three", ["--arm", "H2", "--max-folds", "0"]),
    ("H3_three", ["--arm", "H3", "--max-folds", "0"]),
    (
        "T0_task",
        [
            "--arm",
            "S0",
            "--with-task",
            "--skip-three",
            "--max-folds",
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
    p = argparse.ArgumentParser(description="5090 scheme16 full chain")
    p.add_argument("--from", dest="from_step", default="", help="resume from step name")
    args = p.parse_args()

    names = [n for n, _ in STEPS]
    start_i = 0
    if args.from_step:
        if args.from_step not in names:
            raise SystemExit(f"--from must be one of {names}")
        start_i = names.index(args.from_step)

    state = {
        "package": "5090_three_hier_loss_accpaper",
        "device": "RTX 5090",
        "started": datetime.now().isoformat(timespec="seconds"),
        "steps": names,
        "from": args.from_step or names[start_i],
        "done": [],
        "failed": None,
        "python": str(PY),
    }
    _save(state)
    _log(f"chain start package=5090 free_gb={_free_gb():.1f} py={PY}")

    for name, argv in STEPS[start_i:]:
        _log(f"=== begin {name} argv={argv} free_gb={_free_gb():.1f} ===")
        cmd = [str(PY), "-u", str(HERE / "run_arm.py"), *argv]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(HERE))
        dt = time.time() - t0
        if proc.returncode != 0:
            state["failed"] = {"step": name, "code": proc.returncode, "sec": round(dt, 1)}
            _save(state)
            _log(f"FAILED {name} code={proc.returncode} sec={dt:.1f}")
            raise SystemExit(proc.returncode)
        state["done"].append({"step": name, "sec": round(dt, 1)})
        _save(state)
        _log(f"=== done {name} sec={dt:.1f} free_gb={_free_gb():.1f} ===")

    state["finished"] = datetime.now().isoformat(timespec="seconds")
    _save(state)
    _log("chain ALL DONE")


if __name__ == "__main__":
    main()
