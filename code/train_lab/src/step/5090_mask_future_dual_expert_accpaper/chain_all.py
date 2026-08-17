"""一键全链：A0 → A1 → P0 → A2 → P1 → B* → P2 → C*

Usage:
  python chain_all.py
  python chain_all.py --from P1
  python chain_all.py --max-folds 1
  python chain_all.py --dry-run
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
# .../MI/code/train_lab/src/step/<pkg> → parents[4] == MI
REPO_ROOT = HERE.parents[4]
_VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
PY = _VENV_PY if _VENV_PY.is_file() else Path(sys.executable)

STATE = HERE / "chain_state.json"
LOG = HERE / "chain_all_stdout.log"

from arms_registry import CHAIN_ORDER, chain_steps  # noqa: E402


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
    p = argparse.ArgumentParser(description="5090 mask-future dual-expert full chain")
    p.add_argument("--from", dest="from_step", default="", help="resume from arm id")
    p.add_argument("--max-folds", type=int, default=0, help="传给每臂；0=五折")
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--include-skipped",
        action="store_true",
        help="包含 A1_600 / L1 等默认跳过臂",
    )
    args = p.parse_args()

    names = chain_steps(include_skipped=args.include_skipped)
    # 校验 from
    start_i = 0
    if args.from_step:
        if args.from_step not in names:
            # 也允许从完整 CHAIN_ORDER 名恢复
            if args.from_step in CHAIN_ORDER:
                names = chain_steps(include_skipped=True)
            if args.from_step not in names:
                raise SystemExit(f"--from must be one of {names}")
        start_i = names.index(args.from_step)

    state = {
        "package": "5090_mask_future_dual_expert_accpaper",
        "device": "RTX 5090",
        "started": datetime.now().isoformat(timespec="seconds"),
        "steps": names,
        "from": args.from_step or names[start_i],
        "max_folds": args.max_folds,
        "done": [],
        "failed": None,
        "python": str(PY),
    }
    _save(state)
    _log(f"chain start free_gb={_free_gb():.1f} py={PY} steps={names}")

    if args.dry_run:
        for n in names[start_i:]:
            _log(f"DRY {n}")
        return

    for name in names[start_i:]:
        argv = ["--arm", name, "--max-folds", str(args.max_folds)]
        if args.num_workers >= 0:
            argv += ["--num-workers", str(args.num_workers)]
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
