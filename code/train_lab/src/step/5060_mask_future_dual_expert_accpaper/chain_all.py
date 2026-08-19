"""一键链 · 5060 低内存：默认每臂 fold0。

主线门控（默认）：A0 → A1 → P0 → A2 → P1 → P2
全消融请加 --full-chain，或改用 5090 包五折。

Usage:
  python chain_all.py
  python chain_all.py --from P1
  python chain_all.py --full-chain
  python chain_all.py --max-folds 0
  python chain_all.py --dry-run
  cd D:\cyy\MI\code\train_lab\src\step\5060_mask_future_dual_expert_accpaper
  D:\cyy\MI\.venv\Scripts\python.exe -u chain_all.py --full-chain --max-folds 0
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
REPO_ROOT = HERE.parents[4]
_VENV_PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
PY = _VENV_PY if _VENV_PY.is_file() else Path(sys.executable)

STATE = HERE / "chain_state.json"
LOG = HERE / "chain_all_stdout.log"

from arms_registry import ARMS, CHAIN_ORDER, chain_steps  # noqa: E402

# 5060 旁路默认只跑主线阶梯（省时间/内存）；B/C 用 --full-chain
GATE_ORDER = ["A0_ref", "A0", "A1", "P0", "A2", "P1", "P2"]


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
    p = argparse.ArgumentParser(description="5060 mask-future dual-expert chain (low-mem)")
    p.add_argument("--from", dest="from_step", default="", help="resume from arm id")
    p.add_argument(
        "--max-folds",
        type=int,
        default=1,
        help="传给每臂；默认 1=fold0；0=五折（本机慎用）",
    )
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--full-chain",
        action="store_true",
        help="跑完整 CHAIN_ORDER（含 B/C）；默认仅 GATE_ORDER 主线",
    )
    p.add_argument(
        "--include-skipped",
        action="store_true",
        help="包含 A1_600 / L1 等默认跳过臂（需同时 --full-chain）",
    )
    args = p.parse_args()

    if args.full_chain:
        names = chain_steps(include_skipped=args.include_skipped)
    else:
        names = [n for n in GATE_ORDER if n in ARMS and not ARMS[n].skip_in_auto_chain]

    start_i = 0
    if args.from_step:
        if args.from_step not in names:
            if args.from_step in CHAIN_ORDER:
                names = chain_steps(include_skipped=True)
            if args.from_step not in names:
                raise SystemExit(f"--from must be one of {names}")
        start_i = names.index(args.from_step)

    free = _free_gb()
    if 0 <= free < 3.5:
        _log(f"WARN free_phys={free:.2f}G < 3.5G；建议关多余程序后再跑")

    state = {
        "package": "5060_mask_future_dual_expert_accpaper",
        "device": "RTX 5060 Laptop",
        "started": datetime.now().isoformat(timespec="seconds"),
        "steps": names,
        "from": args.from_step or names[start_i],
        "max_folds": args.max_folds,
        "full_chain": bool(args.full_chain),
        "done": [],
        "failed": None,
        "python": str(PY),
    }
    _save(state)
    _log(
        f"chain start free_gb={free:.1f} py={PY} max_folds={args.max_folds} "
        f"steps={names}"
    )

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
