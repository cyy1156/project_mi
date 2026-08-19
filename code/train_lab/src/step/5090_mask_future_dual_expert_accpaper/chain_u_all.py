"""U 系列全量五折链（5090 · 无内存守护 · shared_hparams 默认超参）。

顺序（对齐实验方案）：
  必做：U1 → U3 → U2
  组合附报：U13 → U12 → U123

Usage:
  python chain_u_all.py
  python chain_u_all.py --from U3
  python chain_u_all.py --skip-combo          # 仅 U1/U3/U2
  python chain_u_all.py --dry-run
  python chain_u_all.py --num-workers 4       # 显式覆盖（Windows 建议 0）
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
PY = Path(sys.executable)
if (not PY.is_file()) and _VENV_PY.is_file():
    PY = _VENV_PY

STATE = HERE / "u_chain_state.json"
LOG = HERE / "u_chain_all_stdout.log"

from arms_registry import U_COMBO_ORDER, U_SERIES_ORDER, assert_u_arm_flags  # noqa: E402
from shared_hparams import SHARED, shared_as_dict  # noqa: E402


def u_chain_steps(*, include_combo: bool = True) -> list[str]:
    assert_u_arm_flags()
    steps = list(U_SERIES_ORDER)
    if include_combo:
        steps.extend(U_COMBO_ORDER)
    return steps


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


def _run_argv(
    arm: str,
    *,
    max_folds: int,
    num_workers: int,
    batch_train: int,
    max_epochs: int,
    patience: int,
) -> list[str]:
    argv = ["--arm", arm, "--max-folds", str(max_folds)]
    if num_workers >= 0:
        argv += ["--num-workers", str(num_workers)]
    if batch_train > 0:
        argv += ["--batch-train", str(batch_train)]
    if max_epochs > 0:
        argv += ["--max-epochs", str(max_epochs)]
    if patience > 0:
        argv += ["--patience", str(patience)]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(description="5090 U-series full k-fold chain (no mem guard)")
    p.add_argument("--from", dest="from_step", default="", help="断点续跑，如 U3 / U13")
    p.add_argument("--max-folds", type=int, default=0, help="0=五折全量")
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help=">=0 覆盖；默认 -1 由 run_arm 决定（Windows→0，Linux 可用 4）",
    )
    p.add_argument(
        "--batch-train",
        type=int,
        default=0,
        help=">0 覆盖；默认 0=用 shared_hparams.batch_train=%d" % SHARED.batch_train,
    )
    p.add_argument("--max-epochs", type=int, default=0, help=">0 覆盖 shared max_epochs")
    p.add_argument("--patience", type=int, default=0, help=">0 覆盖 shared patience")
    p.add_argument(
        "--skip-combo",
        action="store_true",
        help="仅跑 U1/U3/U2，跳过 U13/U12/U123",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    names = u_chain_steps(include_combo=not args.skip_combo)
    start_i = 0
    if args.from_step:
        if args.from_step not in names:
            raise SystemExit(f"--from must be one of {names}")
        start_i = names.index(args.from_step)

    hp_note = shared_as_dict()
    state = {
        "package": "5090_mask_future_dual_expert_accpaper",
        "mode": "U_SERIES_FULL",
        "device": "RTX 5090",
        "started": datetime.now().isoformat(timespec="seconds"),
        "steps": names,
        "from": args.from_step or names[start_i],
        "max_folds": args.max_folds,
        "hparams": hp_note,
        "run_argv_defaults": {
            "num_workers": args.num_workers,
            "batch_train": args.batch_train or SHARED.batch_train,
            "batch_eval": SHARED.batch_eval,
            "max_epochs": args.max_epochs or SHARED.max_epochs,
            "patience": args.patience or SHARED.patience,
        },
        "done": [],
        "failed": None,
        "python": str(PY),
    }
    _save(state)
    _log(
        f"U-chain start free_gb={_free_gb():.1f} py={PY} "
        f"steps={names[start_i:]} batch={SHARED.batch_train}/{SHARED.batch_eval}"
    )

    if args.dry_run:
        for n in names[start_i:]:
            argv = _run_argv(
                n,
                max_folds=args.max_folds,
                num_workers=args.num_workers,
                batch_train=args.batch_train,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            _log(f"DRY {n} argv={argv}")
        return

    for name in names[start_i:]:
        argv = _run_argv(
            name,
            max_folds=args.max_folds,
            num_workers=args.num_workers,
            batch_train=args.batch_train,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        _log(f"=== begin {name} argv={argv} free_gb={_free_gb():.1f} ===")
        cmd = [str(PY), "-u", str(HERE / "run_arm.py"), *argv]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(HERE))
        dt = time.time() - t0
        if proc.returncode != 0:
            state["failed"] = {
                "step": name,
                "code": proc.returncode,
                "sec": round(dt, 1),
            }
            _save(state)
            _log(f"FAILED {name} code={proc.returncode} sec={dt:.1f}")
            raise SystemExit(proc.returncode)
        state["done"].append({"step": name, "sec": round(dt, 1)})
        _save(state)
        _log(f"=== done {name} sec={dt:.1f} free_gb={_free_gb():.1f} ===")

    state["finished"] = datetime.now().isoformat(timespec="seconds")
    _save(state)
    _log("U-chain ALL DONE")


if __name__ == "__main__":
    main()
