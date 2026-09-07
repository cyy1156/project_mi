"""方案 14 全臂串行：A1 → B1 → A2 → B2 → S0。

形状冒烟已通过，跳过 A0/B0（可用 --with-smoke 打开）。
A1/B1 = fold0；A2/B2/S0 = 五折。全部 Task+Three。

每臂结束后写 chain_state.json，便于中断后续跑。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = Path(r"D:\cyy\MI\.venv\Scripts\python.exe")
LOG = HERE / "chain_all_stdout.log"
STATE = HERE / "chain_state.json"


def _append(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg if msg.endswith("\n") else msg + "\n")
        f.flush()


def _save_state(arm: str, status: str, remaining: list[str]) -> None:
    STATE.write_text(
        json.dumps(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "last_arm": arm,
                "status": status,
                "remaining": remaining,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(arm: str, extra: list[str] | None, remaining_after: list[str]) -> None:
    cmd = [str(PY), "-u", str(HERE / "run_arm.py"), "--arm", arm, "--num-workers", "0"]
    if extra:
        cmd.extend(extra)
    line = f"\n======== [{datetime.now().isoformat(timespec='seconds')}] launching {arm} ========\n"
    print(line, flush=True)
    _append(line)
    _save_state(arm, "running", [arm, *remaining_after])

    # 子进程单独写日志；创建_new_process_group 降低被父会话连带杀掉的概率
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    with open(LOG, "a", encoding="utf-8") as f:
        rc = subprocess.call(
            cmd,
            cwd=str(HERE),
            stdout=f,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    if rc != 0:
        msg = f"FAIL {arm} rc={rc}\n"
        print(msg, flush=True)
        _append(msg)
        _save_state(arm, f"fail:{rc}", [arm, *remaining_after])
        sys.exit(rc)
    done = f"======== DONE {arm} ========\n"
    print(done, flush=True)
    _append(done)
    _save_state(arm, "done", remaining_after)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--with-smoke", action="store_true", help="先跑 A0/B0（2 ep）")
    p.add_argument(
        "--from-arm",
        default="",
        help="从指定臂开始（含），如 B2 跳过已完成臂",
    )
    args = p.parse_args()

    seq: list[tuple[str, list[str]]] = []
    if args.with_smoke:
        seq.append(("A0", ["--max-epochs", "2", "--patience", "2"]))
        seq.append(("B0", ["--max-epochs", "2", "--patience", "2"]))
    seq.extend(
        [
            ("A1", []),
            ("B1", []),
            ("A2", []),
            ("B2", []),
            ("S0", []),
        ]
    )

    if args.from_arm:
        names = [a for a, _ in seq]
        if args.from_arm not in names:
            raise SystemExit(f"--from-arm must be one of {names}")
        i = names.index(args.from_arm)
        seq = seq[i:]

    header = (
        f"\n#### chain_all start {datetime.now().isoformat(timespec='seconds')} "
        f"arms={[a for a, _ in seq]} ####\n"
    )
    print(header, flush=True)
    _append(header)

    for i, (arm, extra) in enumerate(seq):
        remaining_after = [a for a, _ in seq[i + 1 :]]
        run(arm, extra, remaining_after)

    end = f"ALL DONE {datetime.now().isoformat(timespec='seconds')}\n"
    print(end, flush=True)
    _append(end)
    _save_state("ALL", "all_done", [])


if __name__ == "__main__":
    main()
