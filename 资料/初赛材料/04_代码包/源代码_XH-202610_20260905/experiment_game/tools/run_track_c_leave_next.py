"""Exp34 轨 C 入口：封装 Leave-Next 5 轮爬坡。

示例：
  python -m experiment_game.tools.run_track_c_leave_next --subject syj0828 --dry-run
  python -m experiment_game.tools.run_track_c_leave_next --subject syj0828 --execute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "run_leave_next_e1f_task_ramp.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp34 track C · Leave-Next R0–R5")
    ap.add_argument("--subject", action="append", default=None)
    ap.add_argument("--all-core", action="store_true", help="传给底层 --all（syj+fnz）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("extra", nargs="*", help="透传到底层脚本的额外参数")
    args = ap.parse_args()

    cmd = [sys.executable, str(TOOL)]
    if args.all_core:
        cmd.append("--all")
    if args.subject:
        for s in args.subject:
            cmd.extend(["--subject", s])
    cmd.extend(args.extra)

    print("=== Exp34 轨 C · 自采 Leave-Next ===")
    print("R0 零样本 → R1…R5 共 5 次采后 FT · all4+force · F5")
    print("不进指定集 60 分主行；登记 结果登记表.md 表2")
    print("命令:", " ".join(cmd))

    if args.dry_run or not args.execute:
        print("(dry-run) 未执行。确认参数后加 --execute")
        return 0
    if not TOOL.is_file():
        raise SystemExit(f"missing {TOOL}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
