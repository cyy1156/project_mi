"""串跑多个单模型基线（可选工具）。

日常对照优先直接跑某一个：
  python baseline_eegnet.py --data merged_2s

本脚本只是按顺序 subprocess 调用各 baseline_*.py，不替代一模型一脚本约定。

PyCharm 运行参数示例（不要给模型名再加引号）：
  --data merged_2s
  --models eegnet,shallow,deep,eegtcnet,conformer
  --models dbn,gcbnet,dgcnn
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent

ALL_MODELS = (
    "eegnet",
    "shallow",
    "deep",
    "eegtcnet",
    "conformer",
    "dbn",
    "gcbnet",
    "dgcnn",
)


def _clean_name(s: str) -> str:
    """去掉外壳引号/空白，兼容误写成 'eegnet' 或 \"eegnet\"。"""
    return s.strip().strip("\"'").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="依次串跑多个单模型基线脚本")
    p.add_argument(
        "--data",
        default="merged_2s",
        choices=("merged_2s", "bci2a_2s", "stieger_2s"),
        help="传给各 baseline_*.py 的 --data",
    )
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"逗号分隔模型名；可选: {', '.join(ALL_MODELS)}",
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="额外参数原样转发给每个脚本（写在 --extra 之后）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = [_clean_name(x) for x in args.models.split(",") if _clean_name(x)]
    if not names:
        raise SystemExit("未指定任何模型")
    bad = [n for n in names if n not in ALL_MODELS]
    if bad:
        raise SystemExit(f"未知模型: {bad}; 可选: {', '.join(ALL_MODELS)}")

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    for name in names:
        script = DIR / f"baseline_{name}.py"
        if not script.is_file():
            raise SystemExit(f"脚本不存在: {script}")
        cmd = [sys.executable, str(script), "--data", args.data, *extra]
        print(f"\n===== {name} =====", flush=True)
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(DIR))
        if r.returncode != 0:
            raise SystemExit(f"{name} 失败，exit={r.returncode}")

    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
