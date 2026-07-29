"""串跑五个单模型基线（可选工具）。

日常对照优先直接跑某一个：
  python baseline_eegnet.py --data merged_2s

本脚本只是按顺序 subprocess 调用各 baseline_*.py，不替代一模型一脚本约定。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
ALL_MODELS = ("eegnet", "shallow", "deep", "eegtcnet", "conformer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="依次运行 baselines_single 下五个 baseline_*.py")
    p.add_argument(
        "--data",
        default="merged_2s",
        help="传给每个 baseline：merged_2s | bci2a_2s | stieger_2s",
    )
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="逗号分隔子集，例如 eegnet,shallow",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    names = [x.strip() for x in args.models.split(",") if x.strip()]
    unknown = [n for n in names if n not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"未知模型: {unknown}；可选: {', '.join(ALL_MODELS)}")

    print(f"data={args.data} models={names}", flush=True)
    for name in names:
        script = DIR / f"baseline_{name}.py"
        if not script.is_file():
            raise SystemExit(f"缺少脚本: {script}")
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run(
            [sys.executable, str(script), "--data", args.data],
            cwd=DIR,
        )
        if r.returncode != 0:
            raise SystemExit(f"{name} failed: {r.returncode}")

    print("\nall done", flush=True)


if __name__ == "__main__":
    main()
