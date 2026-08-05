"""一次性串跑固定窗 Cue+2~4s 基线（仅 BCI2a；不做 merged / Stieger）。

用法（在 baselines_fixed_2s/ 目录下）：
  python run_all.py
  python run_all.py --models eegnet,shallow,deep
  python run_all.py --continue-on-error
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
    "dbn_raw",
    "gcbnet_raw",
    "dgcnn_raw",
)

DATA_TAG = "bci2a_2s"


def _clean_name(s: str) -> str:
    return s.strip().strip("\"'").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="串跑 baselines_fixed_2s（BCI2a 固定窗；BalAcc+balbatch）")
    p.add_argument("--data", default=DATA_TAG, choices=(DATA_TAG,), help="仅 bci2a_2s")
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"逗号分隔；可选: {', '.join(ALL_MODELS)}",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单模型失败时继续下一个（默认遇错退出）",
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="额外参数原样转发（写在 --extra 之后）",
    )
    return p.parse_args()


def run_one(data_tag: str, name: str, extra: list[str], continue_on_error: bool) -> int:
    script = DIR / f"baseline_{name}.py"
    if not script.is_file():
        print(f"脚本不存在: {script}", flush=True)
        return 1
    cmd = [sys.executable, str(script), "--data", data_tag, *extra]
    print(f"\n===== {data_tag} / {name} =====", flush=True)
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(DIR))
    if r.returncode != 0:
        msg = f"{data_tag}/{name} 失败，exit={r.returncode}"
        if continue_on_error:
            print("WARN:", msg, flush=True)
            return r.returncode
        raise SystemExit(msg)
    return 0


def main() -> None:
    args = parse_args()
    models = [_clean_name(x) for x in args.models.split(",") if _clean_name(x)]
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise SystemExit(f"未知模型: {bad}；可选: {ALL_MODELS}")
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    codes = []
    for name in models:
        codes.append(run_one(args.data, name, extra, args.continue_on_error))
    if any(c != 0 for c in codes):
        raise SystemExit(1)
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
