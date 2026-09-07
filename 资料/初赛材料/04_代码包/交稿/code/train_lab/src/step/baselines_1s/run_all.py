"""一次性串跑 1s 基线（单库；不做 merged）。

用法（在 baselines_1s/ 目录下）：
  python run_all.py --data bci2a_1s
  python run_all.py --data stieger_1s
  python run_all.py --data both
  python run_all.py --data bci2a_1s --models eegnet,dbn_raw,gcbnet_raw
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
    # Self_development_model：TemporalEncoder + 原始时域
    "dbn_raw",
    "gcbnet_raw",
    "dgcnn_raw",
)

DATA_CHOICES = ("bci2a_1s", "stieger_1s", "both")


def _clean_name(s: str) -> str:
    return s.strip().strip("\"'").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="串跑 baselines_1s（BalAcc+balbatch；含 *_raw）")
    p.add_argument("--data", default="bci2a_1s", choices=DATA_CHOICES)
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
    names = [_clean_name(x) for x in args.models.split(",") if _clean_name(x)]
    if not names:
        raise SystemExit("未指定任何模型")
    bad = [n for n in names if n not in ALL_MODELS]
    if bad:
        raise SystemExit(f"未知模型: {bad}; 可选: {', '.join(ALL_MODELS)}")

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    data_tags = (
        ("bci2a_1s", "stieger_1s") if args.data == "both" else (args.data,)
    )
    failed: list[str] = []
    for data_tag in data_tags:
        for name in names:
            code = run_one(data_tag, name, extra, args.continue_on_error)
            if code != 0:
                failed.append(f"{data_tag}/{name}")

    if failed:
        raise SystemExit(f"完成但有失败: {failed}")
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
