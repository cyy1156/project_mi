"""串跑 OpenBMI Acc_paper 全 11 模型。

用法：
  python run_all.py
  python run_all.py --models eegnet,shallow --continue-on-error
  python run_all.py --smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent

ALL_MODELS = (
    #"shallow",
    #"deep",
    #"conformer",
    #"eegnet",
    #"eegtcnet",
    "gcbnet",
    "dgcnn",
    "dbn",
    "dbn_raw",
    "gcbnet_raw",
    "dgcnn_raw",
)


def _clean_name(s: str) -> str:
    return s.strip().strip("\"'").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="串跑 baselines_openbmi_2s_hop100_accpaper")
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"逗号分隔；可选: {', '.join(ALL_MODELS)}",
    )
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-three", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟：max-folds=1 max-epochs=2 patience=2",
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="额外参数原样转发（写在 --extra 之后）",
    )
    return p.parse_args()


def run_one(name: str, extra: list[str], continue_on_error: bool) -> int:
    script = DIR / f"baseline_{name}.py"
    if not script.is_file():
        print(f"脚本不存在: {script}", flush=True)
        return 1
    cmd = [sys.executable, str(script), "--data", "openbmi_2s_hop100", *extra]
    print(f"\n===== openbmi_accpaper / {name} =====", flush=True)
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(DIR))
    if r.returncode != 0:
        msg = f"{name} 失败，exit={r.returncode}"
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
    if args.skip_three:
        extra.append("--skip-three")
    if args.smoke:
        extra.extend(["--max-folds", "1", "--max-epochs", "2", "--patience", "2"])

    failed: list[str] = []
    for name in names:
        code = run_one(name, extra, args.continue_on_error)
        if code != 0:
            failed.append(name)
    if failed:
        raise SystemExit(f"完成但有失败: {failed}")
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
