"""串跑十一模型试次级复评（只读 hop100 权重）。

用法：
  python run_all_reeval.py
  python run_all_reeval.py --models eegnet,shallow --continue-on-error
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from official_runs import ALL_MODELS, OFFICIAL_RUNS

DIR = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="串跑 hop100 试次级多数票复评")
    p.add_argument(
        "--models",
        default=",".join(OFFICIAL_RUNS.keys()),
        help=f"逗号分隔；可选: {', '.join(ALL_MODELS)}",
    )
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-three", action="store_true")
    args = p.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    bad = [m for m in models if m not in OFFICIAL_RUNS]
    if bad:
        raise SystemExit(f"未知模型: {bad}")

    codes = []
    for name in models:
        cmd = [sys.executable, str(DIR / "reeval_kfold.py"), "--model", name]
        if args.skip_three:
            cmd.append("--skip-three")
        print(f"\n===== reeval {name} =====", flush=True)
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(DIR))
        codes.append(r.returncode)
        if r.returncode != 0 and not args.continue_on_error:
            raise SystemExit(f"{name} 失败 exit={r.returncode}")
        if r.returncode != 0:
            print(f"WARN: {name} exit={r.returncode}", flush=True)

    if any(c != 0 for c in codes):
        raise SystemExit(1)
    print("\n全部复评完成。", flush=True)


if __name__ == "__main__":
    main()
