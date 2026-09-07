"""单模型入口（兼容）：python train_one.py --model eegnet

推荐直接：python baseline_eegnet.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

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


def main() -> None:
    p = argparse.ArgumentParser(description="2s/hop100 Acc_paper 重训（转发到 baseline_*.py）")
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    args, unknown = p.parse_known_args()
    script = HERE / f"baseline_{args.model}.py"
    if not script.is_file():
        raise SystemExit(f"缺少 {script}")
    cmd = [sys.executable, str(script), *unknown]
    raise SystemExit(subprocess.call(cmd, cwd=str(HERE)))


if __name__ == "__main__":
    main()
