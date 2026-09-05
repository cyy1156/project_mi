"""扫描 session EEG 电极平线 / 饱和（Cz/CPz 等）。

用法:
  python experiment_game/tools/scan_session_electrode.py experiment_game/data/sessions/fnz_ws03_...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from experiment_game.experiment.session_electrode import scan_session_electrodes


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描 session 电极质量")
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args()
    rep = scan_session_electrodes(args.session_dir.resolve())
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
