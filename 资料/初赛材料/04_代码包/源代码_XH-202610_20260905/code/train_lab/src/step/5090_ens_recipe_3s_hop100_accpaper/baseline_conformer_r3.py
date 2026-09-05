"""方案 26 · R3 conformer 五折（R1 配方 + wd 5e-4）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
sys.path.insert(0, str(PKG24))
sys.path.insert(0, str(HERE))

from patch_recipe import install_recipe  # noqa: E402

install_recipe("R3")

from baseline_conformer import build_model  # noqa: E402
from task_runner import run_baseline_main  # noqa: E402


if __name__ == "__main__":
    run_baseline_main(
        model_name="conformer_recipe_r3",
        build_model=build_model,
        input_kind="time",
        structure_note="Conformer · scheme26 R3 recipe",
        extra_meta={
            "conformer": True,
            "accpaper": True,
            "experiment": 26,
            "arm": "R3",
            "device": "5090",
            "win_sec": 3.0,
            "hop_sec": 0.1,
        },
    )
