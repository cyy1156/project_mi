"""hop100 正式读数 run（与 01 实验结果汇总一致；只读）。"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
HOP100_OUT = TRAIN_LAB / "out" / "baseline_2s_hop100"

# model_name -> run stamp under <model>_2s_hop100_balbatch_balacc/bci2a_2s_hop100/
OFFICIAL_RUNS: dict[str, str] = {
    "conformer": "20260804_070726",
    "dbn": "20260804_074123",
    "dbn_raw": "20260804_080802",
    "deep": "20260804_024048",
    "dgcnn": "20260804_075929",
    "dgcnn_raw": "20260804_090545",
    "eegnet": "20260804_005414",
    "eegtcnet": "20260804_061913",
    "gcbnet": "20260804_074751",
    "gcbnet_raw": "20260804_083414",
    "shallow": "20260804_022251",
}

ALL_MODELS = tuple(OFFICIAL_RUNS.keys())


def hop100_run_dir(model_name: str) -> Path:
    stamp = OFFICIAL_RUNS[model_name]
    return (
        HOP100_OUT
        / f"{model_name}_2s_hop100_balbatch_balacc"
        / "bci2a_2s_hop100"
        / f"run_{stamp}"
    )
