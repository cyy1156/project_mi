"""伪在线实验 07：OpenBMI S3 shallow × Stieger 3s/hop100。"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]

OPENBMI_S3_WEIGHT_ROOT = (
    TRAIN_LAB / "out" / "5060_baseline_openbmi_3s_hop100_accpaper"
)
OPENBMI_S3_SHALLOW_RUN = "run_20260821_190504"
FT_WEIGHT_ROOT = TRAIN_LAB / "out" / "stieger_ft_3s_hop100_accpaper"

DATA_TAG = "stieger_3s_hop100"
DATA_DIR = CODE_ROOT / "preprocess_lab" / "out" / "stieger_3s_hop100"

DOCS_07 = REPO_ROOT / "资料" / "伪在线实验" / "07_旁路_OpenBMI_3s滑窗_Stieger零样本"
RESULTS_ROOT = DOCS_07 / "results"

OPENBMI_CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]

SFREQ = 250.0
WIN_SEC = 3.0
HOP_SEC = 0.1
N_TIMES = int(WIN_SEC * SFREQ)  # 750
N_FOLDS = 5

PROTOCOL = (
    "stieger_pseudo_online Tw=3s hop=100ms "
    "arm=07_openbmi_s3_stieger_reproduce_01to06 "
    "model=shallow weights=openbmi_3s_hop100_balbatch_accpaper "
    "data=stieger_3s_hop100 no_channel_remap "
    "gates=H0-H3_online_erd_laterality "
    "ft=cue_order_half full_model balbatch early_stop=acc_paper"
)
