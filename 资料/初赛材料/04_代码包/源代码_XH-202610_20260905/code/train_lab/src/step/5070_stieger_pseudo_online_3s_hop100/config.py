"""伪在线实验 08：OpenBMI S3 shallow × Stieger 3s/hop100 · **RTX 5070**。

协议同臂 08；权重 / FT out / 文档目录独立为 5070，禁止写入 5060 / 07 results。
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]

OPENBMI_S3_WEIGHT_ROOT = (
    TRAIN_LAB / "out" / "5070_baseline_openbmi_3s_hop100_accpaper"
)
# 尚无正式五折时留空；resolve 会取最新完整 run
OPENBMI_S3_SHALLOW_RUN = ""
FT_WEIGHT_ROOT = TRAIN_LAB / "out" / "5070_stieger_ft_3s_hop100_accpaper"

DATA_TAG = "stieger_3s_hop100"
DATA_DIR = CODE_ROOT / "preprocess_lab" / "out" / "stieger_3s_hop100"

DOCS_08 = (
    REPO_ROOT / "资料" / "伪在线实验" / "08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070"
)
RESULTS_ROOT = DOCS_08 / "results"

OPENBMI_CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]

SFREQ = 250.0
WIN_SEC = 3.0
HOP_SEC = 0.1
N_TIMES = int(WIN_SEC * SFREQ)  # 750
N_FOLDS = 5

PROTOCOL = (
    "stieger_pseudo_online Tw=3s hop=100ms "
    "arm=08_openbmi_s3_stieger_reproduce_01to06_5070 "
    "model=shallow weights=5070_openbmi_3s_hop100_balbatch_accpaper "
    "data=stieger_3s_hop100 no_channel_remap "
    "gates=H0-H3_online_erd_laterality "
    "ft=cue_order_half full_model balbatch early_stop=acc_paper device=5070"
)

# 兼容从 07 包拷贝的脚本里仍写 DOCS_07 / RESULTS 的引用
DOCS_07 = DOCS_08
