"""方案 25 · 域增广训练 + 增量 FT 配套。"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]

OUT_ROOT_TAG = "5070_aug_3s_accpaper"
OUT_ROOT_TAG_5090 = "5090_aug_3s_accpaper"

# A0 锚点 S3（只读）；G1 在 OpenBMI 上重训
BASELINE_S3_ROOT_5070 = TRAIN_LAB / "out" / "5070_baseline_openbmi_3s_hop100_accpaper"
BASELINE_S3_RUN_5070 = "run_20260822_094942"
BASELINE_S3_ROOT_5090 = TRAIN_LAB / "out" / "5090_alg_incr_3s_hop100_accpaper"
BASELINE_S3_RUN_5090 = "run_20260823_095327"

# 向后兼容
BASELINE_S3_ROOT = BASELINE_S3_ROOT_5070
BASELINE_S3_RUN_DEFAULT = BASELINE_S3_RUN_5070

AUG_WEIGHT_ROOT = TRAIN_LAB / "out" / OUT_ROOT_TAG
AUG_WEIGHT_ROOT_5090 = TRAIN_LAB / "out" / OUT_ROOT_TAG_5090


def aug_weight_root(train_device: str = "5070") -> Path:
    d = (train_device or "5070").strip().lower()
    if d in ("5090", "90"):
        return AUG_WEIGHT_ROOT_5090
    return AUG_WEIGHT_ROOT


def aug_out_root_tag(train_device: str = "5070") -> str:
    d = (train_device or "5070").strip().lower()
    if d in ("5090", "90"):
        return OUT_ROOT_TAG_5090
    return OUT_ROOT_TAG


def _norm_device(train_device: str = "5070") -> str:
    d = (train_device or "5070").strip().lower()
    return "5090" if d in ("5090", "90") else "5070"


def baseline_s3_root(train_device: str = "5070") -> Path:
    if _norm_device(train_device) == "5090":
        return BASELINE_S3_ROOT_5090
    return BASELINE_S3_ROOT_5070


def baseline_s3_run_default(train_device: str = "5070") -> str:
    if _norm_device(train_device) == "5090":
        return BASELINE_S3_RUN_5090
    return BASELINE_S3_RUN_5070


def anchor_s3_openbmi_three(train_device: str = "5070") -> float:
    if _norm_device(train_device) == "5090":
        return ANCHOR_S3_OPENBMI_THREE_5090
    return ANCHOR_S3_OPENBMI_THREE_5070

DOCS_25 = (
    REPO_ROOT
    / "资料"
    / "模型训练"
    / "25_旁路_域增广训练_增量FT配套_openbmi_accpaper"
)
RESULTS_ROOT = DOCS_25 / "results"

# Stieger · 与 07/09 一致
S07_PKG = STEP / "stieger_pseudo_online_3s_hop100"
OTTA_PKG = STEP / "5070_stieger_otta_3s_hop100"
DATA_TAG = "stieger_3s_hop100"
N_TIMES = 750
N_FOLDS = 5

# 只读锚点（macro Three Acc_paper）
ANCHOR_S07_ZEROSHOT_THREE = 0.4198
ANCHOR_09_A0_EVAL_HALF_THREE = 0.4119
ANCHOR_S3_OPENBMI_THREE_5070 = 0.5873
ANCHOR_S3_OPENBMI_THREE_5090 = 0.5839
ANCHOR_S3_OPENBMI_THREE = ANCHOR_S3_OPENBMI_THREE_5070

# 增量 FT 爬坡 checkpoint（cue 数，按时间序前半累计）
INCREMENTAL_K_LIST = (0, 10, 20, 40, 80, -1)  # -1 = 全前半

PROTOCOL = (
    "scheme25 aug+incremental_ft Tw=3s hop=100ms "
    "openbmi_train=stieger_eval noz_unified device=5070"
)
