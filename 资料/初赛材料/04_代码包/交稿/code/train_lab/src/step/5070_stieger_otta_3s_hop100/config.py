"""伪在线实验 09：OTTA（EA + AdaBN）· Stieger 3s/hop100。

权重与 07 同源（5060 S3 + S07 FT ckpt）；结果写入 09 文档目录。
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]

# 与 07 冻结一致：5060 S3 + S07 FT（不重训）
OPENBMI_S3_WEIGHT_ROOT = (
    TRAIN_LAB / "out" / "5060_baseline_openbmi_3s_hop100_accpaper"
)
OPENBMI_S3_SHALLOW_RUN = "run_20260821_190504"
FT_WEIGHT_ROOT = TRAIN_LAB / "out" / "stieger_ft_3s_hop100_accpaper"
FT_STAMP_DEFAULT = "20260822_153300"

DATA_TAG = "stieger_3s_hop100"
DATA_DIR = CODE_ROOT / "preprocess_lab" / "out" / "stieger_3s_hop100"

OPENBMI_NOZ_DIR = CODE_ROOT / "preprocess_lab" / "out" / "openbmi_3s_hop100_noz"
OPENBMI_MAT_GLOB = REPO_ROOT / "DATA" / "openbmi" / "sess*_subj*_EEG_MI.mat"

DOCS_09 = (
    REPO_ROOT
    / "资料"
    / "伪在线实验"
    / "09_旁路_OpenBMI_3s滑窗_OTTA_EA_AdaBN_Stieger"
)
RESULTS_ROOT = DOCS_09 / "results"
CACHE_ROOT = TRAIN_LAB / "out" / "stieger_otta_3s_hop100_accpaper"
EA_REF_SRC_CACHE = CACHE_ROOT / "ea_ref_src_cov.npy"

OPENBMI_CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]

SFREQ = 250.0
WIN_SEC = 3.0
HOP_SEC = 0.1
N_TIMES = int(WIN_SEC * SFREQ)  # 750
N_FOLDS = 5

# 只读锚点（07 全量 trial；OTTA 臂为后半评，配对 Δ 用同跑 A0/B0）
ANCHOR_A0_READONLY = {"task": (0.5789, 0.0501), "three": (0.4198, 0.0585)}
ANCHOR_B0_READONLY = {"task": (0.7868, 0.0797), "three": (0.6590, 0.0929)}
ANCHOR_A0 = ANCHOR_A0_READONLY
ANCHOR_B0 = ANCHOR_B0_READONLY

ADABN_VERSION = "v1.2"
PROTOCOL_VERSION = "v1.2"
ADABN_PREDICT_FIRST = True  # 严格因果：先预测后更新 running stats
INPUT_PIPELINE = "noz_unified"

PROTOCOL = (
    "stieger_otta Tw=3s hop=100ms "
    "arm=09_otta_ea_adabn_stieger_v1.2 "
    "model=shallow weights=openbmi_3s_hop100_balbatch_accpaper(5060) "
    "ft=shallow_stieger_ft_half_balbatch_accpaper "
    "data=stieger_3s_hop100 eval_half=causal_stream "
    "input=noz_unified ea=A:src|B:cal_whiten "
    "adabn=v1.2_predict_first_update "
    "device=5070"
)

# 复用 07 包路径（data / infer / weights 等）
S07_PKG = STEP / "stieger_pseudo_online_3s_hop100"
