"""伪在线实验 10：5060 统一包 · 3s 复现 + 2s 对照 + OTTA v1.2。"""

from __future__ import annotations

from pathlib import Path

from tw_profile import TwProfile, build_profiles

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]

DOCS_10 = (
    REPO_ROOT
    / "资料"
    / "伪在线实验"
    / "10_旁路_OpenBMI_3s滑窗_Stieger_5060_复现与OTTA"
)
RESULTS_ROOT = DOCS_10 / "results"

TW_PROFILES = build_profiles(
    code_root=CODE_ROOT, train_lab=TRAIN_LAB, docs_10=DOCS_10
)
_ACTIVE: TwProfile = TW_PROFILES["3s"]

# --- 动态窗长（apply_tw 后更新）---
DATA_TAG: str = _ACTIVE.data_tag
DATA_DIR: Path = _ACTIVE.data_dir
N_TIMES: int = _ACTIVE.n_times
WIN_SEC: float = _ACTIVE.win_sec
HOP_SEC: float = _ACTIVE.hop_sec
OPENBMI_WEIGHT_ROOT: Path = _ACTIVE.openbmi_weight_root
OPENBMI_SHALLOW_RUN: str = _ACTIVE.openbmi_shallow_run
FT_WEIGHT_ROOT: Path = _ACTIVE.ft_weight_root

# --- OTTA Part III（仅 3s）---
OTTATW = TW_PROFILES["3s"]
OPENBMI_S3_WEIGHT_ROOT = OTTATW.openbmi_weight_root
OPENBMI_S3_SHALLOW_RUN = OTTATW.openbmi_shallow_run
FT_STAMP_DEFAULT = "20260822_153300"

OPENBMI_NOZ_DIR = CODE_ROOT / "preprocess_lab" / "out" / "openbmi_3s_hop100_noz"
OPENBMI_MAT_GLOB = REPO_ROOT / "DATA" / "openbmi" / "sess*_subj*_EEG_MI.mat"
CACHE_ROOT = TRAIN_LAB / "out" / "stieger_otta_3s_hop100_accpaper"
EA_REF_SRC_CACHE = CACHE_ROOT / "ea_ref_src_cov.npy"

OPENBMI_CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
SFREQ = 250.0
N_FOLDS = 5

PROTOCOL_VERSION = "v1.2"
ADABN_VERSION = "v1.2"
ADABN_PREDICT_FIRST = True
INPUT_PIPELINE = "noz_unified"

ANCHOR_A0_READONLY = {"task": (0.5789, 0.0501), "three": (0.4198, 0.0585)}
ANCHOR_B0_READONLY = {"task": (0.7868, 0.0797), "three": (0.6590, 0.0929)}
ANCHOR_A0 = ANCHOR_A0_READONLY
ANCHOR_B0 = ANCHOR_B0_READONLY

PROTOCOL_OTTA = (
    "stieger_otta Tw=3s hop=100ms "
    "arm=10_unified_otta_v1.2 "
    "model=shallow weights=5060_openbmi_3s "
    "ft=shallow_stieger_ft_half_balbatch_accpaper "
    "data=stieger_3s_hop100 eval_half=causal_stream "
    "input=noz_unified ea=A:src|B:cal_whiten "
    "adabn=v1.2_predict_first_update device=5060"
)

S07_PKG = STEP / "stieger_pseudo_online_3s_hop100"


def apply_tw(tw: str) -> TwProfile:
    """切换 2s/3s；更新模块级 DATA_DIR / N_TIMES / 权重根。"""
    global _ACTIVE, DATA_TAG, DATA_DIR, N_TIMES, WIN_SEC, HOP_SEC
    global OPENBMI_WEIGHT_ROOT, OPENBMI_SHALLOW_RUN, FT_WEIGHT_ROOT
    key = tw.strip().lower()
    if key not in TW_PROFILES:
        raise ValueError(f"--tw 须为 2s|3s，得到 {tw!r}")
    _ACTIVE = TW_PROFILES[key]
    DATA_TAG = _ACTIVE.data_tag
    DATA_DIR = _ACTIVE.data_dir
    N_TIMES = _ACTIVE.n_times
    WIN_SEC = _ACTIVE.win_sec
    HOP_SEC = _ACTIVE.hop_sec
    OPENBMI_WEIGHT_ROOT = _ACTIVE.openbmi_weight_root
    OPENBMI_SHALLOW_RUN = _ACTIVE.openbmi_shallow_run
    FT_WEIGHT_ROOT = _ACTIVE.ft_weight_root
    return _ACTIVE


def active_profile() -> TwProfile:
    return _ACTIVE


def protocol_repro() -> str:
    p = _ACTIVE
    return (
        f"stieger_pseudo_online Tw={p.tw} hop=100ms "
        f"arm=10_unified_repro_{p.arm_zeroshot} "
        f"model=shallow weights=5060_openbmi_{p.tw} "
        f"data={p.data_tag} device=5060"
    )


def results_zeroshot_dir() -> Path:
    return RESULTS_ROOT / _ACTIVE.zeroshot_subdir


def results_ft_half_dir() -> Path:
    return RESULTS_ROOT / _ACTIVE.ft_half_subdir
