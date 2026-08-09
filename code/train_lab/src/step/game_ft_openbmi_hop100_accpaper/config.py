"""路径与常量 · 05 OpenBMI shallow 前半训 / 后半评。"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]

OPENBMI_WEIGHT_ROOT = TRAIN_LAB / "out" / "5060_baseline_openbmi_2s_hop100_accpaper"
FT_WEIGHT_ROOT = TRAIN_LAB / "out" / "openbmi_game_ft_hop100_accpaper"
SESSIONS_ROOT = REPO_ROOT / "experiment_game" / "data" / "sessions"
DOCS_OUT = REPO_ROOT / "资料" / "伪在线实验" / "05_旁路_OpenBMI_前半微调后半评"
PSEUDO_ONLINE = STEP / "game_pseudo_online_hop100"

DEFAULT_SESSIONS = (
    "sub02_ses01_20260723_180607",
    "sub03_ses01_20260723_185153",
)

MODELS = ("shallow",)

# 与 04 / channel_fe 冻结一致
OPENBMI_SHALLOW_RUN = "run_20260807_135828"

PROTOCOL = (
    "game_pseudo_online_ft Tw=2s hop=100ms "
    "arm=05_openbmi_finetune_first_half_train_second_half_eval "
    "finetune_mode=full_model freeze_backbone=false head_only=false "
    "split=trial_order_half "
    "init=openbmi_2s_hop100_balbatch_accpaper "
    "channel_remap=game_to_openbmi "
    "train=game_first_half_windows eval=game_second_half_pseudo_online "
    "early_stop=val_acc_paper max_epochs=300 patience=20 lr=1e-4 "
    "balbatch no_rap no_otta model=shallow_only "
    "metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux"
)

SFREQ = 250.0
WIN_SEC = 2.0
HOP_SEC = 0.1
N_TIMES = int(WIN_SEC * SFREQ)  # 500
HOP_SAMPLES = int(HOP_SEC * SFREQ)  # 25
MIN_SEG_SEC = WIN_SEC
BASELINE_SEC = 0.5
N_FOLDS = 5
