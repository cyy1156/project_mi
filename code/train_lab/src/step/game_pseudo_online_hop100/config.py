"""路径与常量。"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]  # .../train_lab
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]
ACCPAPER = STEP / "baselines_2s_hop100_accpaper"
WEIGHT_ROOT = TRAIN_LAB / "out" / "baseline_2s_hop100_accpaper"
SESSIONS_ROOT = REPO_ROOT / "experiment_game" / "data" / "sessions"
DOCS_OUT = REPO_ROOT / "资料" / "伪在线实验" / "01_不微调_零样本"

DEFAULT_SESSIONS = (
    "sub02_ses01_20260723_180607",
    "sub03_ses01_20260723_185153",
)

TOP5_MODELS = (
    "shallow",
    "deep",
    "conformer",
    "eegnet",
    "eegtcnet",
)

PROTOCOL = (
    "game_pseudo_online Tw=2s hop=100ms "
    "arm=01_no_finetune_zeroshot "
    "seg=acquire_mi|rest concat=per_subject_trial_order "
    "win=in_segment_only no_cross_boundary "
    "models=shallow,deep,conformer,eegnet,eegtcnet "
    "select_list=trialmaj_task_acc_paper_top5 "
    "weights=bci2a_2s_hop100_balbatch_accpaper "
    "weight_pkg=baselines_2s_hop100_accpaper "
    "metrics=segment_acc_paper_main + balacc_maj + window_balacc_aux "
    "no_finetune no_game_retrain no_rap no_otta"
)

SFREQ = 250.0
WIN_SEC = 2.0
HOP_SEC = 0.1
N_TIMES = int(WIN_SEC * SFREQ)  # 500
HOP_SAMPLES = int(HOP_SEC * SFREQ)  # 25
MIN_SEG_SEC = WIN_SEC
BASELINE_SEC = 0.5
N_FOLDS = 5
