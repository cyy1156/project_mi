"""路径与常量 · 02 微调（前半训 / 后半评）。"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
TRAIN_LAB = HERE.parents[2]
CODE_ROOT = HERE.parents[3]
REPO_ROOT = HERE.parents[4]

ACCPAPER_WEIGHT_ROOT = TRAIN_LAB / "out" / "baseline_2s_hop100_accpaper"
FT_WEIGHT_ROOT = TRAIN_LAB / "out" / "baseline_game_ft_hop100_accpaper"
SESSIONS_ROOT = REPO_ROOT / "experiment_game" / "data" / "sessions"
DOCS_OUT = REPO_ROOT / "资料" / "伪在线实验" / "02_微调_前半训后半评"

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
    "game_pseudo_online_ft Tw=2s hop=100ms "
    "arm=02_finetune_first_half_train_second_half_eval "
    "finetune_mode=full_model freeze_backbone=false head_only=false "
    "split=trial_order_half "
    "init=bci2a_2s_hop100_balbatch_accpaper "
    "train=game_first_half_windows eval=game_second_half_pseudo_online "
    "early_stop=val_acc_paper max_epochs=300 patience=20 lr=1e-4 "
    "balbatch no_rap no_otta "
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
