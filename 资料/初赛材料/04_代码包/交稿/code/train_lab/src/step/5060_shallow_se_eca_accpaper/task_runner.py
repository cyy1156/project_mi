"""L-style task_runner: official Acc_paper loop + this package SHARED/OUT_ROOT_TAG."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import shared_hparams as _sh  # noqa: E402

sys.modules["shared_hparams"] = _sh

from _official_load import HOP100, OFFICIAL, PRE, STEP, load_official  # noqa: E402

for p in (str(HERE), str(STEP), str(PRE), str(HOP100), str(OFFICIAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import md_fold_detail as _md  # noqa: E402
import trial_metrics as _tm  # noqa: E402
import perf_loader as _pl  # noqa: E402

sys.modules["md_fold_detail"] = _md
sys.modules["trial_metrics"] = _tm
sys.modules["perf_loader"] = _pl

_tr = load_official("task_runner")
run_baseline_main = _tr.run_baseline_main
run_kfold = getattr(_tr, "run_kfold", None)
train_one_fold = getattr(_tr, "train_one_fold", None)
