"""Official Acc_paper loop + scheme-16 SHARED/OUT_ROOT + criterion inject."""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

HERE = Path(__file__).resolve().parent
import shared_hparams as _sh  # noqa: E402

sys.modules["shared_hparams"] = _sh

from _official_load import HOP100, OFFICIAL, PRE, STEP, load_official  # noqa: E402
from hier_loss import build_criterion  # noqa: E402

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

# Active arm for criterion factory (set by run_arm before main)
ACTIVE_ARM = "S0"

_orig_train_one_fold = _tr.train_one_fold
# 必须在 patch 前保存真正的 CE 类，否则 build_criterion / S0 会递归炸内存
_REAL_CE = nn.CrossEntropyLoss


def train_one_fold(*args, **kwargs):
    """Inject hier loss; forward all kwargs (incl. src_box) to official fold loop."""
    n_outputs = int(kwargs.get("n_outputs", 3))
    arm = ACTIVE_ARM

    def _factory(*_a, **_k):
        # 绝不能在此处再调用已 patch 的 nn.CrossEntropyLoss
        if n_outputs != 3 or arm == "S0":
            return _REAL_CE()
        return build_criterion(arm, n_outputs, real_ce=_REAL_CE)

    nn.CrossEntropyLoss = _factory  # type: ignore[misc, assignment]
    try:
        return _orig_train_one_fold(*args, **kwargs)
    finally:
        nn.CrossEntropyLoss = _REAL_CE


_tr.train_one_fold = train_one_fold
train_one_fold_official = _orig_train_one_fold
run_baseline_main = _tr.run_baseline_main
