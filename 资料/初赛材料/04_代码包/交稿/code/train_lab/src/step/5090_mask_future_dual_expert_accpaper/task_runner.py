"""Official Acc_paper loop + scheme-17 OUT_ROOT / 5090 hparams（A0_ref 等）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _paths import HOP100, OFFICIAL, PRE, STEP, load_official  # noqa: E402

for p in (str(HERE), str(STEP), str(PRE), str(HOP100), str(OFFICIAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 官方依赖模块名固定，避免 Windows spawn 找不到动态模块
for mod_name in ("md_fold_detail", "trial_metrics", "perf_loader"):
    spec = importlib.util.spec_from_file_location(mod_name, OFFICIAL / f"{mod_name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

_tr = load_official("task_runner")
run_baseline_main = _tr.run_baseline_main
train_one_fold = _tr.train_one_fold
