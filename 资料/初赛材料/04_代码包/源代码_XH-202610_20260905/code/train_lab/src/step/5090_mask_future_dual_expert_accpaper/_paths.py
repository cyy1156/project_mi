"""Path bootstrap · 对齐 5090_three_hier_loss_accpaper._official_load。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# .../train_lab/src/step/<pkg>
STEP = HERE.parent
SRC = STEP.parent
TRAIN_LAB = SRC.parent
CODE = TRAIN_LAB.parent  # code/
REPO = CODE.parent

PRE = CODE / "preprocess_lab"
OFFICIAL = STEP / "5060_baselines_openbmi_2s_hop100_accpaper"
HOP100 = STEP / "baselines_2s_hop100"
SELF_MODEL = REPO / "self_model"
OUT_ROOT = TRAIN_LAB / "out"

SCHEME_DOCS = [
    REPO / "资料" / "模型方案" / "掩码未来表征预测_双专家门控_在线MI",
    REPO / "资料" / "Lejepa_shallow模型方案" / "掩码未来表征预测_双专家门控_在线MI",
]

for p in (str(HERE), str(STEP), str(PRE), str(HOP100), str(OFFICIAL), str(SELF_MODEL)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_official(mod_name: str):
    """复用 5060 Acc_paper 训练环（与方案16 5090 包相同做法）。"""
    path = OFFICIAL / f"{mod_name}.py"
    key = f"_mfde5090_official_{mod_name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_scheme_doc() -> Path:
    for d in SCHEME_DOCS:
        if d.is_dir():
            return d
    return SCHEME_DOCS[0]
