"""model_presets 单测。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from experiment_game.experiment.model_presets import (  # noqa: E402
    active_weights_from_yaml,
    list_model_presets,
    match_preset_id,
    short_weight_label,
)


def test_list_includes_baseline_and_fnz():
    presets = list_model_presets()
    ids = {p["id"] for p in presets}
    assert "openbmi_baseline" in ids
    assert "fnz" in ids
    fnz = next(p for p in presets if p["id"] == "fnz")
    assert fnz["ok"] is True


def test_active_weights_fnz():
    w = active_weights_from_yaml()
    assert w["preset_id"] == "fnz"
    assert w["task_ok"] and w["three_ok"]


def test_short_label():
    assert short_weight_label("experiment_game/data/models/fnz/best_three.pt") == "fnz"
    assert short_weight_label(
        "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/x/best_three.pt"
    ) == "openbmi_baseline"


def test_match_preset():
    presets = {p["id"]: p for p in list_model_presets()}
    fnz = presets["fnz"]
    assert match_preset_id(fnz["task"], fnz["three"]) == "fnz"
