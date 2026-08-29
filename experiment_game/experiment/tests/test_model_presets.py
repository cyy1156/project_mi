"""model_presets 单测。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from experiment_game.experiment.model_presets import (  # noqa: E402
    active_weights_from_yaml,
    campaign_locked_model_preset_id,
    list_model_presets,
    match_preset_id,
    resolve_model_display_label,
    resolve_weight_display_label,
    short_weight_label,
)


def test_list_includes_baseline_and_fnz():
    presets = list_model_presets()
    ids = {p["id"] for p in presets}
    assert "openbmi_baseline" in ids
    assert "fnz" in ids
    fnz = next(p for p in presets if p["id"] == "fnz")
    assert fnz["ok"] is True


def test_active_weights_from_yaml():
    w = active_weights_from_yaml()
    assert w["preset_id"]
    assert w["task_ok"] and w["three_ok"]
    assert w["model_label"]
    assert w["weight_label"]
    assert "E1f" in w["model_label"] or "OpenBMI" in w["model_label"]


def test_compose_weight_label():
    from experiment_game.experiment.model_presets import compose_weight_label, resolve_model_name

    mn = resolve_model_name(preset_id="e1f_four_member")
    wl = compose_weight_label(mn, "A01 · FT · sim_x · run3 · gate FAIL")
    assert wl.startswith("E1f")
    assert "A01 · FT" in wl


def test_short_label():
    assert short_weight_label("experiment_game/data/models/fnz/best_three.pt") == "fnz"
    assert short_weight_label(
        "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/x/best_three.pt"
    ) == "openbmi_baseline"


def test_match_preset():
    presets = {p["id"]: p for p in list_model_presets()}
    fnz = presets["fnz"]
    assert match_preset_id(fnz["task"], fnz["three"]) == "fnz"
