"""§8 收尾冒烟：依赖方向 / 切窗 / 崩溃收尾 / defaults example 可解析。"""

from __future__ import annotations

import json
from pathlib import Path

from experiment_game.core.atomic_io import atomic_write_json
from experiment_game.core.windowing import WINDOWING_VERSION, n_windows_3s_hop100
from experiment_game.experiment.session_finalize import ensure_crash_artifacts


_PKG = Path(__file__).resolve().parents[2]
_EXAMPLE = _PKG / "config" / "operator_defaults.example.json"


def test_example_defaults_is_valid_json():
    raw = json.loads(_EXAMPLE.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    save_root = raw["storage"]["save_root"]
    assert not Path(save_root).is_absolute()
    assert "D:" not in save_root and "C:" not in save_root


def test_windowing_anchor():
    assert WINDOWING_VERSION == "openbmi_align_v1"
    assert n_windows_3s_hop100(4.0) == 11


def test_crash_finalize_writes_aborted(tmp_path: Path):
    root = tmp_path / "sess"
    root.mkdir()
    atomic_write_json(root / "run_config.json", {"ok": True})
    out = ensure_crash_artifacts(root, reason="smoke_test")
    assert out.get("session_meta") is True
    meta = json.loads((root / "session.meta.json").read_text(encoding="utf-8"))
    assert meta.get("aborted") is True
    assert meta.get("abort_reason") == "smoke_test"
