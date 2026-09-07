"""协议自动判定：Align v1（含 cue_s=1）不得误判为 legacy_v3。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiment_game.pipeline.finetune import (
    PROTOCOL_LEGACY_V3,
    PROTOCOL_OPENBMI_ALIGN,
    detect_session_protocol,
)


def _write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def test_align_v1_with_cue_gap_uses_openbmi(tmp_path: Path) -> None:
    """cue_s=1 → t_mi_start−t_cue=1.0，但有 t_rest_* 仍应 openbmi_align。"""
    sd = tmp_path / "v3_fnz0830_w01"
    rows = [
        {
            "trial_id": 1,
            "label": 1,
            "rejected": 0,
            "invalid": 0,
            "t_cue": 10.0,
            "t_mi_start": 11.0,
            "t_mi_end": 15.0,
            "t_rest_start": 6.0,
            "t_rest_end": 10.0,
        },
    ]
    _write_table(sd / "alignment" / "trial_table.csv", rows)
    assert detect_session_protocol(sd) == PROTOCOL_OPENBMI_ALIGN


def test_old_ws_without_rest_markers_is_legacy(tmp_path: Path) -> None:
    """旧 ws01：Cue 与 MI 差 ~2s、无 t_rest_* → legacy_v3。"""
    sd = tmp_path / "fnz_ws01"
    rows = [
        {
            "trial_id": 1,
            "label": 1,
            "rejected": 0,
            "invalid": 0,
            "t_cue": 10.0,
            "t_mi_start": 12.0,
            "t_mi_end": 16.2,
        },
    ]
    _write_table(sd / "alignment" / "trial_table.csv", rows)
    assert detect_session_protocol(sd) == PROTOCOL_LEGACY_V3
