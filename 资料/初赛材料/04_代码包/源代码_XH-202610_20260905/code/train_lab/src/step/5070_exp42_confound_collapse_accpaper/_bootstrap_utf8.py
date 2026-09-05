# -*- coding: utf-8 -*-
"""Bootstrap: rewrite Exp42 package files as UTF-8 (ASCII-safe launcher)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def w(name: str, text: str) -> None:
    path = ROOT / name
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", path.name, "bytes", path.stat().st_size)


def main() -> None:
    w(
        "__init__.py",
        '"""Exp42 confound collapse diagnosis (offline, no deploy pollution)."""\n',
    )
    w(
        "paths.py",
        '''from __future__ import annotations

from pathlib import Path

# .../code/train_lab/src/step/<pkg>/paths.py -> parents[5] = repo root
REPO = Path(__file__).resolve().parents[5]
SUBJECTS = REPO / "experiment_game" / "data" / "subjects"
EXP42_DOC = (
    REPO
    / "\\u8d44\\u6599"
    / "\\u6a21\\u578b\\u8bad\\u7ec3"
    / "42_\\u65c1\\u8def_\\u771f\\u4eba\\u961f\\u5217\\u6df7\\u6742\\u5206\\u89e3\\u4e0e\\u4f1a\\u8bdd\\u7279\\u5f81\\u5749\\u584c\\u8bca\\u65ad_accpaper"
)
ANALYSIS = EXP42_DOC / "analysis_42"
SUMMARY_DIR = EXP42_DOC / "\\u603b\\u7ed3"
OUT_ROOT = REPO / "code" / "train_lab" / "out" / "5070_exp42_confound_collapse"
EXP41_SUMMARY = (
    REPO
    / "\\u8d44\\u6599"
    / "\\u6a21\\u578b\\u8bad\\u7ec3"
    / "41_\\u65c1\\u8def_\\u771f\\u4eba\\u88ab\\u8bd5LeaveNext_F5\\u5168\\u961f\\u5217\\u7edf\\u4e00_openbmi_accpaper"
    / "\\u603b\\u7ed3"
)

CHANS = ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"]
FS = 250.0
WIN_S = 3.0
HOP_S = 0.1
N_TIMES = int(WIN_S * FS)
'''.replace("\\\\u", "\\u").encode().decode("unicode_escape")
        if False
        else None,
    )


if __name__ == "__main__":
    # fix paths properly below
    pass
