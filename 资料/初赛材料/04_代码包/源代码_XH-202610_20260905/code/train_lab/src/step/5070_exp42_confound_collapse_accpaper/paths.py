from __future__ import annotations

from pathlib import Path


def _find_repo(start: Path) -> Path:
    """Walk up until experiment_game/ + code/train_lab/ exist (handles D:\\MI vs D:\\code junctions)."""
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "experiment_game").is_dir() and (p / "code" / "train_lab").is_dir():
            return p
    # fallback: classic depth from .../code/train_lab/src/step/<pkg>
    return start.resolve().parents[5]


REPO = _find_repo(Path(__file__).resolve().parent)
SUBJECTS = REPO / "experiment_game" / "data" / "subjects"


def _find_dir(pattern: str) -> Path:
    model_root = REPO / "\u8d44\u6599" / "\u6a21\u578b\u8bad\u7ec3"
    if model_root.is_dir():
        hits = sorted(p for p in model_root.glob(pattern) if p.is_dir())
        if hits:
            return hits[0]
    hits = sorted(p for p in REPO.rglob(pattern) if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"no dir matching {pattern!r} under {REPO}")
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


EXP42_DOC = _find_dir("42_*accpaper")
ANALYSIS = EXP42_DOC / "analysis_42"
SUMMARY_DIR = EXP42_DOC / "\u603b\u7ed3"
OUT_ROOT = REPO / "code" / "train_lab" / "out" / "5070_exp42_confound_collapse"
EXP41_SUMMARY = _find_dir("41_*LeaveNext*") / "\u603b\u7ed3"

CHANS = ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"]
FS = 250.0
WIN_S = 3.0
HOP_S = 0.1
N_TIMES = int(WIN_S * FS)
