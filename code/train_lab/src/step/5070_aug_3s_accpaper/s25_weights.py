"""解析方案 25 权重：A0 锚点 S3 / G1 增广五折。"""

from __future__ import annotations

from pathlib import Path

from s25_config import (
    BASELINE_S3_ROOT,
    BASELINE_S3_RUN_DEFAULT,
    N_FOLDS,
    aug_weight_root,
)

MODEL_OUT_NAME = "shallow_openbmi_3s_hop100_balbatch_accpaper"
DATA_TAG = "openbmi_3s_hop100"


def _run_complete(run_dir: Path) -> bool:
    for fold in range(N_FOLDS):
        if not (run_dir / "task" / f"fold{fold}" / "best_task.pt").is_file():
            return False
        if not (run_dir / "three" / f"fold{fold}" / "best_three.pt").is_file():
            return False
    return True


def _pick_latest_run(base: Path, *, stamp: str | None) -> Path:
    if not base.is_dir():
        raise FileNotFoundError(f"缺少权重目录: {base}")
    if stamp:
        name = stamp if stamp.startswith("run_") else f"run_{stamp}"
        run_dir = base / name
        if not _run_complete(run_dir):
            raise FileNotFoundError(f"指定 run 不完整: {run_dir}")
        return run_dir
    runs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
        reverse=True,
    )
    for r in runs:
        if _run_complete(r):
            return r
    raise FileNotFoundError(f"无完整 run under {base}")


def resolve_anchor_s3_run(*, run_stamp: str | None = None) -> Path:
    base = BASELINE_S3_ROOT / MODEL_OUT_NAME / DATA_TAG
    stamp = run_stamp or BASELINE_S3_RUN_DEFAULT
    return _pick_latest_run(base, stamp=stamp)


def resolve_g1_run(
    *, run_stamp: str | None = None, train_device: str = "5070"
) -> Path:
    base = aug_weight_root(train_device) / MODEL_OUT_NAME / DATA_TAG
    return _pick_latest_run(base, stamp=run_stamp)


def resolve_weight_run(
    arm: str,
    *,
    run_stamp: str | None = None,
    train_device: str = "5070",
) -> Path:
    a = (arm or "A0").strip().upper()
    if a == "A0":
        return resolve_anchor_s3_run(run_stamp=run_stamp)
    if a in ("G1", "G2", "G3"):
        return resolve_g1_run(run_stamp=run_stamp, train_device=train_device)
    raise ValueError(f"未知 arm={arm!r}")
