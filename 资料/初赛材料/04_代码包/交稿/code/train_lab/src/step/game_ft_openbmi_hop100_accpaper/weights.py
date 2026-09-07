"""解析 OpenBMI Acc_paper init（冻结 shallow run）。"""

from __future__ import annotations

from pathlib import Path

from config import N_FOLDS, OPENBMI_SHALLOW_RUN, OPENBMI_WEIGHT_ROOT


def _run_complete(run_dir: Path) -> bool:
    for fold in range(N_FOLDS):
        if not (run_dir / "task" / f"fold{fold}" / "best_task.pt").is_file():
            return False
        if not (run_dir / "three" / f"fold{fold}" / "best_three.pt").is_file():
            return False
    return True


def resolve_openbmi_accpaper_run(model: str, *, run_stamp: str | None = None) -> Path:
    name = f"{model}_openbmi_2s_hop100_balbatch_accpaper"
    base = OPENBMI_WEIGHT_ROOT / name / "openbmi_2s_hop100"
    if not base.is_dir():
        raise FileNotFoundError(f"缺少 OpenBMI 权重目录: {base}")
    stamp = run_stamp
    if not stamp and model == "shallow":
        stamp = OPENBMI_SHALLOW_RUN
    if stamp:
        run_dir = (
            base / stamp if stamp.startswith("run_") else base / f"run_{stamp}"
        )
        if not _run_complete(run_dir):
            raise FileNotFoundError(f"指定 OpenBMI run 不完整: {run_dir}")
        return run_dir
    runs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
        reverse=True,
    )
    for r in runs:
        if _run_complete(r):
            return r
    raise FileNotFoundError(
        f"{model}: 无完整 OpenBMI Acc_paper run under {base}"
    )


def ckpt_path(run_dir: Path, *, head: str, fold: int) -> Path:
    if head == "task":
        return run_dir / "task" / f"fold{fold}" / "best_task.pt"
    if head == "three":
        return run_dir / "three" / f"fold{fold}" / "best_three.pt"
    raise ValueError(head)
