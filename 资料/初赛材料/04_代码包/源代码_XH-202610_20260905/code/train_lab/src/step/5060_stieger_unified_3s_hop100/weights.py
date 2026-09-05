"""解析 5060 OpenBMI shallow 权重（2s / 3s）。"""

from __future__ import annotations

from pathlib import Path

from config import N_FOLDS, active_profile


def _run_complete(run_dir: Path) -> bool:
    for fold in range(N_FOLDS):
        if not (run_dir / "task" / f"fold{fold}" / "best_task.pt").is_file():
            return False
        if not (run_dir / "three" / f"fold{fold}" / "best_three.pt").is_file():
            return False
    return True


def resolve_openbmi_s3_run(model: str = "shallow", *, run_stamp: str | None = None) -> Path:
    prof = active_profile()
    name = prof.openbmi_model_pkg
    base = prof.openbmi_weight_root / name / prof.openbmi_dataset_key
    if not base.is_dir():
        raise FileNotFoundError(
            f"缺少 {prof.tw} OpenBMI 权重目录: {base}\n"
            f"请同步 5060 正式 run（{prof.openbmi_shallow_run}）"
        )
    stamp = run_stamp or prof.openbmi_shallow_run
    run_dir = base / stamp if stamp.startswith("run_") else base / f"run_{stamp}"
    if not _run_complete(run_dir):
        raise FileNotFoundError(f"指定 OpenBMI {prof.tw} run 不完整: {run_dir}")
    return run_dir


def ckpt_path(run_dir: Path, *, head: str, fold: int) -> Path:
    if head == "task":
        return run_dir / "task" / f"fold{fold}" / "best_task.pt"
    if head == "three":
        return run_dir / "three" / f"fold{fold}" / "best_three.pt"
    raise ValueError(head)
