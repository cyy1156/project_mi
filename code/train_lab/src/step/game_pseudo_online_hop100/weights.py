"""解析 Acc_paper 重训权重（禁止 balbatch_balacc / trialmaj）。"""

from __future__ import annotations

from pathlib import Path

from config import N_FOLDS, WEIGHT_ROOT


def _run_complete(run_dir: Path) -> bool:
    for fold in range(N_FOLDS):
        if not (run_dir / "task" / f"fold{fold}" / "best_task.pt").is_file():
            return False
        if not (run_dir / "three" / f"fold{fold}" / "best_three.pt").is_file():
            return False
    return True


def resolve_accpaper_run(model: str, *, run_stamp: str | None = None) -> Path:
    """返回 `<model>_2s_hop100_balbatch_accpaper/bci2a_2s_hop100/run_<stamp>`。"""
    name = f"{model}_2s_hop100_balbatch_accpaper"
    base = WEIGHT_ROOT / name / "bci2a_2s_hop100"
    if not base.is_dir():
        raise FileNotFoundError(f"缺少权重目录: {base}")
    path_s = str(base).replace("\\", "/")
    if "balbatch_balacc" in path_s or "/baseline_2s_hop100_trialmaj/" in path_s:
        raise RuntimeError("禁止使用 01/trialmaj 权重路径")
    if run_stamp:
        run_dir = (
            base / run_stamp
            if run_stamp.startswith("run_")
            else base / f"run_{run_stamp}"
        )
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
    raise FileNotFoundError(
        f"{model}: 无完整 5 折 Task+Three Acc_paper run under {base}"
    )


def ckpt_path(run_dir: Path, *, head: str, fold: int) -> Path:
    if head == "task":
        return run_dir / "task" / f"fold{fold}" / "best_task.pt"
    if head == "three":
        return run_dir / "three" / f"fold{fold}" / "best_three.pt"
    raise ValueError(head)
