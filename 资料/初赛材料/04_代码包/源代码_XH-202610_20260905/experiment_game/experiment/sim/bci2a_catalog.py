"""扫描 BCI2a mat 与预处理 shard。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.experiment.sim.bci2a_mat_loader import list_labeled_runs, load_bci2a_run

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_DATA = _REPO / "DATA" / "bci2a"
_SHARD_ROOT = _REPO / "code" / "preprocess_lab" / "out" / "bci2a_3s_hop100" / "shards"


def default_bci2a_data_dir() -> Path:
    return _DEFAULT_DATA


def resolve_mat_path(subject_id: str, *, data_dir: Optional[Path] = None) -> Path:
    sid = str(subject_id or "").strip().upper()
    if not sid.startswith("A") or len(sid) != 3:
        raise ValueError(f"仿真被试须为 A01–A09: {subject_id}")
    base = Path(data_dir) if data_dir else default_bci2a_data_dir()
    p = base / f"{sid}T.mat"
    if not p.is_file():
        raise FileNotFoundError(f"未找到 {p}")
    return p


def shard_exists(subject_id: str, run_id: str) -> bool:
    sid = str(subject_id).upper()
    shard = _SHARD_ROOT / f"{sid}_{run_id}"
    return (shard / "X.npy").is_file()


def list_subject_runs(subject_id: str, *, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """列出被试 mat 内可用 run 及 trial 数。"""
    mat_path = resolve_mat_path(subject_id, data_dir=data_dir)
    runs = list_labeled_runs(mat_path)
    out: List[Dict[str, Any]] = []
    for rid in runs:
        try:
            rd = load_bci2a_run(mat_path, rid)
            from experiment_game.experiment.sim.bci2a_mat_loader import count_run_capacity

            n_l, n_r, n_rest, n_total = count_run_capacity(rd)
        except Exception:
            n_l = n_r = n_rest = n_total = 0
        out.append(
            {
                "run_id": rid,
                "n_lr_trials": n_l + n_r,
                "n_left": n_l,
                "n_right": n_r,
                "n_rest_trials": n_rest,
                "n_total_trials": n_total,
                "shard_ok": shard_exists(subject_id, rid),
                "mat_path": str(mat_path),
            }
        )
    return out
