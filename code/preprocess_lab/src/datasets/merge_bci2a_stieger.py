"""方案 A：离线拼接 BCI2a + Stieger → out/merged_2s/merged_*.npy

用法（仓库根 .venv）::

    cd code/preprocess_lab
    set PYTHONPATH=.
    python -m src.datasets.merge_bci2a_stieger

不覆盖 bci2a_2s / stieger_2s。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

PRE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BCI2A = PRE_ROOT / "out" / "bci2a_2s"
DEFAULT_STIEGER = PRE_ROOT / "out" / "stieger_2s"
DEFAULT_OUT = PRE_ROOT / "out" / "merged_2s"


def _load_bank(data_dir: Path, prefix: str) -> dict[str, np.ndarray]:
    need = ("X", "y_task", "y_three", "subjects")
    paths = {k: data_dir / f"{prefix}_{k}.npy" for k in need}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}")
    X = np.load(paths["X"])
    y_task = np.load(paths["y_task"])
    y_three = np.load(paths["y_three"])
    subjects = np.load(paths["subjects"], allow_pickle=True)
    n = len(X)
    if not (len(y_task) == len(y_three) == len(subjects) == n):
        raise ValueError(f"{prefix}: 长度不一致 X={n} yt={len(y_task)} y3={len(y_three)} s={len(subjects)}")
    return {"X": X, "y_task": y_task, "y_three": y_three, "subjects": subjects}


def _prefix_subjects(subjects: np.ndarray, dataset: str) -> np.ndarray:
    out = []
    for s in subjects:
        name = str(s)
        if name.startswith(f"{dataset}:"):
            out.append(name)
        else:
            out.append(f"{dataset}:{name}")
    return np.asarray(out, dtype=object)


def sanity_check(
    X: np.ndarray,
    y_task: np.ndarray,
    y_three: np.ndarray,
    subjects: np.ndarray,
    dataset: np.ndarray,
) -> dict:
    assert X.ndim == 4 and X.shape[1:] == (1, 8, 500), X.shape
    assert X.dtype == np.float32
    n = len(X)
    assert len(y_task) == len(y_three) == len(subjects) == len(dataset) == n
    assert set(np.unique(y_task)).issubset({0, 1})
    assert set(np.unique(y_three)).issubset({0, 1, 2})
    assert np.all((y_three == 0) == (y_task == 0))
    uniq = sorted(set(map(str, subjects)))
    assert all(":" in s for s in uniq), "被试 ID 必须带数据集前缀"
    n_bci = int(np.sum(dataset == "bci2a"))
    n_sti = int(np.sum(dataset == "stieger"))
    return {
        "N": n,
        "X_shape": list(X.shape),
        "n_subjects": len(uniq),
        "subjects": uniq,
        "n_bci2a": n_bci,
        "n_stieger": n_sti,
        "y_task_counts": np.bincount(y_task, minlength=2).tolist(),
        "y_three_counts": np.bincount(y_three, minlength=3).tolist(),
    }


def merge_and_save(
    bci2a_dir: Path,
    stieger_dir: Path,
    out_dir: Path,
) -> dict:
    b = _load_bank(bci2a_dir, "bci2a")
    s = _load_bank(stieger_dir, "stieger")

    sub_b = _prefix_subjects(b["subjects"], "bci2a")
    sub_s = _prefix_subjects(s["subjects"], "stieger")
    ds_b = np.asarray(["bci2a"] * len(sub_b), dtype=object)
    ds_s = np.asarray(["stieger"] * len(sub_s), dtype=object)

    X = np.concatenate([b["X"], s["X"]], axis=0).astype(np.float32, copy=False)
    y_task = np.concatenate([b["y_task"], s["y_task"]], axis=0)
    y_three = np.concatenate([b["y_three"], s["y_three"]], axis=0)
    subjects = np.concatenate([sub_b, sub_s], axis=0)
    dataset = np.concatenate([ds_b, ds_s], axis=0)

    stats = sanity_check(X, y_task, y_three, subjects, dataset)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "merged_X.npy", X)
    np.save(out_dir / "merged_y_task.npy", y_task)
    np.save(out_dir / "merged_y_three.npy", y_three)
    np.save(out_dir / "merged_subjects.npy", subjects)
    np.save(out_dir / "merged_dataset.npy", dataset)

    meta = {
        "scheme": "A",
        "scheme_name": "simple_concat",
        "created": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "bci2a": str(bci2a_dir.resolve()),
            "stieger": str(stieger_dir.resolve()),
        },
        "prefix": "merged",
        "stats": stats,
    }
    (out_dir / "merge_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("saved →", out_dir.resolve())
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="方案A：合并 bci2a_2s + stieger_2s")
    p.add_argument("--bci2a", type=Path, default=DEFAULT_BCI2A)
    p.add_argument("--stieger", type=Path, default=DEFAULT_STIEGER)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    merge_and_save(args.bci2a, args.stieger, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
