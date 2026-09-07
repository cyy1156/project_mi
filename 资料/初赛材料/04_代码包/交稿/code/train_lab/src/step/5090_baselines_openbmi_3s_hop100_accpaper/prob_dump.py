"""方案 24 · 窗级 softmax dump / 加载（V/E 共用格式）。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DUMP_COLUMNS = (
    "subject",
    "fold",
    "split",
    "trial_id",
    "t0_sec",
    "pred",
    "y",
    "p_max",
    "p0",
    "p1",
    "p2",
)


def dump_rows_to_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DUMP_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in DUMP_COLUMNS})


def load_prob_dump(path: Path) -> dict[str, np.ndarray]:
    """快速加载 prob dump CSV。"""
    path = Path(path)
    try:
        import pandas as pd

        df = pd.read_csv(path)
        return {
            "subject": df["subject"].astype(str).to_numpy(dtype=object),
            "fold": df["fold"].to_numpy(dtype=np.int64),
            "split": df["split"].astype(str).to_numpy(dtype=object),
            "trial_id": df["trial_id"].to_numpy(dtype=np.int64),
            "t0_sec": df["t0_sec"].to_numpy(dtype=np.float32),
            "pred": df["pred"].to_numpy(dtype=np.int64),
            "y": df["y"].to_numpy(dtype=np.int64),
            "p_max": df["p_max"].to_numpy(dtype=np.float32),
            "probs": df[["p0", "p1", "p2"]].to_numpy(dtype=np.float32),
        }
    except ImportError:
        pass
    subs, folds, splits, tids, t0s = [], [], [], [], []
    preds, ys, pmax = [], [], []
    probs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subs.append(row["subject"])
            folds.append(int(row["fold"]))
            splits.append(row["split"])
            tids.append(int(row["trial_id"]))
            t0s.append(float(row["t0_sec"]))
            preds.append(int(row["pred"]))
            ys.append(int(row["y"]))
            pmax.append(float(row["p_max"]))
            probs.append([float(row["p0"]), float(row["p1"]), float(row["p2"])])
    return {
        "subject": np.asarray(subs, dtype=object),
        "fold": np.asarray(folds, dtype=np.int64),
        "split": np.asarray(splits, dtype=object),
        "trial_id": np.asarray(tids, dtype=np.int64),
        "t0_sec": np.asarray(t0s, dtype=np.float32),
        "pred": np.asarray(preds, dtype=np.int64),
        "y": np.asarray(ys, dtype=np.int64),
        "p_max": np.asarray(pmax, dtype=np.float32),
        "probs": np.asarray(probs, dtype=np.float32),
    }


def merge_prob_dumps(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [load_prob_dump(p) for p in paths]
    out: dict[str, list] = {k: [] for k in parts[0] if k in parts[0]}
    for part in parts:
        for k in out:
            out[k].append(part[k])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}
