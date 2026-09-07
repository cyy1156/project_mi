"""方案 19 · 配对 μ/β 语料加载。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_paths import resolve_data


def load_openbmi_bundle(data_tag: str) -> dict:
    root, prefix = resolve_data(data_tag)
    paths = {
        "X": root / f"{prefix}_X.npy",
        "y_task": root / f"{prefix}_y_task.npy",
        "y_three": root / f"{prefix}_y_three.npy",
        "subjects": root / f"{prefix}_subjects.npy",
        "trial_id": root / f"{prefix}_trial_id.npy",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"缺少 {data_tag} 文件: {p}")
    meta = root / "preprocess_meta.json"
    return {
        "root": root,
        "x_path": str(paths["X"]),
        "X": np.load(paths["X"], mmap_mode="r"),
        "y_task": np.load(paths["y_task"]),
        "y_three": np.load(paths["y_three"]),
        "subjects": np.load(paths["subjects"], allow_pickle=True),
        "trial_id": np.load(paths["trial_id"]),
        "meta_path": str(meta) if meta.exists() else None,
    }


def assert_paired(mu: dict, beta: dict) -> None:
    n = len(mu["y_three"])
    if len(beta["y_three"]) != n:
        raise ValueError(f"窗数不一致: mu={n} beta={len(beta['y_three'])}")
    if not np.array_equal(mu["y_three"], beta["y_three"]):
        raise ValueError("y_three 不对齐：请用同一批 mat / 同几何重跑预处理")
    if not np.array_equal(mu["y_task"], beta["y_task"]):
        raise ValueError("y_task 不对齐")
    if not np.array_equal(mu["trial_id"], beta["trial_id"]):
        raise ValueError("trial_id 不对齐")
    # subjects 可能是 object 串
    if not np.array_equal(
        np.asarray(mu["subjects"], dtype=object),
        np.asarray(beta["subjects"], dtype=object),
    ):
        raise ValueError("subjects 不对齐")


class PairedIndexDataset(Dataset):
    """同一 indices 从 μ/β 两路 mmap 取窗。"""

    def __init__(
        self,
        y: np.ndarray,
        indices: np.ndarray,
        *,
        x_mu_path: str,
        x_beta_path: str,
    ):
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.x_mu_path = x_mu_path
        self.x_beta_path = x_beta_path
        self._mu = None
        self._beta = None
        n = int(np.load(x_mu_path, mmap_mode="r").shape[0])
        assert len(self.y) == n
        if len(self.indices):
            assert int(self.indices.min()) >= 0
            assert int(self.indices.max()) < n

    def _views(self):
        if self._mu is None:
            self._mu = np.load(self.x_mu_path, mmap_mode="r")
            self._beta = np.load(self.x_beta_path, mmap_mode="r")
        return self._mu, self._beta

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        mu, beta = self._views()
        x_mu = np.array(mu[idx], dtype=np.float32, copy=True)
        x_beta = np.array(beta[idx], dtype=np.float32, copy=True)
        if x_mu.ndim == 3 and x_mu.shape[0] == 1:
            x_mu = x_mu[0]
        if x_beta.ndim == 3 and x_beta.shape[0] == 1:
            x_beta = x_beta[0]
        assert x_mu.shape == (8, 500) and x_beta.shape == (8, 500), (
            x_mu.shape,
            x_beta.shape,
        )
        return (
            torch.from_numpy(x_mu),
            torch.from_numpy(x_beta),
            torch.tensor(self.y[idx], dtype=torch.long),
        )
