from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TaskHeadDataset(Dataset):
    """
    阶段 A：静息(0) / 任务(1)。
    输出 x 形状 (8, 1000)，供 braindecode.EEGNet 使用。
    """

    def __init__(self, data_dir: str | Path, split: str = "train"):
        data_dir = Path(data_dir)
        X = np.load(data_dir / f"{split}_X.npy").astype(np.float32)
        # (N, 1, 8, 1000) → (N, 8, 1000)
        if X.ndim == 4 and X.shape[1] == 1:
            X = X[:, 0, :, :]
        assert X.ndim == 3 and X.shape[1:] == (8, 1000), X.shape
        self.X = X
        self.y_task = np.load(data_dir / f"{split}_y_task.npy").astype(np.int64)
        assert len(self.X) == len(self.y_task)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (8, 1000)
        y = torch.tensor(self.y_task[idx], dtype=torch.long)
        return x, y


class ThreeHeadDataset(Dataset):
    """
    阶段 B：空闲(0) / 左手(1) / 右手(2)。
    输出 x 形状 (8, 1000)，供 braindecode.EEGNet 使用。
    """

    def __init__(self, data_dir: str | Path, split: str = "train"):
        data_dir = Path(data_dir)
        X = np.load(data_dir / f"{split}_X.npy").astype(np.float32)
        # (N, 1, 8, 1000) → (N, 8, 1000)
        if X.ndim == 4 and X.shape[1] == 1:
            X = X[:, 0, :, :]
        assert X.ndim == 3 and X.shape[1:] == (8, 1000), X.shape
        self.X = X
        self.y_three = np.load(data_dir / f"{split}_y_three.npy").astype(np.int64)
        assert len(self.X) == len(self.y_three)
        assert set(np.unique(self.y_three)).issubset({0, 1, 2})

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # (8, 1000)
        y = torch.tensor(self.y_three[idx], dtype=torch.long)
        return x, y


class ArrayTaskDataset(Dataset):
    """五折用：已经按 mask 切好的 X, y_task。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 4 and X.shape[1] == 1:
            X = X[:, 0, :, :]
        assert X.ndim == 3 and X.shape[1:] == (8, 1000), X.shape
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        assert len(self.X) == len(self.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class ArrayThreeDataset(Dataset):
    """五折用：已经按 mask 切好的 X, y_three。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 4 and X.shape[1] == 1:
            X = X[:, 0, :, :]
        assert X.ndim == 3 and X.shape[1:] == (8, 1000), X.shape
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        assert len(self.X) == len(self.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)