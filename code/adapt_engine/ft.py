"""FT 配方（三可替换接口之三）与增量微调。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FTRecipe:
    """微调配方。扩展点：25-G2 调 replay_ratio；25-G3 加轻增广（aug_fn 钩子）。"""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    batch_size: int = 32
    replay_ratio: float = 0.15
    seed: int = 42
    aug_fn: Optional[object] = None  # 25-G3： callable(window (8,750)) → window

    def to_dict(self) -> dict:
        return {
            "lr": self.lr, "weight_decay": self.weight_decay, "epochs": self.epochs,
            "batch_size": self.batch_size, "replay_ratio": self.replay_ratio,
            "seed": self.seed, "aug": self.aug_fn is not None,
        }


class ReplayPool:
    """源域回放池（按类平衡采样；25-G2）。"""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, seed: int = 42):
        self.windows = np.asarray(windows, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self._rng = random.Random(seed)
        self._by_class = {
            int(c): np.where(self.labels == c)[0].tolist() for c in np.unique(self.labels)
        }

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n <= 0 or len(self.labels) == 0:
            return np.zeros((0,) + self.windows.shape[1:], np.float32), np.zeros((0,), np.int64)
        classes = list(self._by_class.keys())
        idxs: List[int] = []
        for i in range(n):
            c = classes[i % len(classes)]
            pool = self._by_class[c]
            idxs.append(pool[self._rng.randrange(len(pool))])
        return self.windows[idxs], self.labels[idxs]


class IncrementalFinetuner:
    """在同一被试上一轮微调权重上继续（采集流程 v2.0 §4.2）。

    - train_round()：本轮 12（标定）或 16（游戏）试次的窗 + 回放混合；
    - ckpt 链：每轮前 save_checkpoint(tag)，供漂移保护回滚。
    """

    def __init__(
        self,
        model: nn.Module,
        recipe: FTRecipe,
        *,
        replay_pool: Optional[ReplayPool] = None,
        device: str = "cpu",
        ckpt_dir: Optional[Path] = None,
    ):
        self.model = model
        self.recipe = recipe
        self.replay_pool = replay_pool
        self.device = device
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay
        )
        self.round_no = 0
        self._history: List[dict] = []

    # —— ckpt 链 ——
    def snapshot_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def save_checkpoint(self, tag: str) -> Optional[Path]:
        if self.ckpt_dir is None:
            return None
        p = self.ckpt_dir / f"round{self.round_no:02d}_{tag}.pt"
        torch.save({"round": self.round_no, "tag": tag, "model": self.snapshot_state()}, p)
        return p

    def rollback(self, state: dict) -> None:
        self.model.load_state_dict(state)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.recipe.lr * getattr(self, "_lr_scale", 1.0),
            weight_decay=self.recipe.weight_decay,
        )

    def halve_lr(self) -> float:
        scale = getattr(self, "_lr_scale", 1.0) * 0.5
        self._lr_scale = scale
        for g in self.optimizer.param_groups:
            g["lr"] = self.recipe.lr * scale
        return self.recipe.lr * scale

    # —— 训练 ——
    def train_round(self, windows: np.ndarray, labels: np.ndarray, *, frozen: bool = False) -> dict:
        """一轮增量微调。frozen=True（漂移保护二档）只记录不更新。"""
        X = np.asarray(windows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        n = len(X)
        if frozen or n == 0:
            self.round_no += 1
            rec = {"round": self.round_no, "n": n, "frozen": True, "loss": None}
            self._history.append(rec)
            return rec

        # 回放混合（25-G2）
        if self.replay_pool is not None and self.recipe.replay_ratio > 0:
            n_rep = int(round(n * self.recipe.replay_ratio))
            Xr, yr = self.replay_pool.sample(n_rep)
            if len(Xr):
                X = np.concatenate([X, Xr], axis=0)
                y = np.concatenate([y, yr], axis=0)

        if self.recipe.aug_fn is not None:  # 25-G3 钩子
            X = np.stack([self.recipe.aug_fn(w) for w in X])

        self.model.train()
        rng = random.Random(self.recipe.seed + self.round_no)
        idx = list(range(len(X)))
        gen = torch.Generator().manual_seed(self.recipe.seed + self.round_no)
        losses: List[float] = []
        for _ep in range(self.recipe.epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), self.recipe.batch_size):
                batch = idx[s : s + self.recipe.batch_size]
                xb = torch.from_numpy(X[batch]).to(self.device)
                yb = torch.from_numpy(y[batch]).to(self.device)
                if xb.dim() == 3:
                    pass  # (B,8,T)
                self.optimizer.zero_grad()
                try:
                    logits = self.model(xb)
                except RuntimeError:
                    logits = self.model(xb.unsqueeze(1))
                if logits.dim() == 3:
                    logits = logits.reshape(logits.shape[0], -1)
                loss = torch.nn.functional.cross_entropy(logits, yb)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach()))
        self.model.eval()
        self.round_no += 1
        rec = {
            "round": self.round_no,
            "n_new": n,
            "n_train": len(X),
            "frozen": False,
            "loss": float(np.mean(losses)) if losses else None,
        }
        self._history.append(rec)
        return rec

    @property
    def history(self) -> List[dict]:
        return list(self._history)
