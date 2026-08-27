"""FT 配方（三可替换接口之三）与增量微调。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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
    balanced_batch: bool = False  # 每 batch 三类均衡采样（不足则重复抽样）
    aug_fn: Optional[object] = None  # 25-G3： callable(window (8,750)) → window

    def to_dict(self) -> dict:
        return {
            "lr": self.lr, "weight_decay": self.weight_decay, "epochs": self.epochs,
            "batch_size": self.batch_size, "replay_ratio": self.replay_ratio,
            "seed": self.seed, "balanced_batch": self.balanced_batch,
            "aug": self.aug_fn is not None,
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

    def _iter_batch_indices(self, y: np.ndarray, rng: random.Random) -> List[List[int]]:
        """返回一个 epoch 的 batch 下标列表。"""
        n = len(y)
        bs = self.recipe.batch_size
        if not self.recipe.balanced_batch:
            idx = list(range(n))
            rng.shuffle(idx)
            return [idx[s : s + bs] for s in range(0, n, bs)]

        classes = sorted(int(c) for c in np.unique(y))
        by_class = {c: [i for i in range(n) if int(y[i]) == c] for c in classes}
        if not classes or any(len(by_class[c]) == 0 for c in classes):
            idx = list(range(n))
            rng.shuffle(idx)
            return [idx[s : s + bs] for s in range(0, n, bs)]

        per_class = bs // len(classes)
        extra = bs % len(classes)
        n_batches = max(1, n // bs)
        batches: List[List[int]] = []
        for _ in range(n_batches):
            batch: List[int] = []
            for j, c in enumerate(classes):
                need = per_class + (1 if j < extra else 0)
                pool = by_class[c]
                batch.extend(rng.choices(pool, k=need))
            rng.shuffle(batch)
            batches.append(batch)
        return batches

    def _mix_training_xy(self, windows: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """被试窗 + replay 混合；返回 (X, y, n_new)。"""
        X = np.asarray(windows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        n = len(X)
        if self.replay_pool is not None and self.recipe.replay_ratio > 0:
            n_rep = int(round(n * self.recipe.replay_ratio))
            Xr, yr = self.replay_pool.sample(n_rep)
            if len(Xr):
                X = np.concatenate([X, Xr], axis=0)
                y = np.concatenate([y, yr], axis=0)
        if self.recipe.aug_fn is not None:  # 25-G3 钩子
            X = np.stack([self.recipe.aug_fn(w) for w in X])
        return X, y, n

    def train_one_epoch(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        *,
        epoch_idx: int = 0,
    ) -> float:
        """单 epoch 训练；返回平均 loss。"""
        X, y, _ = self._mix_training_xy(windows, labels)
        self.model.train()
        rng = random.Random(self.recipe.seed + self.round_no + int(epoch_idx))
        losses: List[float] = []
        for batch in self._iter_batch_indices(y, rng):
            xb = torch.from_numpy(X[batch]).to(self.device)
            yb = torch.from_numpy(y[batch]).to(self.device)
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
        return float(np.mean(losses)) if losses else 0.0

    def train_with_early_stop(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        eval_fn: Callable[[], float],
        *,
        max_epochs: int = 20,
        patience: int = 5,
        min_epochs: int = 1,
    ) -> dict:
        """按 heldout 指标早停；结束时恢复 best 权重。"""
        X = np.asarray(windows, dtype=np.float32)
        n_new = len(X)
        if n_new == 0:
            self.round_no += 1
            rec = {
                "round": self.round_no,
                "n_new": 0,
                "frozen": True,
                "early_stop": True,
                "epochs_run": 0,
                "best_epoch": 0,
                "best_heldout_acc": float("nan"),
                "history": [],
            }
            self._history.append(rec)
            return rec

        best_state = self.snapshot_state()
        best_acc = float(eval_fn())
        best_epoch = 0
        wait = 0
        history: List[dict] = []
        epochs_run = 0

        for ep in range(int(max_epochs)):
            loss = self.train_one_epoch(windows, labels, epoch_idx=ep)
            acc = float(eval_fn())
            epochs_run = ep + 1
            history.append({"epoch": epochs_run, "loss": loss, "heldout_acc": acc})
            if acc > best_acc:
                best_acc = acc
                best_state = self.snapshot_state()
                best_epoch = epochs_run
                wait = 0
            else:
                wait += 1
                if wait >= int(patience) and epochs_run >= int(min_epochs):
                    break

        self.rollback(best_state)
        self.model.eval()
        self.round_no += 1
        _, _, n_mixed = self._mix_training_xy(windows, labels)
        rec = {
            "round": self.round_no,
            "n_new": n_new,
            "n_train": n_mixed,
            "frozen": False,
            "early_stop": True,
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_heldout_acc": best_acc,
            "history": history,
            "loss": history[-1]["loss"] if history else None,
        }
        self._history.append(rec)
        return rec

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

        X, y, n_new = self._mix_training_xy(windows, labels)

        self.model.train()
        rng = random.Random(self.recipe.seed + self.round_no)
        losses: List[float] = []
        for _ep in range(self.recipe.epochs):
            for batch in self._iter_batch_indices(y, rng):
                xb = torch.from_numpy(X[batch]).to(self.device)
                yb = torch.from_numpy(y[batch]).to(self.device)
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
            "n_new": n_new,
            "n_train": len(X),
            "frozen": False,
            "loss": float(np.mean(losses)) if losses else None,
        }
        self._history.append(rec)
        return rec

    @property
    def history(self) -> List[dict]:
        return list(self._history)
