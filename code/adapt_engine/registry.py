"""模型注册表（三可替换接口之一）。

默认：S3 双头单模型（braindecode ShallowFBCSPNet，n_times=750）。
扩展点（实验 24 阳性后即插即用，不改调用方）：
  - 24-E 集成：ckpt 列表 → 概率平均
  - 24-W 窗长模型组：按判定时刻选模型
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HeadEntry:
    model: nn.Module
    n_outputs: int
    source: str  # ckpt 路径或 "scratch"


def _build_shallow(n_chans: int, n_times: int, n_outputs: int, drop_prob: float = 0.5) -> nn.Module:
    """与 5070_baselines_openbmi_3s_hop100_accpaper/baseline_shallow.py 完全同构。"""
    from braindecode.models import ShallowFBCSPNet

    return ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        drop_prob=drop_prob,
    )


def load_head(
    ckpt_path: str | Path,
    *,
    n_chans: int = 8,
    n_times: int = 750,
    device: str = "cpu",
    build_fn=None,
) -> HeadEntry:
    """加载单头 checkpoint（task 或 three）。ckpt 内容由 task_runner 产出：
    {"model": state_dict, "n_outputs": int, ...}。"""
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = blob["model"]
    n_outputs = int(blob.get("n_outputs", 3))
    build = build_fn or (lambda n_out: _build_shallow(n_chans, n_times, n_out))
    model = build(n_outputs)
    model.load_state_dict(state)
    model.to(device).eval()
    return HeadEntry(model=model, n_outputs=n_outputs, source=str(ckpt_path))


class ModelRegistry:
    """双头（task/three）注册表；支持多头 ckpt 集成（概率平均）。

    Parameters
    ----------
    task_ckpts / three_ckpts : 单个或多个 ckpt 路径。多个 → 24-E 集成模式。
    """

    def __init__(
        self,
        task_ckpts: Sequence[str | Path] | str | Path,
        three_ckpts: Sequence[str | Path] | str | Path,
        *,
        n_chans: int = 8,
        n_times: int = 750,
        device: str = "cpu",
        build_fn=None,
    ) -> None:
        def _as_list(x):
            if isinstance(x, (str, Path)):
                return [x]
            return list(x)

        self.device = device
        self.task_heads: List[HeadEntry] = [
            load_head(p, n_chans=n_chans, n_times=n_times, device=device, build_fn=build_fn)
            for p in _as_list(task_ckpts)
        ]
        self.three_heads: List[HeadEntry] = [
            load_head(p, n_chans=n_chans, n_times=n_times, device=device, build_fn=build_fn)
            for p in _as_list(three_ckpts)
        ]

    @staticmethod
    def _forward_ensemble(heads: List[HeadEntry], window: np.ndarray) -> np.ndarray:
        """window: (8, 750) float32 → 概率 (n_outputs,)。多 ckpt 取概率平均。"""
        probs: List[np.ndarray] = []
        for h in heads:
            x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
            # 适配两种张量约定：(B,8,T) 与 (B,1,8,T)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            with torch.no_grad():
                try:
                    logits = h.model(x)
                except RuntimeError:
                    logits = h.model(x.unsqueeze(1))
            if logits.dim() == 3:
                logits = logits.reshape(logits.shape[0], -1)
            p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            # 集成成员 n_outputs 不一致时截到最短（防御，正常不触发）
            n_min = min(hh.n_outputs for hh in heads)
            probs.append(p[:n_min])
        return np.mean(np.stack(probs, axis=0), axis=0)

    def forward_heads(self, window: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """单窗推理：{"p_task": (2,) 或 None, "p_three": (3,)}。"""
        return {
            "p_task": self._forward_ensemble(self.task_heads, window) if self.task_heads else None,
            "p_three": self._forward_ensemble(self.three_heads, window),
        }

    def trainable_models(self) -> List[nn.Module]:
        """微调用：返回 three 头模型（在线适配作用在三分类主头）。"""
        return [h.model for h in self.three_heads]
