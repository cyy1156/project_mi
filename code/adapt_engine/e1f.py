"""E1f 四成员在线推理：多架构 three 头 + 温度/加权融合（窗级）。

试次级 smooth + tau_conf 早停见 readout.e1f_conf_stop_from_judgments。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from adapt_engine.registry import HeadEntry, load_head

_REPO = Path(__file__).resolve().parents[2]
_BASELINE_PKG = _REPO / "code" / "train_lab" / "src" / "step" / "5090_baselines_openbmi_3s_hop100_accpaper"


def _ensure_baseline_path() -> None:
    p = str(_BASELINE_PKG)
    if p not in sys.path:
        sys.path.insert(0, p)


def build_fn_for_arch(arch: str) -> Callable[[int], nn.Module]:
    _ensure_baseline_path()
    if arch == "shallow":
        from baseline_shallow import build_model as build

        return lambda n_out: build(8, 750, n_out, 0.5)
    if arch == "eegnet":
        from baseline_eegnet import build_model as build

        return lambda n_out: build(8, 750, n_out, 0.5)
    if arch == "conformer":
        from baseline_conformer import build_model as build

        return lambda n_out: build(8, 750, n_out, 0.5)
    raise ValueError(f"unknown arch {arch!r}")


def apply_temperature_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-4)
    p = np.clip(probs.astype(np.float64), 1e-8, 1.0)
    logits = np.log(p) / t
    logits = logits - logits.max()
    exp = np.exp(logits)
    return (exp / exp.sum()).astype(np.float32)


def fuse_member_probs(
    member_probs: Sequence[np.ndarray],
    *,
    temperatures: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    calibrated = [
        apply_temperature_probs(p, t) for p, t in zip(member_probs, temperatures)
    ]
    stacked = np.stack(calibrated, axis=0)
    return np.tensordot(w, stacked, axes=(0, 0)).astype(np.float32)


@dataclass
class E1fFusionConfig:
    temperatures: tuple[float, ...]
    weights: tuple[float, ...]
    smooth_radius: int = 1
    tau_conf: float = 0.4

    @classmethod
    def from_dict(cls, blob: dict) -> "E1fFusionConfig":
        return cls(
            temperatures=tuple(float(x) for x in blob["temperatures"]),
            weights=tuple(float(x) for x in blob["weights"]),
            smooth_radius=int(blob.get("smooth_radius", 1)),
            tau_conf=float(blob.get("tau_conf", 0.4)),
        )


@dataclass
class E1fMemberSpec:
    name: str
    arch: str
    three_ckpt: str
    task_ckpt: str = ""


@dataclass
class E1fStackConfig:
    id: str
    label: str
    readout_mode: str
    primary_judge_mode: str
    task_ckpt: str
    members: List[E1fMemberSpec]
    fusion: E1fFusionConfig
    test_acc_paper: Optional[float] = None

    @classmethod
    def load_json(cls, path: Path | str, *, repo_root: Path | None = None) -> "E1fStackConfig":
        root = repo_root or _REPO
        p = Path(path)
        if not p.is_file():
            p = root / path
        blob = json.loads(p.read_text(encoding="utf-8"))
        members = [
            E1fMemberSpec(
                name=m["name"],
                arch=m["arch"],
                three_ckpt=m["three_ckpt"],
                task_ckpt=str(m.get("task_ckpt") or ""),
            )
            for m in blob["members"]
        ]
        return cls(
            id=str(blob.get("id", "e1f")),
            label=str(blob.get("label", "E1f")),
            readout_mode=str(blob.get("readout_mode", "e1f")),
            primary_judge_mode=str(blob.get("primary_judge_mode", "majority")),
            task_ckpt=str(blob.get("task_ckpt", "")),
            members=members,
            fusion=E1fFusionConfig.from_dict(blob["fusion"]),
            test_acc_paper=blob.get("test_acc_paper"),
        )

    def resolve_paths(self, *, repo_root: Path | None = None) -> "E1fStackConfig":
        root = repo_root or _REPO

        def _rel(p: str) -> str:
            return str((root / p).resolve()) if p and not Path(p).is_absolute() else p

        return E1fStackConfig(
            id=self.id,
            label=self.label,
            readout_mode=self.readout_mode,
            primary_judge_mode=self.primary_judge_mode,
            task_ckpt=_rel(self.task_ckpt),
            members=[
                E1fMemberSpec(
                    name=m.name,
                    arch=m.arch,
                    three_ckpt=_rel(m.three_ckpt),
                    task_ckpt=_rel(m.task_ckpt) if m.task_ckpt else "",
                )
                for m in self.members
            ],
            fusion=self.fusion,
            test_acc_paper=self.test_acc_paper,
        )

    def missing_paths(self, *, repo_root: Path | None = None) -> List[str]:
        root = repo_root or _REPO
        missing: List[str] = []
        if self.task_ckpt:
            p = Path(self.task_ckpt)
            if not p.is_file():
                p = root / self.task_ckpt
            if not p.is_file():
                missing.append(f"缺 E1f task 权重: {self.task_ckpt}")
        for m in self.members:
            p = Path(m.three_ckpt)
            if not p.is_file():
                p = root / m.three_ckpt
            if not p.is_file():
                missing.append(f"缺 E1f 成员 {m.name} three 权重: {m.three_ckpt}")
            if m.task_ckpt:
                tp = Path(m.task_ckpt)
                if not tp.is_file():
                    tp = root / m.task_ckpt
                if not tp.is_file():
                    missing.append(f"缺 E1f 成员 {m.name} task 权重: {m.task_ckpt}")
        return missing

    def with_member_overrides(
        self,
        overrides: Dict[str, Dict[str, str]],
    ) -> "E1fStackConfig":
        """按成员名覆盖 three/task 路径（用于被试 current all4 叠加）。

        overrides 例::
            {"shallow": {"three_ckpt": "...", "task_ckpt": "..."}}
        """
        members: List[E1fMemberSpec] = []
        for m in self.members:
            ov = overrides.get(m.name) or {}
            members.append(
                E1fMemberSpec(
                    name=m.name,
                    arch=m.arch,
                    three_ckpt=str(ov.get("three_ckpt") or m.three_ckpt),
                    task_ckpt=str(ov.get("task_ckpt") or m.task_ckpt or ""),
                )
            )
        task = self.task_ckpt
        if "shallow" in overrides and overrides["shallow"].get("task_ckpt"):
            task = str(overrides["shallow"]["task_ckpt"])
        return E1fStackConfig(
            id=self.id,
            label=self.label,
            readout_mode=self.readout_mode,
            primary_judge_mode=self.primary_judge_mode,
            task_ckpt=task,
            members=members,
            fusion=self.fusion,
            test_acc_paper=self.test_acc_paper,
        )


class E1fRegistry:
    """四成员 E1f 注册表；接口与 ModelRegistry 对齐。"""

    def __init__(
        self,
        stack: E1fStackConfig,
        *,
        n_chans: int = 8,
        n_times: int = 750,
        device: str = "cpu",
    ) -> None:
        self.stack = stack
        self.device = device
        self.fusion = stack.fusion
        self.three_heads: List[HeadEntry] = []
        for m in stack.members:
            build_fn = build_fn_for_arch(m.arch)
            self.three_heads.append(
                load_head(
                    m.three_ckpt,
                    n_chans=n_chans,
                    n_times=n_times,
                    device=device,
                    build_fn=build_fn,
                )
            )
        self.task_heads: List[HeadEntry] = []
        if stack.task_ckpt and Path(stack.task_ckpt).is_file():
            self.task_heads.append(
                load_head(
                    stack.task_ckpt,
                    n_chans=n_chans,
                    n_times=n_times,
                    device=device,
                    build_fn=build_fn_for_arch("shallow"),
                )
            )

    @staticmethod
    def _forward_one(head: HeadEntry, window: np.ndarray, device: str) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            try:
                logits = head.model(x)
            except RuntimeError:
                logits = head.model(x.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    def forward_heads(self, window: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        member_probs = [
            self._forward_one(h, window, self.device)[:3] for h in self.three_heads
        ]
        p_three = fuse_member_probs(
            member_probs,
            temperatures=self.fusion.temperatures,
            weights=self.fusion.weights,
        )
        p_task = None
        if self.task_heads:
            p_task = self._forward_one(self.task_heads[0], window, self.device)
        return {"p_task": p_task, "p_three": p_three}

    @staticmethod
    def _forward_batch(head: HeadEntry, windows: np.ndarray, device: str) -> np.ndarray:
        """windows: (N, C, T) → softmax (N, n_out)。"""
        x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).to(device)
        with torch.no_grad():
            try:
                logits = head.model(x)
            except RuntimeError:
                logits = head.model(x.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def forward_three_batch(
        self, windows: np.ndarray, *, batch_size: int = 64
    ) -> np.ndarray:
        """批量融合 three：windows (N,8,T) → p_three (N,3)。"""
        n = int(windows.shape[0])
        out = np.zeros((n, 3), dtype=np.float32)
        temps = self.fusion.temperatures
        weights = self.fusion.weights
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            xb = windows[s:e]
            member_probs = [
                self._forward_batch(h, xb, self.device)[:, :3] for h in self.three_heads
            ]
            for i in range(e - s):
                out[s + i] = fuse_member_probs(
                    [mp[i] for mp in member_probs],
                    temperatures=temps,
                    weights=weights,
                )
        return out

    def trainable_models(self) -> List[nn.Module]:
        """在线 FT 默认只训 shallow 成员（three 头）。"""
        if not self.three_heads:
            return []
        return [self.three_heads[0].model]
