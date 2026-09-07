# -*- coding: utf-8 -*-
"""Exp35 配置与路径（勿命名为 shared_hparams，以免与 A59 包冲突）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRAIN_DEVICE_LABEL = "NVIDIA GeForce RTX 5070 Laptop"
OUT_ROOT_TAG = "5070_challenge_rankflip_accpaper"
EXP34_A59_OUT = "5070_challenge_mi_59ch_accpaper"
EXP34_B8_OUT = "5070_challenge_mi_8ch_ft_accpaper"
MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")
OPENBMI_WEIGHTS = {
    "shallow": 0.2,
    "shallow_b": 0.2,
    "eegnet": 0.3,
    "conformer": 0.3,
}
ANCHOR_E1F_A59 = 0.558
ANCHOR_CONFORMER_A59 = 0.511
ANCHOR_SHALLOW_A59 = 0.453


@dataclass(frozen=True)
class FusionConstraints:
    w_eegnet_max: float | None = None
    w_conformer_min: float | None = None
    rank_prior: bool = False
    rank_prior_relaxed: bool = False
    t_min: float | None = None
    t_max: float | None = None
    fixed_weights: tuple[float, ...] | None = None
    name: str = ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def train_lab_out() -> Path:
    return Path(__file__).resolve().parents[3] / "out"


def exp35_out() -> Path:
    return train_lab_out() / OUT_ROOT_TAG


def scheme_doc() -> Path:
    return (
        repo_root()
        / "资料"
        / "模型训练"
        / "35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper"
    )


def ranking_doc_permanent() -> Path:
    """跨实验常驻排名对照表（轨 R 升格）。"""
    return repo_root() / "资料" / "模型训练" / "跨域三分类成员排名对照表.md"
