# -*- coding: utf-8 -*-
"""Exp39 配置与路径。"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT_TAG = "5070_challenge_exp39_closing_replay_accpaper"
PREFER_TAG = "full_20260902_1930"
MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")
N_BOOT = 2000
UNI50_POOLED_LINE = 0.53  # R-uni50 进入工程候选门槛
TIE_PP = 0.01  # 工程平手：nested 差 <1pp → 最简


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def train_lab_out() -> Path:
    return Path(__file__).resolve().parents[3] / "out"


def exp39_out() -> Path:
    return train_lab_out() / OUT_ROOT_TAG


def scheme_doc() -> Path:
    return (
        repo_root()
        / "资料"
        / "模型训练"
        / "39_旁路_官方主交卷_收尾回放与工程选卷_accpaper"
    )


def exp37_out() -> Path:
    return train_lab_out() / "5070_challenge_exp37_nested_mcnemar_accpaper"


def exp38_out() -> Path:
    return train_lab_out() / "5070_challenge_exp38_diversity_pool_accpaper"


def a59_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_mi_59ch_accpaper"


def b8_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_mi_8ch_ft_accpaper"


def exp37_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_exp37_nested_mcnemar_accpaper"


def rankflip_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_rankflip_accpaper"
