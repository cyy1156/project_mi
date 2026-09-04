# -*- coding: utf-8 -*-
"""Exp36 配置与路径。"""

from __future__ import annotations

from pathlib import Path

TRAIN_DEVICE_LABEL = "NVIDIA GeForce RTX 5070 Laptop"
OUT_ROOT_TAG = "5070_challenge_exp36_pool_xtrack_accpaper"
PREFER_TAG = "full_20260902_1930"
ANCHOR_S0 = 0.558
REPLACE_LINE = 0.568  # S0 + 1pp
W_B8_MAX = 0.40
MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def train_lab_out() -> Path:
    return Path(__file__).resolve().parents[3] / "out"


def exp36_out() -> Path:
    return train_lab_out() / OUT_ROOT_TAG


def scheme_doc() -> Path:
    return (
        repo_root()
        / "资料"
        / "模型训练"
        / "36_旁路_官方主交卷_扩池与跨轨融合_accpaper"
    )


def rankflip_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_rankflip_accpaper"


def a59_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_mi_59ch_accpaper"
