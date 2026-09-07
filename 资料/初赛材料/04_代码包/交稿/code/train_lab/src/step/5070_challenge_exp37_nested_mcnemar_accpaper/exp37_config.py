# -*- coding: utf-8 -*-
"""Exp37 配置与路径。"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT_TAG = "5070_challenge_exp37_nested_mcnemar_accpaper"
PREFER_TAG = "full_20260902_1930"
W_B8_MAX = 0.40
W_C1_MAX = 0.50
DELTA_LINE = 0.01  # nested Δ ≥ +1pp
N_BOOT = 2000
MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def train_lab_out() -> Path:
    return Path(__file__).resolve().parents[3] / "out"


def exp37_out() -> Path:
    return train_lab_out() / OUT_ROOT_TAG


def scheme_doc() -> Path:
    return (
        repo_root()
        / "资料"
        / "模型训练"
        / "37_旁路_官方主交卷_嵌套融合McNemar确认_accpaper"
    )


def exp36_out() -> Path:
    return train_lab_out() / "5070_challenge_exp36_pool_xtrack_accpaper"


def rankflip_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_rankflip_accpaper"


def a59_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_mi_59ch_accpaper"


def exp36_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_exp36_pool_xtrack_accpaper"
