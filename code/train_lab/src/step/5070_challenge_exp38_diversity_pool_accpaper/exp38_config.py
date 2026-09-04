# -*- coding: utf-8 -*-
"""Exp38 配置。Exp37 未换卷 → A0 = nested-S0。"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT_TAG = "5070_challenge_exp38_diversity_pool_accpaper"
PREFER_TAG = "full_20260902_1930"
RUN_TAG = "exp38_d1_20260904"
DELTA_LINE = 0.01
GREEDY_STOP_PP = 0.005  # 单步增益 <0.5pp 停
POOL_MAX = 6
W_B8_MAX = 0.40
W_C1_MAX = 0.50
MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")

# D1 神经候选（dgcnn_raw 锁 8ch，改用 Deep4Net 作更深 CNN 家族）
NEURAL_CANDIDATES = ("eegtcnet", "deep4")
CLASSICAL_CANDIDATES = ("fbcsp_lda", "riemann_tsc")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def train_lab_out() -> Path:
    return Path(__file__).resolve().parents[3] / "out"


def exp38_out() -> Path:
    return train_lab_out() / OUT_ROOT_TAG


def scheme_doc() -> Path:
    return (
        repo_root()
        / "资料"
        / "模型训练"
        / "38_旁路_官方主交卷_误差去相关选池_accpaper"
    )


def exp37_out() -> Path:
    return train_lab_out() / "5070_challenge_exp37_nested_mcnemar_accpaper"


def exp36_out() -> Path:
    return train_lab_out() / "5070_challenge_exp36_pool_xtrack_accpaper"


def a59_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_mi_59ch_accpaper"


def rankflip_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_rankflip_accpaper"


def exp37_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_exp37_nested_mcnemar_accpaper"
