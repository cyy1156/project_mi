# -*- coding: utf-8 -*-
"""Exp40 配置。"""

from __future__ import annotations

from pathlib import Path

OUT_ROOT_TAG = "5070_challenge_exp40_csv_harden_tta_accpaper"
PREFER_TAG = "full_20260902_1930"
MEMBER_KEYS = ("shallow", "shallow_b", "eegnet", "conformer")
TTA_DELTAS = (0, 20, 40)  # 向内缩采样点
RB8_ANCHOR = 0.54
DELTA_TTA_LINE = 0.005  # +0.5pp
FOLD_OK_MIN = 4
TIE_PP = 0.01


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def train_lab_out() -> Path:
    return Path(__file__).resolve().parents[3] / "out"


def exp40_out() -> Path:
    return train_lab_out() / OUT_ROOT_TAG


def exp39_out() -> Path:
    return train_lab_out() / "5070_challenge_exp39_closing_replay_accpaper"


def scheme_doc() -> Path:
    return (
        repo_root()
        / "资料"
        / "模型训练"
        / "40_旁路_官方主交卷_CSV加固_边际校正与TTA_accpaper"
    )


def b8_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_mi_8ch_ft_accpaper"


def exp36_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_exp36_pool_xtrack_accpaper"


def exp39_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_exp39_closing_replay_accpaper"


def exp37_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_exp37_nested_mcnemar_accpaper"


def rankflip_step() -> Path:
    return Path(__file__).resolve().parent.parent / "5070_challenge_rankflip_accpaper"
