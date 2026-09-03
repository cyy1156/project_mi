# -*- coding: utf-8 -*-
"""轨 F / M 臂定义。"""

from __future__ import annotations

from exp35_config import MEMBER_KEYS, OPENBMI_WEIGHTS, FusionConstraints


def openbmi_weight_tuple(names: list[str]) -> tuple[float, ...]:
    raw = [OPENBMI_WEIGHTS[n] for n in names]
    s = sum(raw)
    return tuple(x / s for x in raw)


def uniform_weight_tuple(n: int) -> tuple[float, ...]:
    return tuple([1.0 / n] * n)


F_ARMS: dict[str, dict] = {
    "F0": {
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(name="F0_unconstrained"),
        "desc": "Exp34 无约束复现",
    },
    "F1": {
        "pool": list(MEMBER_KEYS),
        "cons": None,
        "fixed_openbmi": True,
        "desc": "OpenBMI 权重冻结 · T 可拟合",
    },
    "F2": {
        "pool": list(MEMBER_KEYS),
        "cons": None,
        "fixed_uniform": True,
        "desc": "均匀融合",
    },
    "F3": {
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(rank_prior=True, name="F3_rank_prior"),
        "desc": "排名先验",
    },
    "F4": {
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(w_eegnet_max=0.15, name="F4_eeg_cap"),
        "desc": "w_eegnet≤0.15",
    },
    "F5": {
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(w_conformer_min=0.40, name="F5_conf_floor"),
        "desc": "w_conformer≥0.40",
    },
    "F6": {
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(
            w_eegnet_max=0.15, w_conformer_min=0.40, name="F6_cap_and_floor"
        ),
        "desc": "F4∧F5",
    },
    "F7": {
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(t_min=0.5, t_max=5.0, name="F7_t_clamp"),
        "desc": "温度夹紧 [0.5,5]",
    },
}

M_ARMS: dict[str, dict] = {
    "M0": {
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(name="M0_unconstrained"),
        "desc": "四成员 sanity=F0",
    },
    "M1": {
        "pool": ["conformer"],
        "cons": FusionConstraints(name="M1_conformer_only"),
        "desc": "单 conformer",
    },
    "M2": {
        "pool": ["shallow", "shallow_b", "conformer"],
        "cons": FusionConstraints(name="M2_no_eeg"),
        "desc": "去 eegnet",
    },
    "M2c": {
        "pool": ["shallow", "shallow_b", "conformer"],
        "cons": FusionConstraints(w_conformer_min=0.40, name="M2c_no_eeg_F6like"),
        "desc": "M2 + w_conformer≥0.40",
    },
    "M3": {
        "pool": ["shallow", "conformer"],
        "cons": FusionConstraints(name="M3_S_C"),
        "desc": "{S,C}",
    },
    "M4": {
        "pool": ["conformer", "eegnet"],
        "cons": FusionConstraints(name="M4_C_E"),
        "desc": "{C,E}",
    },
    "M5": {
        "pool": ["shallow", "shallow_b", "eegnet"],
        "cons": FusionConstraints(name="M5_no_conformer"),
        "desc": "去 conformer",
    },
}

D_ARMS: dict[str, dict] = {
    "D0": {
        "b8_arm": "scratch",
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(name="D0_scratch_uncon"),
        "desc": "B8-scratch 无约束",
    },
    "D1": {
        "b8_arm": "ft",
        "pool": list(MEMBER_KEYS),
        "cons": FusionConstraints(name="D1_ft_uncon"),
        "desc": "B8-ft 无约束",
    },
    "D2": {
        "b8_arm": "ft",
        "pool": list(MEMBER_KEYS),
        "fixed_openbmi": True,
        "cons": None,
        "desc": "B8-ft OpenBMI-w",
    },
    "D3": {
        "b8_arm": "ft",
        "pool": ["shallow", "shallow_b", "conformer"],
        "cons": FusionConstraints(name="D3_ft_no_eeg"),
        "desc": "B8-ft 去 eegnet",
    },
    "D4": {
        "b8_arm": "scratch",
        "pool": ["shallow", "shallow_b", "conformer"],
        "cons": FusionConstraints(name="D4_scratch_no_eeg"),
        "desc": "B8-scratch 去 eegnet",
    },
}


def resolve_cons(arm_def: dict, names: list[str]) -> FusionConstraints:
    if arm_def.get("fixed_openbmi"):
        return FusionConstraints(
            fixed_weights=openbmi_weight_tuple(names),
            name=str(arm_def.get("desc", "openbmi_w")),
        )
    if arm_def.get("fixed_uniform"):
        return FusionConstraints(
            fixed_weights=uniform_weight_tuple(len(names)),
            name=str(arm_def.get("desc", "uniform_w")),
        )
    cons = arm_def.get("cons")
    if cons is None:
        return FusionConstraints(name="unconstrained")
    return cons
