"""方案 23 · 实验臂注册表。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    note: str
    geom_id: str = "G2s"
    oracle: bool = False
    use_predictor: bool = False
    use_expert_future: bool = False
    use_gate: bool = False
    use_sigreg: bool = False
    use_decoder: bool = False
    predictor_identity: bool = False
    expert_double: bool = False
    lambda_pred: float | None = None
    lambda_sig: float | None = None
    lambda_dec: float | None = None
    cls_cur: bool = True
    cls_final: bool = True
    cls_future: bool = False
    leak_eval_full: bool = False
    train_anchors_all: bool = False
    scheme23: bool = True
    tier: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


def _a(arm_id: str, note: str, **kw: Any) -> ArmSpec:
    return ArmSpec(arm_id=arm_id, note=note, **kw)


def _oracle_suffix(arm: ArmSpec) -> str:
    return f"{arm.arm_id}_oracle" if arm.oracle else arm.arm_id


ARMS: dict[str, ArmSpec] = {
    "O2s_m": _a(
        "O2s_m",
        "23 · G2s 掩码 · A1 机位校准",
        geom_id="G2s",
    ),
    "O2s_f": _a(
        "O2s_f",
        "23 · G2s oracle 上界",
        geom_id="G2s",
        oracle=True,
        leak_eval_full=True,
    ),
    "O1s_m": _a("O1s_m", "23 · G1s 掩码", geom_id="G1s"),
    "O1s_f": _a(
        "O1s_f",
        "23 · G1s oracle",
        geom_id="G1s",
        oracle=True,
        leak_eval_full=True,
    ),
    "O600": _a("O600", "23 · G600 真短输入无零填", geom_id="G600"),
    "L025": _a(
        "L025",
        "23 · A2 λ_pred=0.25",
        geom_id="G2s",
        use_predictor=True,
        use_sigreg=True,
        cls_final=False,
        lambda_pred=0.25,
    ),
    "L050": _a(
        "L050",
        "23 · A2 λ_pred=0.50",
        geom_id="G2s",
        use_predictor=True,
        use_sigreg=True,
        cls_final=False,
        lambda_pred=0.5,
    ),
    "A1_all": _a(
        "A1_all",
        "23 · 训练锚点解禁（含尾窗）",
        geom_id="G2s",
        train_anchors_all=True,
    ),
    "P1_local": _a(
        "P1_local",
        "23 · P1 本机复刻",
        geom_id="G2s",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        lambda_dec=0.0,
        tier=2,
    ),
    "E1": _a(
        "E1",
        "23 · Predictor 恒等 · 集成对照",
        geom_id="G2s",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        predictor_identity=True,
        lambda_dec=0.0,
        tier=2,
    ),
    "E2": _a(
        "E2",
        "23 · 单专家宽度×2 参数量对照",
        geom_id="G2s",
        expert_double=True,
        tier=2,
    ),
}

TIER1_ORDER: list[str] = [
    "O2s_m",
    "O2s_f",
    "O1s_m",
    "O1s_f",
    "O600",
    "L025",
    "L050",
    "A1_all",
]

TIER2_ORDER: list[str] = ["P1_local", "E1", "E2"]

CALIBRATION_ARM = "O2s_m"
CALIBRATION_LO = 0.562
CALIBRATION_HI = 0.585


def run_arm_folder_name(arm: ArmSpec, stamp: str) -> str:
    tag = _oracle_suffix(arm)
    return f"{stamp}_{tag}"


def assert_23_arm_flags() -> None:
    for aid in TIER1_ORDER + TIER2_ORDER:
        assert aid in ARMS, aid
    for arm in ARMS.values():
        if arm.oracle:
            assert arm.leak_eval_full
