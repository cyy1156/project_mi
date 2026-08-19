"""实验臂注册表 · 对齐方案索引 A0→…→C。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArmSpec:
    """单臂开关；训练时由 model/losses 解释。"""

    arm_id: str
    note: str
    data: str  # "a0" | "pf"
    use_predictor: bool = False
    use_expert_future: bool = False
    use_gate: bool = False
    use_sigreg: bool = False
    use_decoder: bool = False
    predictor_temporal: bool = False  # U1
    use_spectral_decoder: bool = False  # U2
    gate_entropy: bool = False  # U3
    lambda_pred: float | None = None  # None → SHARED
    lambda_sig: float | None = None
    lambda_dec: float | None = None
    cls_cur: bool = True
    cls_final: bool = True
    cls_future: bool = False
    fixed_alpha: float | None = None  # B5
    mask_learnable: bool = False  # B8
    no_grad_target: bool = True  # B2 关掉
    ema_target: bool = False  # B10
    leak_eval_full: bool = False  # B9
    a1_600: bool = False  # A1 附报
    skip_in_auto_chain: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _a(
    arm_id: str,
    note: str,
    *,
    data: str = "pf",
    **kw: Any,
) -> ArmSpec:
    return ArmSpec(arm_id=arm_id, note=note, data=data, **kw)


# 主线 + 消融（B 相对 P1；C 相对 P2）
ARMS: dict[str, ArmSpec] = {
    "A0_ref": _a(
        "A0_ref",
        "A0-ref · braindecode Shallow 量级参考（500pt）",
        data="a0",
    ),
    "A0": _a(
        "A0",
        "A0-main · 自写 shallow + Expert_cur（500pt · 仅 Three · 主表）",
        data="a0",
    ),
    "A1": _a("A1", "可见上下文单专家 · 1000pt X_mask"),
    "A1_600": _a(
        "A1_600",
        "A1 附报 · 真 600pt",
        a1_600=True,
        skip_in_auto_chain=True,
    ),
    "P0": _a(
        "P0",
        "最小闭环 · Predictor + Expert_cur",
        use_predictor=True,
    ),
    "A2": _a(
        "A2",
        "预测辅助 · 推理仍单专家",
        use_predictor=True,
    ),
    "P1": _a(
        "P1",
        "双专家 + Gate + SIGReg（无 Decoder）",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        lambda_dec=0.0,
    ),
    "B1": _a(
        "B1",
        "w/o L_pred",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        lambda_pred=0.0,
        lambda_dec=0.0,
    ),
    "B2": _a(
        "B2",
        "w/o no_grad on X_full",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        no_grad_target=False,
        lambda_dec=0.0,
    ),
    "B3": _a(
        "B3",
        "w/o SIGReg",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=False,
        lambda_sig=0.0,
        lambda_dec=0.0,
    ),
    "B4": _a(
        "B4",
        "w/o Expert_future",
        use_predictor=True,
        use_expert_future=False,
        use_gate=False,
        use_sigreg=True,
        lambda_dec=0.0,
    ),
    "B5a": _a(
        "B5a",
        "fixed α=1.0",
        use_predictor=True,
        use_expert_future=True,
        use_gate=False,
        use_sigreg=True,
        fixed_alpha=1.0,
        lambda_dec=0.0,
    ),
    "B5b": _a(
        "B5b",
        "fixed α=0.5",
        use_predictor=True,
        use_expert_future=True,
        use_gate=False,
        use_sigreg=True,
        fixed_alpha=0.5,
        lambda_dec=0.0,
    ),
    "B6": _a(
        "B6",
        "+CE(p_future)",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        cls_future=True,
        lambda_dec=0.0,
    ),
    "B7": _a(
        "B7",
        "w/o CE(p_final)",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        cls_final=False,
        lambda_dec=0.0,
    ),
    "B8": _a(
        "B8",
        "learnable mask token",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        mask_learnable=True,
        lambda_dec=0.0,
    ),
    "B9": _a(
        "B9",
        "泄漏上限（评估看 X_full）",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        leak_eval_full=True,
        lambda_dec=0.0,
    ),
    "B10": _a(
        "B10",
        "EMA target encoder",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        ema_target=True,
        lambda_dec=0.0,
    ),
    "P2": _a(
        "P2",
        "训练定稿主结果 · +Decoder",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        lambda_dec=0.2,
    ),
    "C1": _a(
        "C1",
        "λ_dec=0（相对 P2）",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=False,
        lambda_dec=0.0,
    ),
    "C2a": _a(
        "C2a",
        "L_dec 无 PSD",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        lambda_dec=0.2,
        extra={"dec_no_psd": True},
    ),
    "C2b": _a(
        "C2b",
        "L_dec 无 μβ",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        lambda_dec=0.2,
        extra={"dec_no_mubeta": True},
    ),
    "C2c": _a(
        "C2c",
        "L_dec 无时域 MSE",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        lambda_dec=0.2,
        extra={"dec_no_time": True},
    ),
    "L1": _a(
        "L1",
        "超参扫描占位（chain 默认跳过；手动跑）",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        skip_in_auto_chain=True,
    ),
    # ---- U 系列（相对 P2；默认不进自动 chain）----
    "U1": _a(
        "U1",
        "时间维 Predictor（相对 P2）",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        predictor_temporal=True,
        lambda_dec=0.2,
        skip_in_auto_chain=True,
    ),
    "U2": _a(
        "U2",
        "Spectral Decoder μ/β（相对 P2）",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=False,
        use_spectral_decoder=True,
        lambda_dec=0.2,
        skip_in_auto_chain=True,
    ),
    "U3": _a(
        "U3",
        "Gate + 专家熵（相对 P2）",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        gate_entropy=True,
        lambda_dec=0.2,
        skip_in_auto_chain=True,
    ),
    "U12": _a(
        "U12",
        "U1+U2 附报组合",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=False,
        predictor_temporal=True,
        use_spectral_decoder=True,
        lambda_dec=0.2,
        skip_in_auto_chain=True,
    ),
    "U13": _a(
        "U13",
        "U1+U3 附报组合",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=True,
        predictor_temporal=True,
        gate_entropy=True,
        lambda_dec=0.2,
        skip_in_auto_chain=True,
    ),
    "U123": _a(
        "U123",
        "U1+U2+U3 附报组合",
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_sigreg=True,
        use_decoder=False,
        predictor_temporal=True,
        use_spectral_decoder=True,
        gate_entropy=True,
        lambda_dec=0.2,
        skip_in_auto_chain=True,
    ),
}


# 一键 chain 默认顺序（与方案索引一致；跳过 skip_in_auto_chain）
CHAIN_ORDER: list[str] = [
    "A0_ref",
    "A0",
    "A1",
    "P0",
    "A2",
    "P1",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5a",
    "B5b",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "P2",
    "C1",
    "C2a",
    "C2b",
    "C2c",
]


# U 单改 / 组合（不进 CHAIN_ORDER；手动或 run_u_*_guarded.ps1）
U_SERIES_ORDER: list[str] = ["U1", "U3", "U2"]
# 组合顺序对齐实验方案 U组合_U12_U13_U123.md
U_COMBO_ORDER: list[str] = ["U13", "U12", "U123"]


def assert_u_arm_flags() -> None:
    """启动期自检：组合臂开关与方案/互斥约定一致。"""
    u12, u13, u123 = ARMS["U12"], ARMS["U13"], ARMS["U123"]
    assert u12.predictor_temporal and u12.use_spectral_decoder and not u12.gate_entropy
    assert not u12.use_decoder
    assert u13.predictor_temporal and u13.gate_entropy and not u13.use_spectral_decoder
    assert u13.use_decoder
    assert (
        u123.predictor_temporal
        and u123.use_spectral_decoder
        and u123.gate_entropy
        and not u123.use_decoder
    )
    for aid in U_SERIES_ORDER + U_COMBO_ORDER:
        assert aid in ARMS and ARMS[aid].skip_in_auto_chain


def chain_steps(*, include_skipped: bool = False) -> list[str]:
    out = []
    for a in CHAIN_ORDER:
        spec = ARMS[a]
        if spec.skip_in_auto_chain and not include_skipped:
            continue
        out.append(a)
    return out
