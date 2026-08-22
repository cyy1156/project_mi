"""预注册臂定义 · v1.2 严格因子设计（统一 noz 管线）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    series: Literal["A", "B"]
    weight: Literal["zeroshot", "ft"]
    ea_ref: Literal["off", "src", "cal"]  # 全部走 X_noz 统一管线
    adabn: bool
    gate: str  # H0 | H1 | H2
    conf_tau: float | None = None
    anchor: str | None = None  # 配对基线臂（同后半协议）
    ablation_vs: tuple[str, ...] = ()  # 附报分解对照臂
    readonly_anchor: str | None = None  # 07 只读全量锚点

    @property
    def results_subdir(self) -> str:
        return f"S10-{self.arm_id}"


ARM_REGISTRY: dict[str, ArmSpec] = {
    # A 零样本：统一 noz→zscore 基线；EA 仅 src/cal 两档
    "A0": ArmSpec(
        "A0", "A", "zeroshot", "off", False, "H0",
        readonly_anchor="S07-01",
    ),
    "A1": ArmSpec(
        "A1", "A", "zeroshot", "src", False, "H0", anchor="A0",
    ),
    "A2": ArmSpec(
        "A2", "A", "zeroshot", "off", True, "H0",
        anchor="A0", ablation_vs=("A0",),
    ),
    "A3": ArmSpec(
        "A3", "A", "zeroshot", "src", True, "H0",
        anchor="A0", ablation_vs=("A1", "A2"),
    ),
    "A3b": ArmSpec(
        "A3b", "A", "zeroshot", "cal", True, "H0",
        anchor="A0", ablation_vs=("A3",),
    ),
    "A4": ArmSpec(
        "A4", "A", "zeroshot", "src", True, "H2",
        anchor="A0",
    ),
    # B FT：cal=前半白化到单位参考（非 OpenBMI src）
    "B0": ArmSpec(
        "B0", "B", "ft", "off", False, "H0",
        readonly_anchor="S07-05",
    ),
    "B1": ArmSpec(
        "B1", "B", "ft", "cal", False, "H0", anchor="B0",
    ),
    "B2": ArmSpec(
        "B2", "B", "ft", "off", True, "H0",
        anchor="B0", ablation_vs=("B0",),
    ),
    "B3": ArmSpec(
        "B3", "B", "ft", "cal", True, "H0",
        anchor="B0", ablation_vs=("B1", "B2"),
    ),
    "B4": ArmSpec(
        "B4", "B", "ft", "cal", True, "H2",
        anchor="B0",
    ),
    "B5": ArmSpec(
        "B5", "B", "ft", "cal", True, "H1",
        anchor="B0",
    ),
}


def get_arm(arm_id: str) -> ArmSpec:
    key = arm_id.strip().upper()
    if key == "A3B":
        key = "A3b"
    if key not in ARM_REGISTRY:
        raise KeyError(f"未知臂 {arm_id!r}；可选: {sorted(ARM_REGISTRY)}")
    return ARM_REGISTRY[key]


def parse_arms(text: str) -> list[ArmSpec]:
    arms = []
    for part in text.split(","):
        part = part.strip()
        if part:
            arms.append(get_arm(part))
    return arms
