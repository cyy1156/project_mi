"""预注册臂定义（A/B 系列）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    series: Literal["A", "B"]
    weight: Literal["zeroshot", "ft"]
    ea_ref: str | None  # None | "src" | "cal"
    adabn: bool
    gate: str  # H0 | H1 | H2
    conf_tau: float | None = None
    anchor: str | None = None  # A0 | B0

    @property
    def results_subdir(self) -> str:
        return f"S09-{self.arm_id}"


ARM_REGISTRY: dict[str, ArmSpec] = {
    "A0": ArmSpec("A0", "A", "zeroshot", None, False, "H0", None, "A0"),
    "A1": ArmSpec("A1", "A", "zeroshot", "src", False, "H0"),
    "A2": ArmSpec("A2", "A", "zeroshot", None, True, "H0"),
    "A3": ArmSpec("A3", "A", "zeroshot", "src", True, "H0"),
    "A3b": ArmSpec("A3b", "A", "zeroshot", "cal", True, "H0"),
    "A4": ArmSpec("A4", "A", "zeroshot", "src", True, "H2"),
    "B0": ArmSpec("B0", "B", "ft", None, False, "H0", None, "B0"),
    "B1": ArmSpec("B1", "B", "ft", "src", False, "H0"),
    "B2": ArmSpec("B2", "B", "ft", None, True, "H0"),
    "B3": ArmSpec("B3", "B", "ft", "src", True, "H0"),
    "B4": ArmSpec("B4", "B", "ft", "src", True, "H2"),
    "B5": ArmSpec("B5", "B", "ft", "src", True, "H1"),
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
