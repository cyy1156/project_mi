"""方案 19 臂定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    fuse: str
    lambda_aux: float
    note: str


ARMS: dict[str, ArmSpec] = {
    "V2": ArmSpec("V2", "gate", 0.5, "μ20+β20 + Gate · 主臂"),
    "V2_a05": ArmSpec("V2_a05", "fixed05", 0.5, "固定 α=0.5"),
    "V2_cat": ArmSpec("V2_cat", "concat", 0.5, "concat(z) 单头"),
    "V2_z0": ArmSpec("V2_z0", "gate", 0.0, "Gate · 无辅助 CE"),
}

# V1 = 方案18 S0，不在本包训练
V1_REF = {
    "arm": "V1(=S0)",
    "source": "18_旁路_手写vs库_ShallowFBCSP",
    "three_acc_paper": "0.5427±0.0243",
    "task_acc_paper": "0.6979±0.0356",
    "run": "run_20260819_162152",
}
