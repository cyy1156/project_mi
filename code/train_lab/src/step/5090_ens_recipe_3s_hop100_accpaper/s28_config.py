"""方案 28 · 成员经济性回放消融 · 预注册判定线。"""

from __future__ import annotations

from s26_config import ANCHOR_S3_THREE

# 只读锚点（test Three Acc_paper）
ANCHOR_E1F = 0.6173
SANITY_TOL = 0.0005

R1_ADOPT_THREE = 0.5980
R4_ADOPT_THREE = 0.6143

# 臂 → 成员池（名称与 s26_config.MemberRuns 字段一致）
ARM_MEMBERS: dict[str, tuple[str, ...]] = {
    "R0": ("shallow",),
    "R1": ("shallow",),
    "R2": ("shallow",),
    "R3": ("shallow", "eegnet"),
    "R4": ("shallow", "eegnet"),
    "R5": ("shallow", "conformer"),
    "R6": ("shallow", "t_shallow", "eegnet", "conformer"),
}

R28_ARMS = tuple(ARM_MEMBERS.keys())

# 复用 S3 锚点命名
ANCHOR_S3 = ANCHOR_S3_THREE
