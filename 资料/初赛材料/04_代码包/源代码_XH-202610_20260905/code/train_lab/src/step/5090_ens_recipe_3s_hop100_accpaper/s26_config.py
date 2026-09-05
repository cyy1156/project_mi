"""方案 26 · 5090 锚点 run 与预注册判定线（只读）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRAIN_LAB = Path(__file__).resolve().parents[3]
REPO = TRAIN_LAB.parent.parent
OUT_ALG = TRAIN_LAB / "out" / "5090_alg_incr_3s_hop100_accpaper"
OUT_S26 = TRAIN_LAB / "out" / "5090_ens_recipe_3s_hop100_accpaper"
FE_CACHE = TRAIN_LAB / "out" / "_fe_cache"

PKG24 = Path(__file__).resolve().parent.parent / "5090_baselines_openbmi_3s_hop100_accpaper"


@dataclass(frozen=True)
class MemberRuns:
    shallow: Path
    t_shallow: Path
    eegnet: Path
    conformer: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "shallow": self.shallow,
            "t_shallow": self.t_shallow,
            "eegnet": self.eegnet,
            "conformer": self.conformer,
        }


# 方案 24 正式 anchor（5090 · three 子目录）
DEFAULT_MEMBERS = MemberRuns(
    shallow=OUT_ALG / "shallow_openbmi_3s_hop100_balbatch_accpaper" / "openbmi_3s_hop100" / "run_20260823_095327" / "three",
    t_shallow=OUT_ALG / "shallow_openbmi_3s_hop100_balbatch_accpaper" / "openbmi_3s_hop100" / "run_20260823_123900" / "three",
    eegnet=OUT_ALG / "eegnet_openbmi_3s_hop100_balbatch_accpaper" / "openbmi_3s_hop100" / "run_20260823_131435" / "three",
    conformer=OUT_ALG / "conformer_openbmi_3s_hop100_balbatch_accpaper" / "openbmi_3s_hop100" / "run_20260823_135213" / "three",
)

# 预注册判定线（Three Acc_paper）
ANCHOR_E_UNIFORM = 0.5958
ANCHOR_S3_THREE = 0.5839
E1_ADOPT_PP = 0.5
E1_REPORT_PP = 0.2
R1_ADOPT_PP = 0.5
R3_MEMBER_PP = 0.5
E2_FUSION_PP = 0.3

import numpy as np

WEIGHT_GRID_STEP = 0.05
SMOOTH_R_CANDIDATES = (0, 1, 2)
TAU_CONF_GRID = tuple(round(float(x), 2) for x in np.arange(0.35, 0.71, 0.05))
CONSIST_C_CANDIDATES = (2, 3)
