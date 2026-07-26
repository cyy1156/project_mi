"""MI_model 全局配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

MI_ROOT = Path(__file__).resolve().parent
DATA_ROOT = MI_ROOT / "data"
DATA_SUBDIR = "C3  C4  CZ  CP3  CP4  CPZ  FC3  FC4"

SFREQ = 250
N_CHANNELS = 8
CHANNEL_COLUMNS = ["C3", "C4", "CZ", "CP3", "CP4", "CPZ", "FC3", "FC4"]

# 采集端已完成（离线不重复）
ACQUISITION_FILTER = "0.5-45Hz+50Hz-notch"

# 离线唯一滤波步骤
MI_BANDPASS_LOW_HZ = 8.0
MI_BANDPASS_HIGH_HZ = 30.0

WARMUP_SEC = 2.0
WARMUP_SAMPLES = int(WARMUP_SEC * SFREQ)

WIN_SEC = 2.0
WIN_SAMPLES = int(WIN_SEC * SFREQ)
WIN_STEP_SEC = 0.5
WIN_STEP_SAMPLES = int(WIN_STEP_SEC * SFREQ)

EPOCH_SEC = 4.0
EPOCH_SAMPLES = int(EPOCH_SEC * SFREQ)

CSP_N_COMPONENTS = 4
SVM_C = 1.0
CV_N_SPLITS = 5
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42

# 每类使用相同数量的 CSV session（取各类可用文件数的最小值）
BALANCE_SESSIONS_PER_CLASS = True

# QC
WARMUP_AMP_WARN_UV = 200.0
SESSION_AMP_SKIP_UV = 500.0

PARADIGM = "continuous_mi"
OFFLINE_FILTER = "8-30Hz-only"


@dataclass
class StageConfig:
    stage_id: int
    name: str
    classes: Dict[str, int]
    class_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.class_names:
            self.class_names = [
                name for name, _ in sorted(self.classes.items(), key=lambda x: x[1])
            ]


# 文件名以 `_no` 结尾的 CSV 为质量问题标记，不参与训练（见 preprocessing.is_bad_session_file）
STAGES: Dict[int, StageConfig] = {
    1: StageConfig(
        stage_id=1,
        name="stage1",
        classes={"front": 0, "back": 1},
        class_names=["front", "back"],
    ),
    2: StageConfig(
        stage_id=2,
        name="stage2",
        classes={"front": 0, "back": 1, "stop": 2},
    ),
    3: StageConfig(
        stage_id=3,
        name="stage3",
        classes={"front": 0, "back": 1, "stop": 2, "left": 3},
    ),
    4: StageConfig(
        stage_id=4,
        name="stage4",
        classes={"front": 0, "back": 1, "stop": 2, "left": 3, "right": 4},
    ),
}


def data_dir() -> Path:
    return DATA_ROOT / DATA_SUBDIR


def dataset_dir(stage: StageConfig) -> Path:
    return MI_ROOT / "dataset" / stage.name


def models_dir(stage: StageConfig) -> Path:
    return MI_ROOT / "models" / stage.name


def reports_dir(stage: StageConfig) -> Path:
    return MI_ROOT / "reports" / stage.name
