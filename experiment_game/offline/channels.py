"""通道名映射：采集标签 → preprocess_lab TARGET_CHANNELS。"""



from __future__ import annotations



from typing import Dict, List, Sequence



import numpy as np



# preprocess_lab/src/steps/select_channels.py

TARGET_CHANNELS: List[str] = [

    "C3",

    "C4",

    "Cz",

    "CP3",

    "CP4",

    "CPz",

    "FC3",

    "FC4",

]



# experiment_game / lsl_connect 录制名 → 标准名

CHANNEL_ALIASES: Dict[str, str] = {

    "CZ": "Cz",

    "Cz": "Cz",

    "C3": "C3",

    "C4": "C4",

    "CPZ3": "CP3",

    "CP3": "CP3",

    "CPZ4": "CP4",

    "CP4": "CP4",

    "FC4": "FC4",

    "FC3": "FC3",

    "CPZ": "CPz",

    "CPz": "CPz",

}





def normalize_channel_name(name: str) -> str:

    key = name.strip()

    if key in CHANNEL_ALIASES:

        return CHANNEL_ALIASES[key]

    # 宽松：大小写不敏感

    low = {k.lower(): v for k, v in CHANNEL_ALIASES.items()}

    return low.get(key.lower(), key)





def reorder_to_target(

    x: np.ndarray,

    ch_names: Sequence[str],

) -> tuple[np.ndarray, List[str]]:

    """

    x: (n_times, n_ch) → (n_times, 8)，列顺序 = TARGET_CHANNELS。

    """

    norm = [normalize_channel_name(c) for c in ch_names]

    idx: List[int] = []

    missing: List[str] = []

    for want in TARGET_CHANNELS:

        try:

            idx.append(norm.index(want))

        except ValueError:

            missing.append(want)

    if missing:

        raise KeyError(

            f"缺少目标通道 {missing}；当前={list(ch_names)} → 归一化={norm}"

        )

    return np.asarray(x[:, idx], dtype=np.float64), list(TARGET_CHANNELS)


