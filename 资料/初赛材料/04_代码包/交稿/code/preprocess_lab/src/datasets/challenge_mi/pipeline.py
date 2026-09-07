"""官方 trial 预处理：切好后 CAR → notch → bandpass → z-score。"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.resample_zscore import trial_zscore, to_model_tensor
from src.datasets.challenge_mi.load_pkl import extract_trials_from_block, load_block


def preprocess_trial_ct(
    x_ct: np.ndarray,
    fs: float = 250.0,
    *,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> np.ndarray:
    """
    x_ct: (C, T) → (T, C) float32 z-scored。
    """
    # filter_car 期望 (n_times, n_ch)
    x_tc = np.asarray(x_ct, dtype=np.float64).T
    x_tc = car_reference(x_tc)
    x_tc = notch_and_bandpass(x_tc, fs=fs, l_freq=l_freq, h_freq=h_freq)
    x_tc = trial_zscore(x_tc)
    return np.asarray(x_tc, dtype=np.float32)


def preprocess_block_path(
    path,
    *,
    channel_mode: str = "59",
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> dict[str, Any]:
    obj = load_block(path)
    raw = extract_trials_from_block(obj, channel_mode=channel_mode)
    fs = float(raw["fs"])
    trials_tc: list[np.ndarray] = []
    for i in range(raw["X"].shape[0]):
        trials_tc.append(
            preprocess_trial_ct(
                raw["X"][i],
                fs=fs,
                l_freq=l_freq,
                h_freq=h_freq,
            )
        )
    X = to_model_tensor(trials_tc)  # (N,1,C,T)
    y = raw["y"]
    if y is not None:
        # y_task: MI=1 (左/右), Rest=0
        y_task = np.asarray([0 if int(v) == 2 else 1 for v in y], dtype=np.int64)
        y_three = np.asarray(y, dtype=np.int64)
    else:
        y_task = None
        y_three = None
    return {
        "X": X,
        "y_task": y_task,
        "y_three": y_three,
        "starts": raw["starts"],
        "channel_names": raw["channel_names"],
        "fs": fs,
        "has_label": raw["has_label"],
        "personID": raw["personID"],
        "blockID": raw["blockID"],
    }
