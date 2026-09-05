"""Print OpenBMI 8-channel raw vs preprocessed examples (one mat file)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.steps.epoch_baseline import task_window_cue_0_to_4  # noqa: E402
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402
from src.common.steps.resample_zscore import to_model_tensor  # noqa: E402
from src.common.steps.select_channels import TARGET_CHANNELS, select_channels  # noqa: E402
from src.common.steps.slide_3s_hop100 import segment_to_3s_hop100_windows  # noqa: E402
from src.datasets.openbmi.load_mat import load_openbmi_mat  # noqa: E402


def main() -> None:
    mat = Path(r"D:\MI\DATA\openbmi\sess01_subj01_EEG_MI.mat")
    eeg = load_openbmi_mat(mat)[0]

    print("=== raw .mat continuous stream ===")
    print("subject:", eeg.subject)
    print("fs:", eeg.fs, "Hz")
    print("shape x:", eeg.x.shape, "(n_samples, n_all_channels)")
    print("all ch_names count:", len(eeg.ch_names))
    print("first 12 ch_names:", eeg.ch_names[:12])
    print("TARGET 8 channels:", TARGET_CHANNELS)
    print("events count:", len(eeg.events))
    print("first 3 events [cue_sample, label] (1=Left, 2=Right):")
    for ev in eeg.events[:3]:
        print(" ", ev)

    cue0 = int(eeg.events[0, 0])
    i0 = max(cue0, 0)
    i1 = min(i0 + 10, eeg.x.shape[0])
    x8raw = select_channels(eeg.x, eeg.ch_names)

    print("\n=== after channel pick: raw voltage (uV scale, unfiltered) ===")
    print("time samples", i0, "to", i1 - 1, "(around first cue)")
    hdr = "time_idx".ljust(10) + "".join(f"{c:>12}" for c in TARGET_CHANNELS)
    print(hdr)
    for t in range(i0, i1):
        vals = x8raw[t]
        print(str(t).ljust(10) + "".join(f"{v:12.3f}" for v in vals))

    seg1s = x8raw[cue0 : cue0 + int(eeg.fs)]
    print("per-channel mean over 1s after cue (raw 8ch):", np.round(seg1s.mean(0), 3))
    print("per-channel std  over 1s after cue (raw 8ch):", np.round(seg1s.std(0), 3))

    x_car = car_reference(x8raw)
    x_f = notch_and_bandpass(x_car, eeg.fs)
    print("\n=== CAR + 50Hz notch + 8-30Hz bandpass (still 1000 Hz) ===")
    print("CAR check: channel-mean at cue sample =", float(x_car[cue0].mean()))
    print(hdr)
    for t in range(cue0, cue0 + 10):
        vals = x_f[t]
        print(str(t).ljust(10) + "".join(f"{v:12.3f}" for v in vals))
    print(
        "per-channel std over 1s (filtered):",
        np.round(x_f[cue0 : cue0 + int(eeg.fs)].std(0), 3),
    )

    seg = task_window_cue_0_to_4(x_f, cue0, eeg.fs)
    print("\n=== task segment: cue+0..4s with 0.5s baseline correction ===")
    print("segment shape:", None if seg is None else seg.shape, "(samples@1000Hz, 8ch)")

    wins = segment_to_3s_hop100_windows(seg, eeg.fs, zscore=True)
    print("3s/hop100 windows from this trial:", len(wins))
    w0 = wins[0]
    print("window0 shape after resample+zscore:", w0.shape, "(750 @250Hz, 8ch)")
    print("window0 per-channel mean (~0):", np.round(w0.mean(0), 4))
    print("window0 per-channel std  (~1):", np.round(w0.std(0), 4))
    print("window0 first 5 timepoints x 8 channels:")
    print(np.round(w0[:5], 4))
    x = to_model_tensor([w0])
    print("model tensor shape:", x.shape, "-> (N,1,8,750)")

    wins_noz = segment_to_3s_hop100_windows(seg, eeg.fs, zscore=False)
    wz = wins_noz[0]
    print("\n=== same window: 250Hz resampled, before z-score ===")
    print("per-channel mean:", np.round(wz.mean(0), 4))
    print("per-channel std :", np.round(wz.std(0), 4))
    print("first 5 timepoints:")
    print(np.round(wz[:5], 4))


if __name__ == "__main__":
    main()
