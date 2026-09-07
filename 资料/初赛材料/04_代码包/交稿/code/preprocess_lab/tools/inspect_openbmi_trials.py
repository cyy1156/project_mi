"""Inspect OpenBMI raw .mat vs preprocess_lab pipeline / out shards (few trials).

Usage (cyy env):
  python code/preprocess_lab/tools/inspect_openbmi_trials.py
  python code/preprocess_lab/tools/inspect_openbmi_trials.py --subj 01 --trials 3 --out-tag openbmi_3s_hop100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.common.steps.epoch_baseline import task_window_cue_0_to_4  # noqa: E402
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402
from src.common.steps.resample_zscore import to_model_tensor, trial_zscore  # noqa: E402
from src.common.steps.select_channels import TARGET_CHANNELS, select_channels  # noqa: E402
from src.common.steps.slide_2s_hop100 import (  # noqa: E402
    N_TIMES_2S,
    segment_to_2s_hop100_windows,
)
from src.common.steps.slide_3s_hop100 import (  # noqa: E402
    N_TIMES_3S,
    segment_to_3s_hop100_windows,
)
from src.datasets.bci2a.labels import filter_left_right_events  # noqa: E402
from src.datasets.openbmi.load_mat import load_openbmi_mat, subject_key  # noqa: E402

DATA_DIR = REPO / "DATA" / "openbmi"
OUT_ROOT = ROOT / "out"

LABEL3 = {0: "Rest", 1: "Left", 2: "Right"}


def _fmt_row(t: int, vals: np.ndarray) -> str:
    return str(t).ljust(10) + "".join(f"{v:12.3f}" for v in vals)


def _ch_stats(x: np.ndarray, label: str) -> None:
    std = np.std(x, axis=0)
    print(f"  {label} per-ch std (uV):", np.round(std, 2).tolist())
    print(f"  {label} median std:", round(float(np.median(std)), 2), "uV")
    print(f"  {label} peak |x|:", round(float(np.max(np.abs(x))), 2), "µV")


def _v4_hint(median_std: float) -> str:
    if median_std < 3.0:
        return "低于 v4 下限 3 uV（会 low_variance/dead）"
    if median_std > 60.0:
        return "高于 v4 上限 60 uV（会 artifact/saturation）"
    return "落在 v4 静息合格区 3–60 uV"


def inspect_trial(
    eeg,
    cue: int,
    lab_three: int,
    trial_idx: int,
    *,
    hop: str,
) -> dict:
    print(f"\n{'=' * 72}")
    print(f"Trial {trial_idx}  cue_sample={cue}  label={LABEL3.get(lab_three, lab_three)}")

    x8 = select_channels(eeg.x, eeg.ch_names)
    i0 = max(cue, 0)
    i1 = min(i0 + 10, x8.shape[0])
    hdr = "time_idx".ljust(10) + "".join(f"{c:>12}" for c in TARGET_CHANNELS)

    print("\n[1] 原始 .mat（8 导选取后，uV，未滤波）")
    print(hdr)
    for t in range(i0, i1):
        print(_fmt_row(t, x8[t]))
    seg1s = x8[cue : cue + int(eeg.fs)]
    _ch_stats(seg1s, "cue 后 1s 原始")
    print("  →", _v4_hint(float(np.median(seg1s.std(0)))))

    x_car = car_reference(x8)
    x_f = notch_and_bandpass(x_car, eeg.fs, l_freq=8.0, h_freq=30.0)
    print("\n[2] 预处理管线：CAR + 50Hz notch + 8–30 Hz（仍 @1000 Hz，训练用）")
    print("  CAR @cue mean ≈", round(float(x_car[cue].mean()), 6))
    print(hdr)
    for t in range(cue, cue + 10):
        print(_fmt_row(t, x_f[t]))
    seg_f1s = x_f[cue : cue + int(eeg.fs)]
    _ch_stats(seg_f1s, "cue 后 1s 滤波后")

    seg = task_window_cue_0_to_4(x_f, cue, eeg.fs)
    if seg is None:
        print("  task segment 越界，跳过滑窗")
        return {"trial_idx": trial_idx, "cue": cue, "n_windows": 0}

    print(f"\n[3] 任务段 cue+0..4s（基线 0.5s 校正后） shape={seg.shape}")

    if hop == "3s":
        wins_z = segment_to_3s_hop100_windows(seg, eeg.fs, zscore=True)
        wins_uv = segment_to_3s_hop100_windows(seg, eeg.fs, zscore=False)
        n_times = N_TIMES_3S
        hop_name = "3s / hop 100ms → (750,8) @250Hz"
    else:
        wins_z = segment_to_2s_hop100_windows(seg, eeg.fs, zscore=True)
        wins_uv = segment_to_2s_hop100_windows(seg, eeg.fs, zscore=False)
        n_times = N_TIMES_2S
        hop_name = "2s / hop 100ms → (500,8) @250Hz"

    print(f"\n[4] 滑窗 {hop_name}，本 trial 窗数={len(wins_z)}")
    if not wins_z:
        return {"trial_idx": trial_idx, "cue": cue, "n_windows": 0}

    wz = wins_z[0]
    wu = wins_uv[0]
    print("  窗0 z-score 后 per-ch std (~1):", np.round(wz.std(0), 3).tolist())
    print("  窗0 z-score 前 per-ch std (uV):", np.round(wu.std(0), 2).tolist())
    print("  窗0 z-score 后前 3 时间点 × 8 导:")
    print(np.round(wz[:3], 3))
    xt = to_model_tensor([wz])
    print("  模型张量:", xt.shape, "  (N,1,8,T)")

    return {
        "trial_idx": trial_idx,
        "cue": cue,
        "label_three": int(lab_three),
        "n_windows": len(wins_z),
        "win0_std_uv": [round(float(s), 2) for s in wu.std(0)],
        "win0_std_z": [round(float(s), 3) for s in wz.std(0)],
        "tensor_shape": list(xt.shape),
        "n_times": n_times,
    }


def compare_out_shard(
    subj: str,
    sess: str,
    out_tag: str,
    trial_local_id: int,
    *,
    expected_n_times: int,
) -> None:
    shard = OUT_ROOT / out_tag / "shards" / f"sess{sess}_subj{subj}_EEG_MI"
    if not shard.is_dir():
        print(f"\n[5] out shard 不存在: {shard}")
        return

    X = np.load(shard / "X.npy", mmap_mode="r")
    y3 = np.load(shard / "y_three.npy")
    tid = np.load(shard / "trial_id.npy")
    subj_key = subject_key(f"subj{subj}")

    mask = tid == trial_local_id
    idxs = np.where(mask)[0]
    print(f"\n[5] 对比 out/{out_tag} shard（trial_id={trial_local_id}）")
    print(f"  shard 路径: {shard}")
    print(f"  该 trial 窗数: {len(idxs)}  总 X.shape: {X.shape}")
    if len(idxs) == 0:
        return
    w0 = np.asarray(X[idxs[0]])
    if w0.ndim == 4:
        w0_tc = w0[0].T  # (T,8)
    else:
        w0_tc = w0.T
    print(f"  窗0 张量 shape: {w0.shape}  → 时间维 T={w0_tc.shape[0]}")
    print(f"  窗0 per-ch std (z-score 后): {np.round(w0_tc.std(0), 3).tolist()}")
    print(f"  窗0 y_three={int(y3[idxs[0]])} ({LABEL3.get(int(y3[idxs[0]]), '?')})")
    meta_path = OUT_ROOT / out_tag / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  meta.protocol={meta.get('protocol')} zscore={meta.get('zscore')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect OpenBMI raw vs preprocessed trials")
    p.add_argument("--subj", default="01", help="subject id, e.g. 01")
    p.add_argument("--sess", default="01", help="session id, e.g. 01")
    p.add_argument("--trials", type=int, default=3, help="number of MI trials to show")
    p.add_argument(
        "--out-tag",
        default="openbmi_2s_hop100",
        choices=["openbmi_2s_hop100", "openbmi_3s_hop100"],
        help="preprocessed out folder under preprocess_lab/out",
    )
    p.add_argument(
        "--mat",
        type=Path,
        default=None,
        help="override mat path",
    )
    args = p.parse_args()

    subj = str(args.subj).zfill(2)
    sess = str(args.sess).zfill(2)
    mat = args.mat or (DATA_DIR / f"sess{sess}_subj{subj}_EEG_MI.mat")
    hop = "3s" if "3s" in args.out_tag else "2s"
    n_times = N_TIMES_3S if hop == "3s" else N_TIMES_2S

    if not mat.is_file():
        raise FileNotFoundError(f"找不到 mat: {mat}")

    eeg = load_openbmi_mat(mat)[0]
    kept = filter_left_right_events(eeg.events, eeg.artifacts)

    print("=" * 72)
    print("OpenBMI 原始 vs 预处理 对照")
    print("=" * 72)
    print("mat:", mat)
    print("subject:", eeg.subject, " session:", eeg.session)
    print("fs:", eeg.fs, "Hz  连续流 shape:", eeg.x.shape, "(samples, 62ch)")
    print("8 导顺序:", TARGET_CHANNELS)
    print("MI trials (left/right):", len(kept))
    print("out tag:", args.out_tag, "  hop:", hop)

    summaries = []
    for ti, (cue, lab_task, lab_three, _) in enumerate(kept[: max(1, args.trials)]):
        summaries.append(
            inspect_trial(
                eeg,
                int(cue),
                int(lab_three),
                ti,
                hop=hop,
            )
        )
        compare_out_shard(subj, sess, args.out_tag, ti, expected_n_times=n_times)

    print("\n" + "=" * 72)
    print("小结（与 v4 / 在线采集对比）")
    print("=" * 72)
    print(
        "- .mat 原始已是 µV；训练管线 = 选 8 导 → CAR → notch50 → 8–30Hz → "
        "cue+0..4s 段 → 滑窗 → 250Hz 重采样 → 逐窗 z-score。"
    )
    print(
        "- v4 质检用的是 **在线采集** 的 0.5–45Hz µV（不做 z-score），"
        "阈值 3–60 uV；OpenBMI 离线滤波后 1s std 通常也在十几 uV 量级。"
    )
    print("- out/*.npy 存的是 z-score 后模型输入，std≈1 是正常的，不是 uV。")
    print("trial summaries:", json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
