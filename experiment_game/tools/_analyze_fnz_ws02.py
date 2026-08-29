"""分析 fnz_ws02：模型预测 vs 特征 vs 缓冲一致性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from experiment_game.experiment.channel_layout import reorder_device_to_frozen
from experiment_game.experiment.inference_v2 import (
    FS,
    N_TIMES_3S,
    InferenceService,
    OnlinePreprocessor,
    RingBuffer,
)
from adapt_engine.registry import ModelRegistry

SES = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
FNZ_THREE = _REPO / "experiment_game/data/models/fnz/best_three.pt"
FNZ_TASK = _REPO / "experiment_game/data/models/fnz/best_task.pt"
BASE_THREE = _REPO / (
    "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260822_094942/three/fold0/best_three.pt"
)
BASE_TASK = BASE_THREE.parent.parent.parent / "task/fold0/best_task.pt"
LABELS = {0: "Rest", 1: "Left", 2: "Right"}


def load_eeg():
    df = pd.read_csv(SES / "eeg.csv")
    cols = [c for c in df.columns if c != "lsl_time"]
    t = df["lsl_time"].to_numpy(float)
    x = reorder_device_to_frozen(df[cols].to_numpy(float))
    return t, x


def feed_buffer(buf: RingBuffer, buf_t: np.ndarray, buf_x: np.ndarray, t_end: float) -> None:
    """按时间顺序分块灌入，模拟在线 LSL pull（避免单次 push 超 cap）。"""
    mask = buf_t <= t_end
    xs = buf_x[mask]
    chunk = max(1, buf.cap // 4)
    for i in range(0, len(xs), chunk):
        buf.push(xs[i : i + chunk])


def judge_at(t_lsl, mi_t, t_rel, registry, pre, buf_t, buf_x):
    """模拟 RingBuffer + judge（mock local_clock 对齐离线 t_end）。"""
    from unittest.mock import patch

    buf = RingBuffer(capacity_s=120)
    t_end = mi_t + t_rel
    feed_buffer(buf, buf_t, buf_x, t_end)
    infer = InferenceService(buf, registry, pre, task_p_on=0.0, signal_quality=None)
    with patch("pylsl.local_clock", return_value=t_end):
        return infer.judge(mi_t, t_rel)


def offline_window_at(t_lsl, mi_t, t_rel, t_arr, x_filt):
    """不用 buffer，直接从 CSV 切 3s 窗（frozen 序，已滤波）。"""
    from src.common.steps.filter_car import car_reference, notch_and_bandpass
    from src.common.steps.resample_zscore import trial_zscore

    t_end = mi_t + t_rel
    i_end = int(np.searchsorted(t_arr, t_end))
    i_start = i_end - N_TIMES_3S
    if i_start < 0:
        return None
    raw = x_filt[i_start:i_end]
    pre = OnlinePreprocessor()
    # 模拟在线：取尾 12s
    tail_n = int(12 * FS)
    tail_start = max(0, i_end - tail_n)
    tail = x_filt[tail_start:i_end]
    return pre.process(tail)


def main():
    t, x_raw = load_eeg()
    from src.common.steps.filter_car import car_reference, notch_and_bandpass

    x_filt = notch_and_bandpass(car_reference(x_raw), FS, l_freq=8.0, h_freq=30.0)

    # 读 trial features
    rows = []
    with open(SES / "v3_trial_features.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print("=== 1. 协议标签分布 ===")
    labs = [r["label"] for r in rows]
    from collections import Counter

    print(Counter(labs), f"共 {len(rows)} 试次")

    print("\n=== 2. 在线记录：主判定点 pred vs label ===")
    for r in rows:
        pj = r.get("primary_judge") or {}
        pred = pj.get("pred")
        lab = r["label"]
        p3 = pj.get("p_three") or [0, 0, 0]
        ok = pred == lab
        feat = r.get("features") or {}
        grade = feat.get("grade", {}).get("grade") or feat.get("trial_grade", {}).get("grade", "—")
        print(
            f"  T{r['trial_id']:02d} label={LABELS[lab]}({lab}) "
            f"pred={LABELS.get(pred, pred)} p3=[{p3[0]:.3f},{p3[1]:.3f},{p3[2]:.3f}] "
            f"{'OK' if ok else 'WRONG'} grade={grade} block={r['block']} {r['cond']}"
        )

    preds = [r["primary_judge"]["pred"] for r in rows if r.get("primary_judge")]
    print(f"\n  pred 分布: {Counter(preds)}  → 模型几乎总判 Left(1)")

    # trial table timing
    tt = pd.read_csv(SES / "alignment/trial_table.csv")
    mi_trials = tt[tt["label"].isin([1, 2])]

    print("\n=== 3. fnz vs OpenBMI 底座：块2 左手试次 (label=1) ===")
    reg_fnz = ModelRegistry(str(FNZ_TASK), str(FNZ_THREE))
    reg_base = ModelRegistry(str(BASE_TASK), str(BASE_THREE))
    pre = OnlinePreprocessor()

    left_trials = [r for r in rows if r["label"] == 1]
    for r in left_trials:
        tid = r["trial_id"]
        tr = tt[tt["trial_id"] == tid].iloc[0]
        mi_t = float(tr["t_mi_start"])
        t_rel = 4.2
        j_fnz = judge_at(t, mi_t, t_rel, reg_fnz, pre, t, x_raw)
        j_base = judge_at(t, mi_t, t_rel, reg_base, pre, t, x_raw)
        pj = r["primary_judge"]
        print(
            f"  T{tid:02d} online pred={pj['pred']} | "
            f"replay_fnz={j_fnz['pred'] if j_fnz else None} "
            f"p3={[round(x,3) for x in j_fnz['p_three']] if j_fnz else None} | "
            f"replay_base={j_base['pred'] if j_base else None} "
            f"p3={[round(x,3) for x in j_base['p_three']] if j_base else None}"
        )

    print("\n=== 4. RingBuffer vs 直接切窗（缓冲一致性）===")
    tid = 19  # 块2 左手，特征明显
    tr = tt[tt["trial_id"] == tid].iloc[0]
    mi_t = float(tr["t_mi_start"])
    t_rel = 4.2
    t_end = mi_t + t_rel

    # Buffer path
    from unittest.mock import patch

    buf = RingBuffer(capacity_s=120)
    feed_buffer(buf, t, x_raw, t_end)
    infer = InferenceService(buf, reg_fnz, pre, task_p_on=0.0, signal_quality=None)
    with patch("pylsl.local_clock", return_value=t_end):
        j_buf = infer.judge(mi_t, t_rel)

    # Direct CSV window (raw, no reorder issue - already frozen)
    i_end = int(np.searchsorted(t, t_end))
    win_raw = x_raw[i_end - N_TIMES_3S : i_end]
    tail = x_raw[max(0, i_end - int(12 * FS)) : i_end]
    w_online = pre.process(tail)

    diff = np.abs(j_buf["window"] - w_online).max()
    print(f"  T19 t_rel=4.2s: buffer pred={j_buf['pred']} direct_replay pred via same pre")
    print(f"  max|window_buf - window_direct| = {diff:.6f} (应 ≈0)")

    # 检查 buffer 内 3s 窗 vs CSV 同区间 raw
    win_from_buf = buf.window_ending_at(t_end, N_TIMES_3S, t_now_lsl=t_end)
    max_raw_diff = float(np.max(np.abs(win_from_buf - win_raw)))
    print(f"  max|raw_buf_window - raw_csv_window| = {max_raw_diff:.6f} uV")

    print("\n=== 5. 右手试次 (label=2): fnz 仍判 Left，OpenBMI 底座对比 ===")
    for r in rows:
        if r["label"] != 2:
            continue
        tid = r["trial_id"]
        tr = tt[tt["trial_id"] == tid].iloc[0]
        mi_t = float(tr["t_mi_start"])
        t_rel = 4.2
        j_fnz = judge_at(t, mi_t, t_rel, reg_fnz, pre, t, x_raw)
        j_base = judge_at(t, mi_t, t_rel, reg_base, pre, t, x_raw)
        pj = r["primary_judge"]
        feat = r.get("features") or {}
        erd = feat.get("mu_erd_contra")
        lat = feat.get("laterality_pp")
        print(
            f"  T{tid:02d} online pred={pj['pred']} p_left={pj['p_three'][1]:.3f} | "
            f"replay_fnz={j_fnz['pred'] if j_fnz else None} "
            f"replay_base={j_base['pred'] if j_base else None} "
            f"p3_base={[round(x,3) for x in j_base['p_three']] if j_base else None} | "
            f"ERD_contra={erd} lat={lat}pp"
        )


if __name__ == "__main__":
    main()
