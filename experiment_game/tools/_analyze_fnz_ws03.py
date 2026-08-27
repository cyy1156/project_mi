"""分析 fnz_ws03 会话。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(r"d:\MI")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from experiment_game.experiment.channel_layout import reorder_device_to_frozen

SES = _REPO / "experiment_game/data/sessions/fnz_ws03_20260826_174526"
FNZ_THREE = _REPO / "experiment_game/data/models/fnz/best_three.pt"
FNZ_TASK = _REPO / "experiment_game/data/models/fnz/best_task.pt"
BASE_THREE = _REPO / (
    "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260822_094942/three/fold0/best_three.pt"
)
BASE_TASK = BASE_THREE.parent.parent.parent / "task/fold0/best_task.pt"
LABELS = {0: "Rest", 1: "Left", 2: "Right"}


def main():
    rows = []
    p = SES / "v3_trial_features.jsonl"
    if not p.is_file():
        print("NO v3_trial_features.jsonl"); return
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    tt = pd.read_csv(SES / "alignment/trial_table.csv")
    print("=== 1. 协议与完成度 ===")
    print(f"试次: {len(rows)} / trial_table {len(tt)}")
    print(f"标签分布: {Counter(r['label'] for r in rows)}")
    print(f"block/cond: blocks={Counter(r['block'] for r in rows)} cond={Counter(r['cond'] for r in rows)}")
    valid = sum(1 for r in rows if r.get("valid"))
    invalid = len(rows) - valid
    print(f"valid={valid} invalid={invalid}")

    print("\n=== 2. 模型预测 (primary_judge) ===")
    preds = []
    ok = wrong = 0
    for r in rows:
        pj = r.get("primary_judge") or {}
        pred = pj.get("pred")
        lab = r["label"]
        preds.append(pred)
        if pred == lab:
            ok += 1
        else:
            wrong += 1
        p3 = pj.get("p_three") or [0, 0, 0]
        feat = r.get("features") or {}
        grade = feat.get("grade", {}).get("grade", feat.get("trial_grade", {}).get("grade", "-"))
        print(
            f"  T{r['trial_id']:02d} label={LABELS[lab]} pred={LABELS.get(pred,'?')} "
            f"p3=[{p3[0]:.2f},{p3[1]:.2f},{p3[2]:.2f}] "
            f"{'OK' if pred==lab else 'WRONG'} valid={r.get('valid')} grade={grade} "
            f"b{r['block']} {r['cond']}"
        )
    print(f"\n  pred分布: {Counter(preds)}  acc={ok}/{len(rows)}={ok/len(rows):.1%}")

    # confusion
    cm = np.zeros((3, 3), int)
    for r in rows:
        pj = r.get("primary_judge") or {}
        if pj.get("pred") is not None:
            cm[r["label"], pj["pred"]] += 1
    print("\n  confusion (rows=true, cols=pred) Rest/Left/Right:")
    for i, name in LABELS.items():
        print(f"    {name}: {cm[i].tolist()}")

    print("\n=== 3. 特征评级汇总 ===")
    grades = Counter((r.get("features") or {}).get("grade", {}).get("grade", "-") for r in rows)
    print(f"  trial grades: {dict(grades)}")
    erds = [float(r["features"]["mu_erd_contra"]) for r in rows if r.get("features") and "mu_erd_contra" in r["features"]]
    lats = [float(r["features"]["laterality_pp"]) for r in rows if r.get("features") and "laterality_pp" in r["features"]]
    if erds:
        print(f"  ERD_contra: mean={np.mean(erds):.1f}% n={len(erds)}")
    if lats:
        print(f"  laterality_pp: mean={np.mean(lats):.1f} n={len(lats)}")

    # EEG channel sanity
    df = pd.read_csv(SES / "eeg.csv", nrows=5000)
    cols = [c for c in df.columns if c != "lsl_time"]
    x = reorder_device_to_frozen(df[cols].to_numpy(float))
    from experiment_game.experiment.channel_layout import FROZEN_CHANNEL_ORDER

    ch = FROZEN_CHANNEL_ORDER
    print("\n=== 4. EEG 通道 (前5000点) ===")
    for i, n in enumerate(ch):
        col = x[:, i]
        print(f"  {n}: mean={col.mean():.1f} std={col.std():.1f} ptp={col.max()-col.min():.1f}")

    # replay with merged fnz weights
    if (SES / "eeg.csv").is_file() and (SES / "alignment/trial_table.csv").is_file():
        from experiment_game.tools._analyze_fnz_ws02 import feed_buffer, judge_at
        from experiment_game.experiment.inference_v2 import OnlinePreprocessor, RingBuffer
        from adapt_engine.registry import ModelRegistry

        t = pd.read_csv(SES / "eeg.csv")["lsl_time"].to_numpy(float)
        xraw = reorder_device_to_frozen(pd.read_csv(SES / "eeg.csv")[cols].to_numpy(float))
        reg_fnz = ModelRegistry(str(FNZ_TASK), str(FNZ_THREE))
        reg_base = ModelRegistry(str(BASE_TASK), str(BASE_THREE))
        pre = OnlinePreprocessor()
        print("\n=== 5. 离线回放 (t_rel=4.2) 抽样试次 ===")
        sample_ids = [1, 6, 12, 18, 24, 30, 36]
        for tid in sample_ids:
            tr = tt[tt["trial_id"] == tid]
            if tr.empty:
                continue
            tr = tr.iloc[0]
            mi_t = float(tr["t_mi_start"])
            r = next((x for x in rows if x["trial_id"] == tid), None)
            if not r:
                continue
            jf = judge_at(t, mi_t, 4.2, reg_fnz, pre, t, xraw)
            jb = judge_at(t, mi_t, 4.2, reg_base, pre, t, xraw)
            pj = r.get("primary_judge") or {}
            print(
                f"  T{tid:02d} label={LABELS[int(tr['label'])]} "
                f"online={LABELS.get(pj.get('pred'),'?')} "
                f"replay_fnz={LABELS.get(jf['pred'],'?') if jf else None} "
                f"replay_base={LABELS.get(jb['pred'],'?') if jb else None}"
            )


if __name__ == "__main__":
    main()
