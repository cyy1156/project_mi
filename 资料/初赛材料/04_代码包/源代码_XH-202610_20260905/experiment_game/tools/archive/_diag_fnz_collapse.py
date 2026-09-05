"""诊断 fnz 权重塌缩：训练集/新 session 上的预测分布。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from adapt_engine.registry import ModelRegistry, load_head
from experiment_game.tools.ft_subject_from_v3 import build_dataset, _trial_split, N_TIMES

WS01 = _REPO / "experiment_game/data/sessions/fnz_ws01_20260826_164149"
WS02 = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
BASE_THREE = _REPO / (
    "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/"
    "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/"
    "run_20260822_094942/three/fold0/best_three.pt"
)
FNZ_THREE = _REPO / "experiment_game/data/models/fnz/best_three.pt"
LABELS = {0: "Rest", 1: "Left", 2: "Right"}


@torch.no_grad()
def pred_dist(ckpt: Path, X: np.ndarray, y: np.ndarray, device: str = "cpu") -> dict:
    entry = load_head(ckpt, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    model.eval()
    preds = []
    logits_all = []
    bs = 64
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s : s + bs]).to(device)
        try:
            logits = model(xb)
        except RuntimeError:
            logits = model(xb.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        logits_all.append(logits.cpu().numpy())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    pred = np.concatenate(preds)
    logits = np.concatenate(logits_all)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    return {
        "pred_counts": Counter(pred.tolist()),
        "label_counts": Counter(y.tolist()),
        "acc": float((pred == y).mean()),
        "mean_p": probs.mean(axis=0).tolist(),
        "pred": pred,
    }


def ws02_windows():
    """用 FT 同款管线切 ws02 窗（mi 段）。"""
    from experiment_game.tools.ft_subject_from_v3 import _load_eeg, _cut_windows
    from src.common.steps.filter_car import car_reference, notch_and_bandpass

    t, x = _load_eeg(WS02)
    x_f = notch_and_bandpass(car_reference(x), 250.0, l_freq=8.0, h_freq=30.0)
    tt = pd.read_csv(WS02 / "alignment/trial_table.csv")
    wins, labs = [], []
    for r in tt.to_dict("records"):
        if int(r.get("rejected") or 0):
            continue
        ws = _cut_windows(x_f, t, float(r["t_mi_start"]), float(r["t_mi_end"]))
        for w in ws:
            wins.append(w)
            labs.append(int(r["label"]))
    return np.stack(wins), np.asarray(labs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset(WS01, include_invalid=True)
    X, y, tids = ds["X"], ds["y_three"], ds["split_id"]
    tr_m, te_m = _trial_split(tids, train_frac=0.7, seed=42)

    print("=== FT 数据 fnz_ws01 ===")
    print(f"windows={len(X)} y={dict(Counter(y.tolist()))}")
    print(f"train={tr_m.sum()} heldout={te_m.sum()}")

    for name, ckpt in [("OpenBMI base", BASE_THREE), ("fnz FT", FNZ_THREE)]:
        print(f"\n--- {name} on ws01 ALL ---")
        d = pred_dist(ckpt, X, y, device)
        print(f"  acc={d['acc']:.3f} pred={ {LABELS[k]:v for k,v in d['pred_counts'].items()} }")
        print(f"  mean_p Rest/L/R={[round(x,3) for x in d['mean_p']]}")

        print(f"--- {name} on ws01 TRAIN ---")
        dtr = pred_dist(ckpt, X[tr_m], y[tr_m], device)
        print(f"  acc={dtr['acc']:.3f} pred={ {LABELS[k]:v for k,v in dtr['pred_counts'].items()} }")

        print(f"--- {name} on ws01 HELDOUT ---")
        dte = pred_dist(ckpt, X[te_m], y[te_m], device)
        print(f"  acc={dte['acc']:.3f} pred={ {LABELS[k]:v for k,v in dte['pred_counts'].items()} }")

    X2, y2 = ws02_windows()
    print(f"\n=== ws02 windows={len(X2)} y={dict(Counter(y2.tolist()))} ===")
    for name, ckpt in [("OpenBMI base", BASE_THREE), ("fnz FT", FNZ_THREE)]:
        d = pred_dist(ckpt, X2, y2, device)
        print(f"{name}: acc={d['acc']:.3f} pred={ {LABELS[k]:v for k,v in d['pred_counts'].items()} } mean_p={[round(x,3) for x in d['mean_p']]}")

    meta = json.loads((_REPO / "experiment_game/data/models/fnz/meta.json").read_text(encoding="utf-8"))
    print("\n=== FT meta ===")
    print(json.dumps({k: meta[k] for k in ("class_counts_three", "three", "task") if k in meta}, indent=2))


if __name__ == "__main__":
    main()
