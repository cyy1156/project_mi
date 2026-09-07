"""实验 27 跨 session 泛化测试 + D 臂补跑后汇总。

- S 轨 winner（ws01 训练）→ 在 ws02 上测试
- M 轨 winner（ws01+ws02 训练）→ 在 ws03 上测试
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.registry import load_head
from experiment_game.tools.ft_subject_from_v3 import (
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    build_dataset,
)
from experiment_game.tools.exp27_fnz_replay_grid import pred_distribution

WS01 = _REPO / "experiment_game/data/sessions/fnz_ws01_20260826_164149"
WS02 = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
WS03 = _REPO / "experiment_game/data/sessions/fnz_ws03_20260826_174526"
OUT_DIR = _REPO / "experiment_game/data/models/fnz/exp27"

LABELS = {0: "Rest", 1: "Left", 2: "Right"}


def trial_majority_acc(model, X: np.ndarray, y: np.ndarray, split_ids: np.ndarray, device: str) -> Dict[str, Any]:
    """试次级多数票准确率。"""
    model.eval()
    preds_all: List[int] = []
    bs = 64
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s : s + bs]).to(device)
        with torch.no_grad():
            try:
                logits = model(xb)
            except RuntimeError:
                logits = model(xb.unsqueeze(1))
            if logits.dim() == 3:
                logits = logits.reshape(logits.shape[0], -1)
            preds_all.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    by_trial: Dict[str, List[int]] = defaultdict(list)
    by_label: Dict[str, int] = {}
    for i, sid in enumerate(split_ids):
        sid = str(sid)
        by_trial[sid].append(int(preds_all[i]))
        if sid not in by_label:
            by_label[sid] = int(y[i])

    trial_pred, trial_true = [], []
    for sid, ps in by_trial.items():
        cnt = Counter(ps)
        top = cnt.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            pred = top[0][0]
        else:
            # 平票：取该 pred 窗数最多的一类（与 session_v3 思路一致）
            pred = top[0][0]
        trial_pred.append(pred)
        trial_true.append(by_label[sid])

    acc = float(np.mean(np.array(trial_pred) == np.array(trial_true)))
    uniq, cnt = np.unique(trial_pred, return_counts=True)
    pred_counts = {LABELS[int(k)]: int(v) for k, v in zip(uniq, cnt)}
    max_frac = float(cnt.max() / len(trial_pred)) if len(trial_pred) else 0.0
    return {
        "n_trials": len(trial_true),
        "acc_trial_majority": acc,
        "pred_counts": pred_counts,
        "max_class_frac": max_frac,
    }


def eval_ckpt_on_session(ckpt: Path, session: Path, device: str) -> Dict[str, Any]:
    ds = build_dataset(session, include_invalid=True)
    X, y = ds["X"], ds["y_three"]
    split_ids = ds["split_id"]
    entry = load_head(ckpt, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    win_acc = _eval_acc(model, X, y, device)
    win_dist = pred_distribution(model, X, y, device)
    trial = trial_majority_acc(model, X, y, split_ids, device)
    return {
        "ckpt": str(ckpt),
        "session": session.name,
        "n_windows": int(len(X)),
        "acc_window": float(win_acc),
        "window_metrics": win_dist,
        "trial_metrics": trial,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report_path = OUT_DIR / "report.json"
    winners = {}
    if report_path.is_file():
        winners = json.loads(report_path.read_text(encoding="utf-8")).get("winners", {})

    s_ckpt = OUT_DIR / f"S_{winners.get('S', 'B3')}" / "best_three.pt"
    m_ckpt = OUT_DIR / f"M_{winners.get('M', 'E1')}" / "best_three.pt"
    if not s_ckpt.is_file():
        s_ckpt = OUT_DIR / "S_B3" / "best_three.pt"
    if not m_ckpt.is_file():
        m_ckpt = OUT_DIR / "M_E1" / "best_three.pt"

    results: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "winners": winners,
        "tests": [],
    }

    scenarios = [
        ("base_on_ws02", DEFAULT_THREE, WS02, "OpenBMI 底座 → ws02"),
        ("S_winner_on_ws02", s_ckpt, WS02, f"ws01 FT ({s_ckpt.parent.name}) → ws02"),
        ("base_on_ws03", DEFAULT_THREE, WS03, "OpenBMI 底座 → ws03"),
        ("M_winner_on_ws03", m_ckpt, WS03, f"ws01+ws02 FT ({m_ckpt.parent.name}) → ws03"),
    ]

    for key, ckpt, session, desc in scenarios:
        print(f"\n=== {desc} ===", flush=True)
        if not session.is_dir():
            print(f"  SKIP: missing {session}")
            continue
        if not Path(ckpt).is_file():
            print(f"  SKIP: missing ckpt {ckpt}")
            continue
        rep = eval_ckpt_on_session(Path(ckpt), session, device)
        rep["key"] = key
        rep["description"] = desc
        results["tests"].append(rep)
        tm = rep["trial_metrics"]
        wm = rep["window_metrics"]
        print(
            f"  window acc={rep['acc_window']:.3f}  trial acc={tm['acc_trial_majority']:.3f}  "
            f"trial pred={tm['pred_counts']}  max_frac={tm['max_class_frac']:.3f}",
            flush=True,
        )
        print(f"  window pred={wm.get('pred_counts')}", flush=True)

    out_json = OUT_DIR / "cross_session_eval.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 实验 27 · 跨 session 泛化测试",
        "",
        f"生成时间：{results['generated_at']}",
        "",
        "| 测试 | session | 窗 acc | 试次多数票 acc | trial pred (R/L/Rest) | max_frac |",
        "|------|---------|--------|----------------|----------------------|----------|",
    ]
    for t in results["tests"]:
        tm = t["trial_metrics"]
        pc = tm["pred_counts"]
        pred_str = f"{pc.get('Rest', 0)}/{pc.get('Left', 0)}/{pc.get('Right', 0)}"
        lines.append(
            f"| {t['description']} | {t['session']} | {t['acc_window']:.3f} | "
            f"{tm['acc_trial_majority']:.3f} | {pred_str} | {tm['max_class_frac']:.3f} |"
        )
    (OUT_DIR / "cross_session_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
