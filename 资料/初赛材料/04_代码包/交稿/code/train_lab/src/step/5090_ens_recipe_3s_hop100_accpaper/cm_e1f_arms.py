#!/usr/bin/env python3
"""E1f 各臂 test 混淆矩阵 / 各类召回率与特异性（只读回放，零训练）。

复用 replay_classic_vs_causal.py 的冻结融合（temps/weights 来自 replay_e1f.json），
在 OpenBMI 54 人五折 prob dump 上重放 W / C 臂并把试次判定广播到窗，
按 test split 统计 3x3 混淆矩阵与官方指标（准确率 / 召回率 / 特异性 / macro-F1）。

对账：W/C 臂 trial 级 Acc_paper 必须分别复现 0.6125 / 0.6188
（见 replay_classic_vs_causal.json at_tau_fixed）。

用法::

    python cm_e1f_arms.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "code"))

from e1_fusion_core import fuse_pipeline  # noqa: E402
from s26_config import DEFAULT_MEMBERS  # noqa: E402
from prob_io import load_members  # noqa: E402

from replay_classic_vs_causal import (  # noqa: E402
    E1F_JSON,
    _broadcast_trial_preds,
)

CLASSES = ["Rest", "Left", "Right"]
OUT_DEFAULT = HERE / "cm_e1f_arms.json"
TAU_FROZEN = 0.4


def confusion_from_rows(y: np.ndarray, pred: np.ndarray, n_cls: int = 3) -> np.ndarray:
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y.astype(int), pred.astype(int)):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    n_cls = cm.shape[0]
    total = int(cm.sum())
    per: List[Dict[str, Any]] = []
    f1s: List[float] = []
    for c in range(n_cls):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = total - tp - fn - fp
        recall = tp / (tp + fn) if (tp + fn) else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        precision = tp / (tp + fp) if (tp + fp) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        if f1 is not None:
            f1s.append(f1)
        per.append(
            {
                "class": CLASSES[c],
                "support": int(tp + fn),
                "recall": recall,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
            }
        )
    acc = float(np.trace(cm)) / total if total else None
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per,
        "confusion": cm.tolist(),
        "labels_row_true_col_pred": CLASSES,
    }


def eval_split_rows(data: dict, split: str) -> Dict[str, Any]:
    m = data["split"] == split
    return metrics_from_cm(confusion_from_rows(data["y"][m], data["pred"][m]))


def main() -> int:
    cfg = json.loads(E1F_JSON.read_text(encoding="utf-8"))["config"]
    temps = list(cfg["temperatures"])
    weights = tuple(float(w) for w in cfg["weights"])

    print("[load] four-member dumps …")
    runs = [
        DEFAULT_MEMBERS.shallow,
        DEFAULT_MEMBERS.t_shallow,
        DEFAULT_MEMBERS.eegnet,
        DEFAULT_MEMBERS.conformer,
    ]
    members = load_members(runs)
    fused_raw = fuse_pipeline(members, temperatures=temps, weights=weights, smooth_radius=0)

    out: Dict[str, Any] = {
        "protocol": "openbmi_3s_hop100_dump · E1f frozen fuse · test-split confusion",
        "tau_frozen": TAU_FROZEN,
        "e1f_config": cfg,
        "member_runs": [str(r) for r in runs],
        "anchors_acc_paper": {"W": 0.6125, "C": 0.6188},
    }

    # 单成员 shallow 窗级 argmax（3s S3 基线的每类指标）
    out["shallow_member_window_argmax"] = {
        split: eval_split_rows(members[0], split) for split in ("val", "test")
    }

    # 融合窗级 argmax（无试次聚合）
    argmax_view = dict(fused_raw)
    argmax_view["pred"] = fused_raw["probs"].argmax(axis=1).astype(np.int64)
    out["fused_window_argmax"] = {
        split: eval_split_rows(argmax_view, split) for split in ("val", "test")
    }

    # W（多数票）与 C（因果 conf-stop）臂：试次判定广播到窗
    for arm in ("W", "C"):
        data, t_decs, _ = _broadcast_trial_preds(fused_raw, arm=arm, tau=TAU_FROZEN)
        res = {split: eval_split_rows(data, split) for split in ("val", "test")}
        from trial_metrics import aggregate_windows_to_trials  # noqa: E402
        import sys as _sys

        m = data["split"] == "test"
        trial = aggregate_windows_to_trials(
            data["y"][m], data["pred"][m], data["subject"][m], data["trial_id"][m], n_classes=3
        )
        acc_paper = float(trial["metrics"]["acc_paper"])
        anchor = out["anchors_acc_paper"][arm]
        if abs(acc_paper - anchor) > 5e-4:
            raise SystemExit(
                f"[FAIL] arm={arm} acc_paper={acc_paper:.4f} != anchor {anchor} — 对账失败，勿用"
            )
        out[f"arm_{arm}_trial_decision"] = {
            **res,
            "acc_paper_test": acc_paper,
            "t_dec_mean_s": float(np.mean(t_decs)),
            "n_trials_timed": len(t_decs),
            "anchor_ok": True,
        }

    OUT_DEFAULT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_DEFAULT}")
    for key in ("arm_W_trial_decision", "arm_C_trial_decision", "fused_window_argmax", "shallow_member_window_argmax"):
        r = out[key]["test"]
        print(f"  {key:32s} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
