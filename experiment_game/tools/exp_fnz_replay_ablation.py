"""fnz FT 对比实验：无 replay vs OpenBMI replay；演示 heldout 划分。

输出 JSON 到 experiment_game/data/models/fnz/replay_ablation_report.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_REPO = Path(r"d:\MI")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner, ReplayPool
from adapt_engine.registry import load_head
from experiment_game.tools.ft_subject_from_v3 import (
    DEFAULT_TASK,
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    _trial_split,
    build_dataset,
    finetune_head,
)

OPENBMI_ROOT = _REPO / "code/preprocess_lab/out/openbmi_3s_hop100"
SESSIONS = [
    _REPO / "experiment_game/data/sessions/fnz_ws01_20260826_164149",
    _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537",
]
LABELS = {0: "Rest", 1: "Left", 2: "Right"}
SEED = 42


def load_openbmi_replay_pool(*, max_windows: int = 12000, seed: int = SEED) -> ReplayPool:
    """从 OpenBMI 3s hop100 训练池抽样（类均衡）。"""
    X = np.load(OPENBMI_ROOT / "openbmi_X.npy", mmap_mode="r")  # (N,1,8,750)
    y = np.load(OPENBMI_ROOT / "openbmi_y_three.npy")
    rng = np.random.default_rng(seed)
    idx_by_class: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    # 分块扫描标签，避免全量读 X
    chunk = 5000
    for s in range(0, len(y), chunk):
        ys = y[s : s + chunk]
        for c in (0, 1, 2):
            idx_by_class[c].extend((s + np.where(ys == c)[0]).tolist())
    per_class = max(1, max_windows // 3)
    pick: List[int] = []
    for c in (0, 1, 2):
        pool = idx_by_class[c]
        rng.shuffle(pool)
        pick.extend(pool[:per_class])
    rng.shuffle(pick)
    pick = pick[:max_windows]
    wins = np.stack([X[i, 0].astype(np.float32) for i in pick], axis=0)  # (M,8,750)
    labs = y[pick].astype(np.int64)
    return ReplayPool(wins, labs, seed=seed)


@torch.no_grad()
def pred_distribution(model, X: np.ndarray, y: np.ndarray, device: str) -> Dict[str, Any]:
    if len(X) == 0:
        return {}
    model.eval()
    preds, logits_all = [], []
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
    uniq, cnt = np.unique(pred, return_counts=True)
    return {
        "acc": float((pred == y).mean()),
        "pred_counts": {LABELS[int(k)]: int(v) for k, v in zip(uniq, cnt)},
        "mean_p": [float(x) for x in probs.mean(axis=0)],
        "max_class_frac": float(cnt.max() / len(pred)),
    }


def finetune_with_replay(
    ckpt: Path,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    replay_pool: ReplayPool | None,
    replay_ratio: float,
    device: str,
) -> Dict[str, Any]:
    entry = load_head(ckpt, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    acc0_tr = _eval_acc(model, X_tr, y_tr, device)
    acc0_te = _eval_acc(model, X_te, y_te, device)
    dist0 = pred_distribution(model, X_te, y_te, device)

    recipe = FTRecipe(lr=1e-4, epochs=5, batch_size=32, replay_ratio=replay_ratio, seed=SEED)
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    rec = fin.train_round(X_tr, y_tr, frozen=False)

    acc1_tr = _eval_acc(model, X_tr, y_tr, device)
    acc1_te = _eval_acc(model, X_te, y_te, device)
    dist1 = pred_distribution(model, X_te, y_te, device)
    return {
        "acc_before_train": acc0_tr,
        "acc_before_heldout": acc0_te,
        "acc_after_train": acc1_tr,
        "acc_after_heldout": acc1_te,
        "heldout_before": dist0,
        "heldout_after": dist1,
        "ft": rec,
    }


def describe_split(split_ids: np.ndarray, tr_m: np.ndarray, te_m: np.ndarray) -> Dict[str, Any]:
    tr_ids = sorted(set(split_ids[tr_m]))
    te_ids = sorted(set(split_ids[te_m]))
    return {
        "n_trials_train": len(tr_ids),
        "n_trials_heldout": len(te_ids),
        "n_windows_train": int(tr_m.sum()),
        "n_windows_heldout": int(te_m.sum()),
        "heldout_trials_sample": te_ids[:8],
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== fnz FT ablation: replay vs no replay ===")
    ds = build_dataset(SESSIONS, include_invalid=True)
    X, y3, split_ids = ds["X"], ds["y_three"], ds["split_id"]
    tr_m, te_m = _trial_split(split_ids, train_frac=0.7, seed=SEED)
    split_info = describe_split(split_ids, tr_m, te_m)
    print("Data:", len(X), "windows from ws01+ws02")
    print("Split:", split_info)

    print("\nBuilding OpenBMI replay pool (subj01-54, sess01+02, EEG_MI_train only)...")
    replay = load_openbmi_replay_pool(max_windows=9000, seed=SEED)
    print(f"  replay windows: {len(replay.windows)}  y={dict(Counter(replay.labels.tolist()))}")

    conditions = [
        ("no_replay", None, 0.0),
        ("openbmi_replay_0.20", replay, 0.20),
    ]
    results: Dict[str, Any] = {
        "subject": "fnz",
        "sessions": ds.get("sessions"),
        "split": split_info,
        "replay_source": {
            "dataset": "openbmi_3s_hop100",
            "path": str(OPENBMI_ROOT),
            "n_subjects": 54,
            "sessions_per_subject": ["sess01", "sess02"],
            "block": "EEG_MI_train only (excludes EEG_MI_test)",
            "n_total_windows": 178200,
            "replay_pool_windows": int(len(replay.windows)),
            "replay_sampling": "class-balanced random, max 9000 windows",
            "excluded_from_replay": ["Stieger", "BCI2a", "fnz own data", "EEG_MI_test"],
        },
        "heldout_explanation_zh": (
            "heldout = 微调时完全不参与梯度更新的试次/窗；"
            "按 trial 划分，同一试次所有滑窗同在 train 或 heldout，避免泄漏。"
        ),
        "conditions": {},
    }

    for name, pool, ratio in conditions:
        print(f"\n--- Condition: {name} (replay_ratio={ratio}) ---")
        rep = finetune_with_replay(
            DEFAULT_THREE,
            X[tr_m],
            y3[tr_m],
            X[te_m],
            y3[te_m],
            replay_pool=pool,
            replay_ratio=ratio,
            device=device,
        )
        print(
            f"  heldout acc: {rep['acc_before_heldout']:.3f} -> {rep['acc_after_heldout']:.3f}  "
            f"train: {rep['acc_before_train']:.3f} -> {rep['acc_after_train']:.3f}"
        )
        print(f"  heldout pred after: {rep['heldout_after'].get('pred_counts')}")
        results["conditions"][name] = rep

    out = _REPO / "experiment_game/data/models/fnz/replay_ablation_report.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
