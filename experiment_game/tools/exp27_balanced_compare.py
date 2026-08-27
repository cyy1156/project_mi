"""实验 27 补充：class-balanced batch vs 默认 batch。

对比：
  M-E1  vs M-E1-bal  (ws01+ws02, 冻 head, 无 replay)
  S-B3  vs S-B3-bal  (ws01, T0 replay 0.15, 全模型)

并做跨 session：S→ws02, M→ws03。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner, ReplayPool
from adapt_engine.registry import load_head
from experiment_game.tools.exp27_cross_session_eval import eval_ckpt_on_session, trial_majority_acc
from experiment_game.tools.exp27_fnz_replay_grid import (
    build_replay_pool,
    freeze_head_only,
    pred_distribution,
    save_winner_ckpt,
)
from experiment_game.tools.ft_subject_from_v3 import (
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    _trial_split,
    build_dataset,
)

WS01 = _REPO / "experiment_game/data/sessions/fnz_ws01_20260826_164149"
WS02 = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
WS03 = _REPO / "experiment_game/data/sessions/fnz_ws03_20260826_174526"
OUT_DIR = _REPO / "experiment_game/data/models/fnz/exp27/balanced_compare"
SEED = 42


def _train_counts(y: np.ndarray) -> Dict[str, int]:
    labels = {0: "Rest", 1: "Left", 2: "Right"}
    return {labels[int(k)]: int((y == k).sum()) for k in np.unique(y)}


def finetune(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    pool: str,
    replay_ratio: float,
    head_only: bool,
    balanced_batch: bool,
    device: str,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    replay_pool = build_replay_pool(pool) if replay_ratio > 0 else None
    entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    if head_only:
        freeze_head_only(model)

    acc0_te = _eval_acc(model, X_te, y_te, device)
    dist0 = pred_distribution(model, X_te, y_te, device)

    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=5,
        batch_size=32,
        replay_ratio=replay_ratio,
        seed=SEED,
        balanced_batch=balanced_batch,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    ft_rec = fin.train_round(X_tr, y_tr, frozen=False)

    acc1_tr = _eval_acc(model, X_tr, y_tr, device)
    acc1_te = _eval_acc(model, X_te, y_te, device)
    dist1 = pred_distribution(model, X_te, y_te, device)

    metrics = {
        "balanced_batch": balanced_batch,
        "pool": pool,
        "replay_ratio": replay_ratio,
        "head_only": head_only,
        "train_class_windows": _train_counts(y_tr),
        "acc_before_heldout": acc0_te,
        "acc_after_heldout": acc1_te,
        "acc_after_train": acc1_tr,
        "delta_heldout": acc1_te - acc0_te,
        "train_minus_heldout": acc1_tr - acc1_te,
        "heldout_before": dist0,
        "heldout_after": dist1,
        "heldout_max_class_frac": dist1.get("max_class_frac"),
        "ft": ft_rec,
    }
    return model, metrics


def save_ckpt(model: torch.nn.Module, out_dir: Path, tag: str, meta: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_best_three.pt"
    torch.save({"model": model.state_dict(), "n_outputs": 3, "tag": tag, "meta": meta}, path)
    (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "tracks": {},
    }

    # --- Track S ---
    ds_s = build_dataset([WS01], include_invalid=True)
    Xs, ys, sid_s = ds_s["X"], ds_s["y_three"], ds_s["split_id"]
    tr_s, te_s = _trial_split(sid_s, seed=SEED)
    s_cfgs = [
        ("S-B3", "t0", 0.15, False, False),
        ("S-B3-bal", "t0", 0.15, False, True),
    ]
    s_rows: Dict[str, Any] = {}
    for tag, pool, ratio, head, bal in s_cfgs:
        print(f"\n=== {tag} balanced={bal} ===", flush=True)
        model, m = finetune(
            Xs[tr_s], ys[tr_s], Xs[te_s], ys[te_s],
            pool=pool, replay_ratio=ratio, head_only=head, balanced_batch=bal, device=device,
        )
        save_ckpt(model, OUT_DIR, tag, m)
        ws02 = eval_ckpt_on_session(OUT_DIR / f"{tag}_best_three.pt", WS02, device)
        m["cross_ws02"] = ws02
        s_rows[tag] = m
        tm = ws02["trial_metrics"]
        print(
            f"  heldout {m['acc_before_heldout']:.3f}->{m['acc_after_heldout']:.3f}  "
            f"ws02 trial {tm['acc_trial_majority']:.3f} pred {tm['pred_counts']}",
            flush=True,
        )

    results["tracks"]["S"] = {
        "train_windows": _train_counts(ys[tr_s]),
        "rows": s_rows,
    }

    # --- Track M ---
    ds_m = build_dataset([WS01, WS02], include_invalid=True)
    Xm, ym, sid_m = ds_m["X"], ds_m["y_three"], ds_m["split_id"]
    tr_m, te_m = _trial_split(sid_m, seed=SEED)
    m_cfgs = [
        ("M-E1", "none", 0.0, True, False),
        ("M-E1-bal", "none", 0.0, True, True),
    ]
    m_rows: Dict[str, Any] = {}
    for tag, pool, ratio, head, bal in m_cfgs:
        print(f"\n=== {tag} balanced={bal} ===", flush=True)
        model, m = finetune(
            Xm[tr_m], ym[tr_m], Xm[te_m], ym[te_m],
            pool=pool, replay_ratio=ratio, head_only=head, balanced_batch=bal, device=device,
        )
        save_ckpt(model, OUT_DIR, tag, m)
        ws03 = eval_ckpt_on_session(OUT_DIR / f"{tag}_best_three.pt", WS03, device)
        m["cross_ws03"] = ws03
        m_rows[tag] = m
        tm = ws03["trial_metrics"]
        print(
            f"  heldout {m['acc_before_heldout']:.3f}->{m['acc_after_heldout']:.3f}  "
            f"ws03 trial {tm['acc_trial_majority']:.3f} pred {tm['pred_counts']}",
            flush=True,
        )

    results["tracks"]["M"] = {
        "train_windows": _train_counts(ym[tr_m]),
        "rows": m_rows,
    }

    out_json = OUT_DIR / "balanced_compare.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Class-balanced batch 对比",
        "",
        f"生成时间：{results['generated_at']}",
        "",
        "## Track S（ws01 训练窗分布）",
        f"`{results['tracks']['S']['train_windows']}`",
        "",
        "| 配置 | balanced | heldout after | max_frac | ws02 trial acc | ws02 pred R/L/R |",
        "|------|----------|---------------|----------|----------------|-----------------|",
    ]
    for tag, row in s_rows.items():
        ho = row["heldout_after"]
        tm = row["cross_ws02"]["trial_metrics"]
        pc = tm["pred_counts"]
        lines.append(
            f"| {tag} | {row['balanced_batch']} | {row['acc_after_heldout']:.3f} | "
            f"{row['heldout_max_class_frac']:.3f} | {tm['acc_trial_majority']:.3f} | "
            f"{pc.get('Rest',0)}/{pc.get('Left',0)}/{pc.get('Right',0)} |"
        )
    lines += [
        "",
        "## Track M（ws01+ws02 训练窗分布）",
        f"`{results['tracks']['M']['train_windows']}`",
        "",
        "| 配置 | balanced | heldout after | max_frac | ws03 trial acc | ws03 pred R/L/R |",
        "|------|----------|---------------|----------|----------------|-----------------|",
    ]
    for tag, row in m_rows.items():
        tm = row["cross_ws03"]["trial_metrics"]
        pc = tm["pred_counts"]
        lines.append(
            f"| {tag} | {row['balanced_batch']} | {row['acc_after_heldout']:.3f} | "
            f"{row['heldout_max_class_frac']:.3f} | {tm['acc_trial_majority']:.3f} | "
            f"{pc.get('Rest',0)}/{pc.get('Left',0)}/{pc.get('Right',0)} |"
        )
    (OUT_DIR / "balanced_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
