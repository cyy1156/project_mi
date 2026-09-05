"""实验 29 · BCI2a Run Ramp（固定 B2 · 在线 trial acc）。

主协议 Leave-Next-Run：R0–R5，9 被试 × 6 = 54 runs。
辅协议 Prefix→run6：R1–R5，9×5 = 45 runs。

用法:
  python experiment_game/tools/exp29_bci2a_ramp_grid.py
  python experiment_game/tools/exp29_bci2a_ramp_grid.py --protocol leave_next
  python experiment_game/tools/exp29_bci2a_ramp_grid.py --protocol prefix_final
  python experiment_game/tools/exp29_bci2a_ramp_grid.py --protocol f310
  python experiment_game/tools/exp29_bci2a_ramp_grid.py --protocol all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner
from adapt_engine.registry import load_head
from experiment_game.tools.exp27_cross_session_eval import trial_majority_acc
from experiment_game.tools.exp27_fnz_replay_grid import build_replay_pool, pred_distribution
from experiment_game.tools.ft_subject_from_v3 import (
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    _trial_split,
)

BCI2A_ROOT = _REPO / "code/preprocess_lab/out/bci2a_3s_hop100"
OUT_DIR = _REPO / "experiment_game/data/models/bci2a/exp29"
REGISTRY_MD = (
    _REPO
    / "资料/模型训练/29_旁路_bci2a被试FT_replay验证_openbmi_accpaper/总结/结果登记表.md"
)

SEED = 42
TRAIN_FRAC = 0.7
TARGET_TRIAL_ACC = 0.60
FNZ_TARGET_WINDOWS = 310  # ≈ fnz ws01 FT 池量级（全窗 ~338；train 70% ≈208）
FNZ_ONLINE_REF = 0.45  # Exp27 B2 · ws01→ws02 trial acc 参照
ALL_SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]
LABELS = {0: "Rest", 1: "Left", 2: "Right"}

_BCI2A_CACHE: Optional[Dict[str, Any]] = None


def _parse_run_num(run_name: str) -> int:
    m = re.match(r"run(\d+)", str(run_name), flags=re.I)
    return int(m.group(1)) if m else 0


def load_bci2a_bank() -> Dict[str, Any]:
    global _BCI2A_CACHE
    if _BCI2A_CACHE is not None:
        return _BCI2A_CACHE
    root = BCI2A_ROOT
    _BCI2A_CACHE = {
        "X": np.load(root / "bci2a_X.npy", mmap_mode="r"),
        "y": np.load(root / "bci2a_y_three.npy"),
        "subjects": np.load(root / "bci2a_subjects.npy", allow_pickle=True),
        "trial_id": np.load(root / "bci2a_trial_id.npy"),
    }
    return _BCI2A_CACHE


def _run_mask(subjects: np.ndarray, subject: str, run_name: str) -> np.ndarray:
    prefix = f"{subject}|{run_name}|"
    return np.array([str(s).startswith(prefix) for s in subjects])


def subject_run_list(subject: str, bank: Dict[str, Any]) -> List[str]:
    subj_arr = bank["subjects"]
    runs = sorted(
        {str(s).split("|")[1] for s in subj_arr if str(s).startswith(f"{subject}|")},
        key=_parse_run_num,
    )
    return runs


def gather_runs(
    subject: str,
    run_names: List[str],
    bank: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合并多个 run → X (N,8,750), y, split_id（试次键）。"""
    Xmm = bank["X"]
    y_all = bank["y"]
    subj_arr = bank["subjects"]
    tid_all = bank["trial_id"]
    xs, ys, sids = [], [], []
    for run_name in run_names:
        m = _run_mask(subj_arr, subject, run_name)
        if not m.any():
            continue
        idx = np.where(m)[0]
        xs.append(np.stack([Xmm[int(i), 0].astype(np.float32) for i in idx], axis=0))
        ys.append(y_all[idx])
        sids.append(np.array([f"{subject}|{run_name}|{tid_all[i]}" for i in idx], dtype=object))
    if not xs:
        empty = np.zeros((0, 8, N_TIMES), np.float32)
        return empty, np.zeros((0,), np.int64), np.array([], dtype=object)
    return np.concatenate(xs, 0), np.concatenate(ys, 0), np.concatenate(sids, 0)


def finetune_b2(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    split_ids: np.ndarray,
    *,
    device: str,
    epochs: int = 5,
) -> Dict[str, Any]:
    """从 OpenBMI 底座重训 B2；返回 heldout + 模型。"""
    replay_pool = build_replay_pool("t0", seed=SEED)
    if replay_pool is None:
        return {"status": "skipped", "reason": "empty t0 replay pool"}

    entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model

    acc0_te = _eval_acc(model, X_te, y_te, device) if len(X_te) else float("nan")

    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=32,
        replay_ratio=0.10,
        seed=SEED,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    fin.train_round(X_tr, y_tr, frozen=False)

    acc1_tr = _eval_acc(model, X_tr, y_tr, device)
    acc1_te = _eval_acc(model, X_te, y_te, device) if len(X_te) else float("nan")
    dist_te = pred_distribution(model, X_te, y_te, device) if len(X_te) else {}

    return {
        "status": "ok",
        "model": model,
        "acc_before_heldout": acc0_te,
        "acc_after_heldout": acc1_te,
        "acc_after_train": acc1_tr,
        "train_minus_heldout": acc1_tr - acc1_te if len(X_te) else float("nan"),
        "heldout_after": dist_te,
        "heldout_max_class_frac": dist_te.get("max_class_frac"),
        "ft": fin.history[-1] if fin.history else {},
    }


@torch.no_grad()
def eval_baseline_online(
    X_ev: np.ndarray,
    y_ev: np.ndarray,
    split_ids: np.ndarray,
    *,
    device: str,
) -> Dict[str, Any]:
    entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    win_acc = _eval_acc(model, X_ev, y_ev, device)
    trial = trial_majority_acc(model, X_ev, y_ev, split_ids, device)
    return {
        "acc_window": win_acc,
        "trial_metrics": trial,
        "window_metrics": pred_distribution(model, X_ev, y_ev, device),
    }


def eval_model_online(
    model,
    X_ev: np.ndarray,
    y_ev: np.ndarray,
    split_ids: np.ndarray,
    *,
    device: str,
) -> Dict[str, Any]:
    win_acc = _eval_acc(model, X_ev, y_ev, device)
    trial = trial_majority_acc(model, X_ev, y_ev, split_ids, device)
    return {
        "acc_window": win_acc,
        "trial_metrics": trial,
        "window_metrics": pred_distribution(model, X_ev, y_ev, device),
    }


def run_leave_next_point(
    subject: str,
    R: int,
    runs: List[str],
    bank: Dict[str, Any],
    *,
    device: str,
    epochs: int,
) -> Dict[str, Any]:
    if len(runs) < 2:
        return {"status": "skipped", "reason": "fewer than 2 runs"}

    eval_idx = R if R >= 1 else 1
    if eval_idx >= len(runs):
        return {"status": "skipped", "reason": f"eval_idx {eval_idx} >= n_runs {len(runs)}"}

    eval_run = runs[eval_idx]
    X_ev, y_ev, sid_ev = gather_runs(subject, [eval_run], bank)

    row: Dict[str, Any] = {
        "protocol": "leave_next",
        "subject": subject,
        "R": R,
        "eval_run": eval_run,
        "train_runs": [],
        "n_windows_train": 0,
        "n_windows_eval": int(len(X_ev)),
    }

    if R == 0:
        online = eval_baseline_online(X_ev, y_ev, sid_ev, device=device)
        row.update(
            {
                "status": "ok",
                "ft": False,
                "online_trial_acc": online["trial_metrics"]["acc_trial_majority"],
                "online_window_acc": online["acc_window"],
                "online_pred_counts": online["trial_metrics"]["pred_counts"],
                "online_max_class_frac": online["trial_metrics"]["max_class_frac"],
                "online_n_trials": online["trial_metrics"]["n_trials"],
            }
        )
        return row

    train_runs = runs[:R]
    X_all, y_all, split_ids = gather_runs(subject, train_runs, bank)
    tr_m, te_m = _trial_split(split_ids, train_frac=TRAIN_FRAC, seed=SEED)
    row["train_runs"] = train_runs
    row["n_windows_train"] = int(tr_m.sum())
    row["n_windows_heldout"] = int(te_m.sum())

    ft_out = finetune_b2(
        X_all[tr_m], y_all[tr_m], X_all[te_m], y_all[te_m], split_ids[tr_m],
        device=device, epochs=epochs,
    )
    if ft_out.get("status") != "ok":
        row.update(ft_out)
        return row

    online = eval_model_online(ft_out["model"], X_ev, y_ev, sid_ev, device=device)
    row.update(
        {
            "status": "ok",
            "ft": True,
            "arm": "B2",
            "epochs": epochs,
            "acc_before_heldout": ft_out["acc_before_heldout"],
            "acc_after_heldout": ft_out["acc_after_heldout"],
            "acc_after_train": ft_out["acc_after_train"],
            "train_minus_heldout": ft_out["train_minus_heldout"],
            "heldout_max_class_frac": ft_out["heldout_max_class_frac"],
            "heldout_pred_counts": (ft_out.get("heldout_after") or {}).get("pred_counts"),
            "online_trial_acc": online["trial_metrics"]["acc_trial_majority"],
            "online_window_acc": online["acc_window"],
            "online_pred_counts": online["trial_metrics"]["pred_counts"],
            "online_max_class_frac": online["trial_metrics"]["max_class_frac"],
            "online_n_trials": online["trial_metrics"]["n_trials"],
        }
    )
    return row


def subsample_trial_preserving(
    X: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
    target_n: int,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """按试次抽取至窗数 ≥ target_n（尽量接近 fnz 量级），保持试次完整。"""
    if len(X) <= target_n:
        return X, y, split_ids, len(X)
    trials = list(np.unique(split_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(trials)
    picked: List[Any] = []
    n_win = 0
    for t in trials:
        m = split_ids == t
        nw = int(m.sum())
        if n_win + nw > target_n and n_win > 0:
            continue
        picked.append(t)
        n_win += nw
        if n_win >= target_n:
            break
    if not picked:
        picked = [trials[0]]
    mask = np.isin(split_ids, picked)
    return X[mask], y[mask], split_ids[mask], int(mask.sum())


def run_f310_point(
    subject: str,
    R: int,
    runs: List[str],
    bank: Dict[str, Any],
    *,
    device: str,
    epochs: int,
    target_n: int = FNZ_TARGET_WINDOWS,
) -> Dict[str, Any]:
    """F@R：从 R 个 run 的训练池下采样至 ~310 窗，Leave-Next 评估。"""
    if R < 1 or R >= len(runs):
        return {"status": "skipped", "reason": f"invalid R={R} for n_runs={len(runs)}"}
    eval_run = runs[R]
    train_runs = runs[:R]
    X_ev, y_ev, sid_ev = gather_runs(subject, [eval_run], bank)
    X_all, y_all, split_ids = gather_runs(subject, train_runs, bank)

    subj_seed = SEED + sum(ord(c) for c in subject) + R * 17
    X_sub, y_sub, sid_sub, n_sub = subsample_trial_preserving(
        X_all, y_all, split_ids, target_n, seed=subj_seed,
    )

    row: Dict[str, Any] = {
        "protocol": "f310",
        "subject": subject,
        "R": R,
        "eval_run": eval_run,
        "train_runs": train_runs,
        "n_windows_pool_full": int(len(X_all)),
        "n_windows_pool_subsampled": n_sub,
        "target_windows": target_n,
        "n_windows_eval": int(len(X_ev)),
    }

    tr_m, te_m = _trial_split(sid_sub, train_frac=TRAIN_FRAC, seed=SEED)
    row["n_windows_train"] = int(tr_m.sum())
    row["n_windows_heldout"] = int(te_m.sum())

    ft_out = finetune_b2(
        X_sub[tr_m], y_sub[tr_m], X_sub[te_m], y_sub[te_m], sid_sub[tr_m],
        device=device, epochs=epochs,
    )
    if ft_out.get("status") != "ok":
        row.update(ft_out)
        return row

    online = eval_model_online(ft_out["model"], X_ev, y_ev, sid_ev, device=device)
    row.update(
        {
            "status": "ok",
            "ft": True,
            "arm": "F-B2",
            "epochs": epochs,
            "acc_after_heldout": ft_out["acc_after_heldout"],
            "online_trial_acc": online["trial_metrics"]["acc_trial_majority"],
            "online_window_acc": online["acc_window"],
            "online_pred_counts": online["trial_metrics"]["pred_counts"],
            "online_max_class_frac": online["trial_metrics"]["max_class_frac"],
            "online_n_trials": online["trial_metrics"]["n_trials"],
        }
    )
    return row


def load_full_ramp_acc(R: int) -> Dict[str, float]:
    """从 ramp_leave_next.json 读取全量 B2 在线 acc。"""
    path = OUT_DIR / "ramp_leave_next.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    for row in data.get("rows", []):
        if row.get("status") == "ok" and int(row.get("R", -1)) == R and row.get("ft"):
            out[str(row["subject"])] = float(row["online_trial_acc"])
    return out


def summarize_f310(rows: List[Dict[str, Any]], R_values: List[int]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_R: Dict[int, List[float]] = defaultdict(list)
    for r in ok:
        by_R[int(r["R"])].append(float(r["online_trial_acc"]))

    compare: Dict[str, Dict[str, Any]] = {}
    for R in R_values:
        full = load_full_ramp_acc(R)
        f_acc = {r["subject"]: float(r["online_trial_acc"]) for r in ok if int(r["R"]) == R}
        subs = sorted(set(full) | set(f_acc))
        rows_cmp = []
        for s in subs:
            fv = full.get(s)
            sv = f_acc.get(s)
            rows_cmp.append(
                {
                    "subject": s,
                    "full_trial_acc": fv,
                    "f310_trial_acc": sv,
                    "delta_f_minus_full": (sv - fv) if fv is not None and sv is not None else None,
                }
            )
        mean_full = float(np.mean([full[s] for s in subs if s in full])) if full else float("nan")
        mean_f = float(np.mean([f_acc[s] for s in subs if s in f_acc])) if f_acc else float("nan")
        compare[str(R)] = {
            "mean_full": mean_full,
            "mean_f310": mean_f,
            "delta_mean": mean_f - mean_full,
            "fnz_ref": FNZ_ONLINE_REF,
            "per_subject": rows_cmp,
        }

    return {
        "target_windows": FNZ_TARGET_WINDOWS,
        "mean_f310_by_R": {str(R): float(np.mean(v)) for R, v in sorted(by_R.items())},
        "compare_to_full": compare,
        "n_ok": len(ok),
    }


def write_f310_md(payload: Dict[str, Any], path: Path) -> None:
    summ = payload.get("summary", {})
    lines = [
        "# 实验 29 · F@~310 窗 fnz 量级对照",
        "",
        f"生成时间：{payload['generated_at']}",
        f"下采样目标：**{FNZ_TARGET_WINDOWS} 窗**（试次级保留）· B2 · epochs={payload.get('epochs', 5)}",
        f"fnz Exp27 在线参照：**{FNZ_ONLINE_REF:.0%}**",
        "",
    ]
    for R, block in summ.get("compare_to_full", {}).items():
        lines += [
            f"## R{R}（Leave-Next · eval run_{int(R)+1}）",
            "",
            f"- 全量 mean trial acc：**{block['mean_full']:.3f}**",
            f"- F@310 mean trial acc：**{block['mean_f310']:.3f}**（Δ {block['delta_mean']:+.3f}）",
            f"- fnz 参照：{block['fnz_ref']:.3f}",
            "",
            "| subject | 全量 B2 | F@310 | Δ |",
            "|---------|---------|-------|---|",
        ]
        for r in block.get("per_subject", []):
            fv = r["full_trial_acc"]
            sv = r["f310_trial_acc"]
            d = r["delta_f_minus_full"]
            lines.append(
                f"| {r['subject']} | {fv:.3f} | {sv:.3f} | {d:+.3f} |"
                if fv is not None and sv is not None and d is not None
                else f"| {r['subject']} | — | — | — |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_f310(
    *,
    subjects: List[str],
    device: str,
    epochs: int,
    R_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    bank = load_bci2a_bank()
    R_list = R_values if R_values is not None else [2, 4]
    rows: List[Dict[str, Any]] = []
    for subject in subjects:
        runs = subject_run_list(subject, bank)
        print(f"\n=== {subject} F@310 ===", flush=True)
        for R in R_list:
            row = run_f310_point(subject, R, runs, bank, device=device, epochs=epochs)
            rows.append(row)
            if row.get("status") == "ok":
                print(
                    f"  F@R{R} pool {row['n_windows_pool_full']}→{row['n_windows_pool_subsampled']} "
                    f"train={row['n_windows_train']} trial_acc={row['online_trial_acc']:.3f}",
                    flush=True,
                )
            else:
                print(f"  F@R{R} SKIP: {row.get('reason')}", flush=True)
    return {
        "experiment": "E29",
        "protocol": "f310",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "epochs": epochs,
        "arm": "F-B2",
        "replay": "t0@0.10",
        "target_windows": FNZ_TARGET_WINDOWS,
        "subjects": subjects,
        "R_values": R_list,
        "rows": rows,
        "summary": summarize_f310(rows, R_list),
    }


def run_prefix_final_point(
    subject: str,
    R: int,
    runs: List[str],
    bank: Dict[str, Any],
    *,
    device: str,
    epochs: int,
) -> Dict[str, Any]:
    if len(runs) < 6:
        return {"status": "skipped", "reason": "need 6 runs"}
    eval_run = runs[-1]
    train_runs = runs[:R]
    X_ev, y_ev, sid_ev = gather_runs(subject, [eval_run], bank)
    X_all, y_all, split_ids = gather_runs(subject, train_runs, bank)

    row: Dict[str, Any] = {
        "protocol": "prefix_final",
        "subject": subject,
        "R": R,
        "eval_run": eval_run,
        "train_runs": train_runs,
        "n_windows_train": 0,
        "n_windows_eval": int(len(X_ev)),
    }

    tr_m, te_m = _trial_split(split_ids, train_frac=TRAIN_FRAC, seed=SEED)
    row["n_windows_train"] = int(tr_m.sum())
    row["n_windows_heldout"] = int(te_m.sum())

    ft_out = finetune_b2(
        X_all[tr_m], y_all[tr_m], X_all[te_m], y_all[te_m], split_ids[tr_m],
        device=device, epochs=epochs,
    )
    if ft_out.get("status") != "ok":
        row.update(ft_out)
        return row

    online = eval_model_online(ft_out["model"], X_ev, y_ev, sid_ev, device=device)
    row.update(
        {
            "status": "ok",
            "ft": True,
            "arm": "B2",
            "epochs": epochs,
            "acc_after_heldout": ft_out["acc_after_heldout"],
            "online_trial_acc": online["trial_metrics"]["acc_trial_majority"],
            "online_window_acc": online["acc_window"],
            "online_pred_counts": online["trial_metrics"]["pred_counts"],
            "online_max_class_frac": online["trial_metrics"]["max_class_frac"],
            "online_n_trials": online["trial_metrics"]["n_trials"],
        }
    )
    return row


def summarize_ramp(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_R: Dict[int, List[float]] = defaultdict(list)
    for r in ok:
        by_R[int(r["R"])].append(float(r["online_trial_acc"]))

    mean_by_R = {str(R): float(np.mean(v)) for R, v in sorted(by_R.items())}
    first_R_mean_60: Optional[int] = None
    for R, v in sorted(by_R.items()):
        if np.mean(v) >= TARGET_TRIAL_ACC:
            first_R_mean_60 = int(R)
            break

    n_subj_ge60 = defaultdict(int)
    for r in ok:
        if float(r["online_trial_acc"]) >= TARGET_TRIAL_ACC:
            n_subj_ge60[int(r["R"])] += 1

    per_subject: Dict[str, Dict[str, Any]] = {}
    for r in ok:
        subj = r["subject"]
        if subj not in per_subject:
            per_subject[subj] = {"by_R": {}, "first_R_ge60": None, "peak_acc": 0.0, "peak_R": None}
        R = int(r["R"])
        acc = float(r["online_trial_acc"])
        per_subject[subj]["by_R"][str(R)] = acc
        if acc >= TARGET_TRIAL_ACC and per_subject[subj]["first_R_ge60"] is None:
            per_subject[subj]["first_R_ge60"] = R
        if acc > per_subject[subj]["peak_acc"]:
            per_subject[subj]["peak_acc"] = acc
            per_subject[subj]["peak_R"] = R

    return {
        "target_trial_acc": TARGET_TRIAL_ACC,
        "mean_online_trial_acc_by_R": mean_by_R,
        "first_R_queue_mean_ge60": first_R_mean_60,
        "n_subjects_ge60_by_R": {str(k): v for k, v in sorted(n_subj_ge60.items())},
        "per_subject": per_subject,
        "n_ok": len(ok),
    }


def write_ramp_md(payload: Dict[str, Any], path: Path) -> None:
    summ = payload.get("summary", {})
    lines = [
        "# 实验 29 · BCI2a Run Ramp 报告",
        "",
        f"生成时间：{payload['generated_at']}",
        f"协议：{payload['protocol']}",
        f"策略：B2 · T0 replay 0.10 · epochs={payload.get('epochs', 5)}",
        f"目标：在线 **trial acc ≥ {TARGET_TRIAL_ACC:.0%}**",
        "",
        "## 队列均值（trial acc）",
        "",
        "| R | mean trial acc | ≥60% 人数 |",
        "|---|----------------|-----------|",
    ]
    mean_by_R = summ.get("mean_online_trial_acc_by_R", {})
    n60 = summ.get("n_subjects_ge60_by_R", {})
    for R in sorted(mean_by_R.keys(), key=int):
        m = mean_by_R[R]
        flag = "**✓**" if m >= TARGET_TRIAL_ACC else ""
        lines.append(f"| {R} | {m:.3f} {flag} | {n60.get(R, 0)}/9 |")
    lines += [
        "",
        f"**首次队列均值 ≥60% 的 R：** {summ.get('first_R_queue_mean_ge60', '—')}",
        "",
        "## 分被试",
        "",
        "| subject | R0 | R1 | R2 | R3 | R4 | R5 | 首次≥60% | 峰值 |",
        "|---------|----|----|----|----|----|----|----------|------|",
    ]
    for subj in sorted(summ.get("per_subject", {}).keys()):
        ps = summ["per_subject"][subj]
        by_R = ps.get("by_R", {})
        cells = [f"{by_R.get(str(i), float('nan')):.3f}" if str(i) in by_R else "—" for i in range(6)]
        lines.append(
            f"| {subj} | {' | '.join(cells)} | {ps.get('first_R_ge60', '—')} | "
            f"{ps.get('peak_acc', 0):.3f}@R{ps.get('peak_R', '—')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_registry_md(payload: Dict[str, Any]) -> None:
    if not REGISTRY_MD.is_file():
        return
    summ = payload.get("summary", {})
    per = summ.get("per_subject", {})
    lines = REGISTRY_MD.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    in_table = False
    for line in lines:
        if line.startswith("| subject | R0 |"):
            in_table = True
            out.append(line)
            continue
        if in_table and line.startswith("| A"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            subj = parts[0]
            if subj in per:
                by_R = per[subj].get("by_R", {})
                cells = [
                    f"{by_R.get(str(i), float('nan')):.3f}" if str(i) in by_R else "—"
                    for i in range(6)
                ]
                first = per[subj].get("first_R_ge60", "—")
                peak = f"{per[subj].get('peak_acc', 0):.3f}"
                line = f"| {subj} | {' | '.join(cells)} | {first} | {peak} |"
            out.append(line)
            continue
        if in_table and line.startswith("| **均值**"):
            mean_by_R = summ.get("mean_online_trial_acc_by_R", {})
            cells = [f"{mean_by_R.get(str(i), float('nan')):.3f}" for i in range(6)]
            first = summ.get("first_R_queue_mean_ge60", "—")
            out.append(f"| **均值** | {' | '.join(cells)} | {first} | — |")
            in_table = False
            continue
        out.append(line)
    REGISTRY_MD.write_text("\n".join(out) + "\n", encoding="utf-8")


def run_ramp(
    *,
    protocol: str,
    subjects: List[str],
    device: str,
    epochs: int,
    R_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    bank = load_bci2a_bank()
    rows: List[Dict[str, Any]] = []

    if protocol == "leave_next":
        R_list = R_values if R_values is not None else list(range(6))
    elif protocol == "prefix_final":
        R_list = R_values if R_values is not None else list(range(1, 6))
    else:
        raise ValueError(protocol)

    for subject in subjects:
        runs = subject_run_list(subject, bank)
        print(f"\n=== {subject} runs={runs} ===", flush=True)
        for R in R_list:
            if protocol == "leave_next":
                row = run_leave_next_point(subject, R, runs, bank, device=device, epochs=epochs)
            else:
                row = run_prefix_final_point(subject, R, runs, bank, device=device, epochs=epochs)
            rows.append(row)
            if row.get("status") == "ok":
                print(
                    f"  R{R} trial_acc={row['online_trial_acc']:.3f} "
                    f"pred={row.get('online_pred_counts')} "
                    f"train_runs={row.get('train_runs', [])} eval={row.get('eval_run')}",
                    flush=True,
                )
            else:
                print(f"  R{R} SKIP: {row.get('reason')}", flush=True)

    payload = {
        "experiment": "E29",
        "protocol": protocol,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "epochs": epochs,
        "arm": "B2",
        "replay": "t0@0.10",
        "subjects": subjects,
        "rows": rows,
        "summary": summarize_ramp(rows),
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 29 BCI2a Run Ramp")
    ap.add_argument(
        "--protocol",
        choices=("leave_next", "prefix_final", "f310", "both", "all"),
        default="leave_next",
    )
    ap.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="仅 A01 · R0,R1,R2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]
    R_smoke = [0, 1, 2] if args.smoke else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.protocol == "f310":
        print(f"\n######## Protocol: f310 · device={device} ########", flush=True)
        payload = run_f310(subjects=subjects, device=device, epochs=args.epochs)
        out_json = OUT_DIR / "f310.json"
        out_md = OUT_DIR / "f310.md"
        out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        write_f310_md(payload, out_md)
        print(f"\nWrote {out_json}")
        for R, block in payload["summary"]["compare_to_full"].items():
            print(
                f"R{R}: full={block['mean_full']:.3f} f310={block['mean_f310']:.3f} "
                f"delta={block['delta_mean']:+.3f} fnz_ref={FNZ_ONLINE_REF:.3f}",
                flush=True,
            )
        return

    protocols = (
        ["leave_next", "prefix_final", "f310"]
        if args.protocol == "all"
        else (["leave_next", "prefix_final"] if args.protocol == "both" else [args.protocol])
    )

    for proto in protocols:
        if proto == "f310":
            print(f"\n######## Protocol: f310 · device={device} ########", flush=True)
            payload = run_f310(subjects=subjects, device=device, epochs=args.epochs)
            out_json = OUT_DIR / "f310.json"
            out_md = OUT_DIR / "f310.md"
            out_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            write_f310_md(payload, out_md)
            print(f"\nWrote {out_json}")
            continue
        print(f"\n######## Protocol: {proto} · device={device} ########", flush=True)
        payload = run_ramp(
            protocol=proto,
            subjects=subjects,
            device=device,
            epochs=args.epochs,
            R_values=R_smoke,
        )
        out_json = OUT_DIR / f"ramp_{proto}.json"
        out_md = OUT_DIR / f"ramp_{proto}.md"
        out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        write_ramp_md(payload, out_md)
        if proto == "leave_next":
            update_registry_md(payload)
        print(f"\nWrote {out_json}")
        summ = payload["summary"]
        print(f"Mean trial acc by R: {summ['mean_online_trial_acc_by_R']}")
        print(f"First R with queue mean >=60%: {summ.get('first_R_queue_mean_ge60')}")


if __name__ == "__main__":
    main()
