"""实验 29 延伸：BCI2a task 串行门控 vs 纯 three（Leave-Next 关键点 R0/R2/R4/R5）。

在 Exp29 Ramp 各点训练匹配的 task 头（B2 · T0 task replay 0.10），
于评估 run 上 sweep task_p_on，比较试次级多数票 acc。

用法:
  python experiment_game/tools/exp29_bci2a_serial_gating.py
  python experiment_game/tools/exp29_bci2a_serial_gating.py --R-values 0,2,4,5
  python experiment_game/tools/exp29_bci2a_serial_gating.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner
from adapt_engine.readout import serial_gating
from adapt_engine.registry import load_head
from experiment_game.tools.exp29_bci2a_ramp_grid import (
    ALL_SUBJECTS,
    OUT_DIR,
    SEED,
    TRAIN_FRAC,
    finetune_b2,
    gather_runs,
    load_bci2a_bank,
    subject_run_list,
)
from experiment_game.tools.ft_subject_from_v3 import (
    DEFAULT_TASK,
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    _trial_split,
)
from experiment_game.tools.openbmi_replay_pool import build_t0_task_replay_pool, three_labels_to_task

OUT_JSON = OUT_DIR / "serial_gating_eval.json"
OUT_MD = OUT_DIR / "serial_gating_eval.md"

LABELS = {0: "Rest", 1: "Left", 2: "Right"}
TASK_P_ON_GRID = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]
DEFAULT_R_VALUES = [0, 2, 4, 5]


def _forward_probs(model, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    bs = 64
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(X[s : s + bs]).to(device)
            try:
                logits = model(xb)
            except RuntimeError:
                logits = model(xb.unsqueeze(1))
            if logits.dim() == 3:
                logits = logits.reshape(logits.shape[0], -1)
            chunks.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def finetune_b2_task(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    device: str,
    epochs: int = 5,
) -> Dict[str, Any]:
    """从 OpenBMI task 底座重训 B2 task 头。"""
    replay_pool = build_t0_task_replay_pool(seed=SEED)
    if replay_pool is None:
        return {"status": "skipped", "reason": "empty t0 task replay pool"}

    y2_tr = three_labels_to_task(y_tr)
    y2_te = three_labels_to_task(y_te)

    entry = load_head(DEFAULT_TASK, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    acc0_te = _eval_acc(model, X_te, y2_te, device) if len(X_te) else float("nan")

    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=32,
        replay_ratio=0.10,
        seed=SEED,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    fin.train_round(X_tr, y2_tr, frozen=False)

    acc1_te = _eval_acc(model, X_te, y2_te, device) if len(X_te) else float("nan")
    return {
        "status": "ok",
        "model": model,
        "acc_before_heldout": acc0_te,
        "acc_after_heldout": acc1_te,
        "ft": fin.history[-1] if fin.history else {},
    }


def eval_serial_gating(
    task_model,
    three_model,
    X: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
    *,
    device: str,
    task_p_on_grid: List[float],
) -> Dict[str, Any]:
    p_task = _forward_probs(task_model, X, device)
    p_three = _forward_probs(three_model, X, device)

    window_rows: Dict[str, Any] = {}
    trial_rows: Dict[str, Any] = {}

    for tpo in task_p_on_grid:
        preds: List[int] = []
        gated_frac = 0.0
        for i in range(len(X)):
            if tpo <= 0.0:
                pred = int(np.argmax(p_three[i]))
                gated = False
            else:
                out = serial_gating(p_task[i], p_three[i], task_p_on=tpo)
                pred = int(out["pred"])
                gated = bool(out["gated"])
            preds.append(pred)
            if gated:
                gated_frac += 1

        preds_arr = np.asarray(preds, dtype=np.int64)
        win_acc = float((preds_arr == y).mean())

        by_trial: Dict[str, List[int]] = defaultdict(list)
        by_label: Dict[str, int] = {}
        for i, sid in enumerate(split_ids):
            sid = str(sid)
            by_trial[sid].append(int(preds_arr[i]))
            by_label[sid] = int(y[i])

        trial_pred, trial_true = [], []
        for sid, ps in by_trial.items():
            cnt = Counter(ps)
            top = cnt.most_common()
            pred = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else top[0][0]
            trial_pred.append(pred)
            trial_true.append(by_label[sid])

        trial_acc = float(np.mean(np.array(trial_pred) == np.array(trial_true)))
        uniq, cnt = np.unique(trial_pred, return_counts=True)
        pred_counts = {LABELS[int(k)]: int(v) for k, v in zip(uniq, cnt)}
        max_frac = float(cnt.max() / len(trial_pred)) if trial_pred else 0.0

        key = f"{tpo:.1f}"
        window_rows[key] = {
            "task_p_on": tpo,
            "acc_window": win_acc,
            "gated_frac": gated_frac / max(len(X), 1),
        }
        trial_rows[key] = {
            "task_p_on": tpo,
            "acc_trial_majority": trial_acc,
            "pred_counts": pred_counts,
            "max_class_frac": max_frac,
        }

    best_tpo = max(task_p_on_grid, key=lambda t: trial_rows[f"{t:.1f}"]["acc_trial_majority"])
    baseline = trial_rows["0.0"]["acc_trial_majority"]
    best_acc = trial_rows[f"{best_tpo:.1f}"]["acc_trial_majority"]

    return {
        "n_windows": int(len(X)),
        "n_trials": len(set(map(str, split_ids))),
        "by_task_p_on": trial_rows,
        "window_by_task_p_on": window_rows,
        "baseline_three_only_trial_acc": baseline,
        "best_task_p_on": best_tpo,
        "best_gated_trial_acc": best_acc,
        "delta_vs_three_only": best_acc - baseline,
    }


def _load_base_models(device: str) -> Tuple[Any, Any]:
    three_entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    task_entry = load_head(DEFAULT_TASK, n_chans=8, n_times=N_TIMES, device=device)
    return task_entry.model, three_entry.model


def run_gating_point(
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
        "protocol": "leave_next_gating",
        "subject": subject,
        "R": R,
        "eval_run": eval_run,
        "train_runs": [],
        "n_windows_eval": int(len(X_ev)),
    }

    if R == 0:
        task_model, three_model = _load_base_models(device)
        row["ft"] = False
    else:
        train_runs = runs[:R]
        X_all, y_all, split_ids = gather_runs(subject, train_runs, bank)
        tr_m, te_m = _trial_split(split_ids, train_frac=TRAIN_FRAC, seed=SEED)
        row["train_runs"] = train_runs
        row["n_windows_train"] = int(tr_m.sum())

        three_out = finetune_b2(
            X_all[tr_m], y_all[tr_m], X_all[te_m], y_all[te_m], split_ids[tr_m],
            device=device, epochs=epochs,
        )
        if three_out.get("status") != "ok":
            row.update(three_out)
            return row

        task_out = finetune_b2_task(
            X_all[tr_m], y_all[tr_m], X_all[te_m], y_all[te_m],
            device=device, epochs=epochs,
        )
        if task_out.get("status") != "ok":
            row.update(task_out)
            return row

        three_model = three_out["model"]
        task_model = task_out["model"]
        row["ft"] = True
        row["three_heldout_after"] = three_out["acc_after_heldout"]
        row["task_heldout_after"] = task_out["acc_after_heldout"]

    gating = eval_serial_gating(
        task_model, three_model, X_ev, y_ev, sid_ev,
        device=device, task_p_on_grid=TASK_P_ON_GRID,
    )
    row.update({"status": "ok", **gating})
    return row


def summarize_by_R(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_R: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in ok:
        by_R[int(r["R"])].append(r)

    summary: Dict[str, Any] = {}
    for R in sorted(by_R.keys()):
        pts = by_R[R]
        three_accs = [float(p["baseline_three_only_trial_acc"]) for p in pts]
        best_accs = [float(p["best_gated_trial_acc"]) for p in pts]
        deltas = [float(p["delta_vs_three_only"]) for p in pts]
        best_tpos = [float(p["best_task_p_on"]) for p in pts]
        n_help = sum(1 for d in deltas if d > 0.02)
        n_hurt = sum(1 for d in deltas if d < -0.02)
        summary[str(R)] = {
            "n_subjects": len(pts),
            "mean_three_only": float(np.mean(three_accs)),
            "mean_best_gated": float(np.mean(best_accs)),
            "mean_delta": float(np.mean(deltas)),
            "median_best_task_p_on": float(np.median(best_tpos)),
            "n_subjects_gating_helps_gt2pp": n_help,
            "n_subjects_gating_hurts_gt2pp": n_hurt,
            "verdict": (
                "有帮助" if float(np.mean(deltas)) > 0.02
                else ("略有害" if float(np.mean(deltas)) < -0.02 else "基本无差")
            ),
        }
    return summary


def write_md(payload: Dict[str, Any], path: Path) -> None:
    summ = payload.get("summary_by_R", {})
    lines = [
        "# 实验 29 · BCI2a task 串行门控评测",
        "",
        f"生成：{payload['generated_at']}",
        "",
        "口径：Leave-Next 评估 run 全窗推理；pred = serial_gating(p_task, p_three, task_p_on)；",
        "报告 **试次级多数票** acc。task_p_on=0 等价于纯 three argmax。",
        f"FT：B2 · T0 replay 0.10 · epochs={payload.get('epochs', 5)}（R>0 时 three+task 各训一头）",
        "",
        "## 队列汇总（按 R）",
        "",
        "| R | mean three only | mean best gated | mean Δ | best task_p_on 中位 | ≥+2pp 人数 | 结论 |",
        "|---|-----------------|-----------------|--------|---------------------|------------|------|",
    ]
    for R in sorted(summ.keys(), key=int):
        s = summ[R]
        lines.append(
            f"| {R} | {s['mean_three_only']:.3f} | {s['mean_best_gated']:.3f} | "
            f"{s['mean_delta']:+.3f} | {s['median_best_task_p_on']:.1f} | "
            f"{s['n_subjects_gating_helps_gt2pp']}/9 | {s['verdict']} |"
        )

    lines += ["", "## 分被试 · 分 R", ""]
    for R in sorted(summ.keys(), key=int):
        pts = [r for r in payload["rows"] if r.get("status") == "ok" and int(r["R"]) == int(R)]
        lines += [
            f"### R{R}",
            "",
            "| subject | three only | best gated | Δ | best task_p_on | pred R/L/R @0.0 | pred @best |",
            "|---------|------------|------------|---|----------------|-----------------|------------|",
        ]
        for p in sorted(pts, key=lambda x: x["subject"]):
            b = p["by_task_p_on"]["0.0"]
            k = f"{p['best_task_p_on']:.1f}"
            g = p["by_task_p_on"][k]
            pc0 = b["pred_counts"]
            pcg = g["pred_counts"]
            lines.append(
                f"| {p['subject']} | {p['baseline_three_only_trial_acc']:.3f} | "
                f"{p['best_gated_trial_acc']:.3f} | {p['delta_vs_three_only']:+.3f} | "
                f"{p['best_task_p_on']:.1f} | "
                f"{pc0.get('Rest', 0)}/{pc0.get('Left', 0)}/{pc0.get('Right', 0)} | "
                f"{pcg.get('Rest', 0)}/{pcg.get('Left', 0)}/{pcg.get('Right', 0)} |"
            )
        lines.append("")

    lines += [
        "## 与 Exp27 fnz 对照",
        "",
        "| 场景 | Exp27 three only | Exp27 best gated | Exp29 对应 R |",
        "|------|------------------|------------------|--------------|",
        "| fnz S_B2 → ws02 | 0.450 | 0.450 (Δ=0) | ≈ R2 单 run FT |",
        "| fnz M_B2 → ws03 | 0.417 | 0.444 (Δ=+0.028 @0.3) | ≈ R4–R5 多 run FT |",
        "",
        "**部署建议不变：** v3 默认 `task_p_on=0`；若 BCI2a 高 acc 段仍无队列级收益，则 fnz 不必默认开门控。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="E29 BCI2a serial gating eval")
    ap.add_argument("--R-values", default=",".join(str(r) for r in DEFAULT_R_VALUES))
    ap.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="仅 A01 · R0,R2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]
    if args.smoke:
        subjects = ["A01"]
        R_list = [0, 2]
    else:
        R_list = [int(x.strip()) for x in args.R_values.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bank = load_bci2a_bank()
    rows: List[Dict[str, Any]] = []

    for subject in subjects:
        runs = subject_run_list(subject, bank)
        print(f"\n=== {subject} runs={runs} ===", flush=True)
        for R in R_list:
            row = run_gating_point(subject, R, runs, bank, device=device, epochs=args.epochs)
            rows.append(row)
            if row.get("status") == "ok":
                print(
                    f"  R{R} three_only={row['baseline_three_only_trial_acc']:.3f} "
                    f"best={row['best_gated_trial_acc']:.3f} @ {row['best_task_p_on']:.1f} "
                    f"Δ={row['delta_vs_three_only']:+.3f}",
                    flush=True,
                )
            else:
                print(f"  R{R} SKIP: {row.get('reason')}", flush=True)

    payload = {
        "experiment": "E29",
        "protocol": "leave_next_serial_gating",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "seed": SEED,
        "epochs": args.epochs,
        "arm": "B2",
        "replay": "t0@0.10",
        "task_p_on_grid": TASK_P_ON_GRID,
        "R_values": R_list,
        "subjects": subjects,
        "rows": rows,
        "summary_by_R": summarize_by_R(rows),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_md(payload, OUT_MD)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    for R, s in sorted(payload["summary_by_R"].items(), key=lambda x: int(x[0])):
        print(
            f"R{R}: three={s['mean_three_only']:.3f} gated={s['mean_best_gated']:.3f} "
            f"Δ={s['mean_delta']:+.3f} ({s['verdict']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
