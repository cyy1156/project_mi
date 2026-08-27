"""实验 29 延伸：无 replay 对照（A0）vs B2（T0 replay 0.10）。

Leave-Next R1–R5：全模型 FT、无 replay，与既有 ramp_leave_next.json（B2）对比。

用法:
  python experiment_game/tools/exp29_bci2a_noreplay_control.py
  python experiment_game/tools/exp29_bci2a_noreplay_control.py --R-values 2,4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner
from adapt_engine.registry import load_head
from experiment_game.tools.exp27_cross_session_eval import trial_majority_acc
from experiment_game.tools.exp27_fnz_replay_grid import pred_distribution
from experiment_game.tools.exp29_bci2a_ramp_grid import (
    ALL_SUBJECTS,
    OUT_DIR,
    SEED,
    TRAIN_FRAC,
    eval_model_online,
    gather_runs,
    load_bci2a_bank,
    subject_run_list,
)
from experiment_game.tools.ft_subject_from_v3 import DEFAULT_THREE, N_TIMES, _eval_acc, _trial_split

OUT_JSON = OUT_DIR / "noreplay_a0.json"
OUT_MD = OUT_DIR / "noreplay_a0.md"
B2_JSON = OUT_DIR / "ramp_leave_next.json"
DEFAULT_R = list(range(1, 6))


def finetune_a0(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    device: str,
    epochs: int = 5,
) -> Dict[str, Any]:
    """A0：全模型 FT、无 replay（对齐 Exp27 A0）。"""
    entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    acc0_te = _eval_acc(model, X_te, y_te, device) if len(X_te) else float("nan")

    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=32,
        replay_ratio=0.0,
        seed=SEED,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=None, device=device)
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
        "heldout_pred_counts": dist_te.get("pred_counts"),
        "ft": fin.history[-1] if fin.history else {},
    }


def run_a0_point(
    subject: str,
    R: int,
    runs: List[str],
    bank: Dict[str, Any],
    *,
    device: str,
    epochs: int,
) -> Dict[str, Any]:
    if R < 1 or len(runs) < 2:
        return {"status": "skipped", "reason": "R<1 or fewer than 2 runs"}

    eval_run = runs[R]
    train_runs = runs[:R]
    X_ev, y_ev, sid_ev = gather_runs(subject, [eval_run], bank)
    X_all, y_all, split_ids = gather_runs(subject, train_runs, bank)
    tr_m, te_m = _trial_split(split_ids, train_frac=TRAIN_FRAC, seed=SEED)

    row: Dict[str, Any] = {
        "protocol": "leave_next",
        "arm": "A0",
        "replay": "none",
        "subject": subject,
        "R": R,
        "eval_run": eval_run,
        "train_runs": train_runs,
        "n_windows_train": int(tr_m.sum()),
        "n_windows_heldout": int(te_m.sum()),
        "n_windows_eval": int(len(X_ev)),
    }

    ft_out = finetune_a0(
        X_all[tr_m], y_all[tr_m], X_all[te_m], y_all[te_m],
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
            "epochs": epochs,
            "acc_before_heldout": ft_out["acc_before_heldout"],
            "acc_after_heldout": ft_out["acc_after_heldout"],
            "acc_after_train": ft_out["acc_after_train"],
            "train_minus_heldout": ft_out["train_minus_heldout"],
            "heldout_max_class_frac": ft_out["heldout_max_class_frac"],
            "heldout_pred_counts": ft_out["heldout_pred_counts"],
            "online_trial_acc": online["trial_metrics"]["acc_trial_majority"],
            "online_window_acc": online["acc_window"],
            "online_pred_counts": online["trial_metrics"]["pred_counts"],
            "online_max_class_frac": online["trial_metrics"]["max_class_frac"],
            "online_n_trials": online["trial_metrics"]["n_trials"],
        }
    )
    return row


def load_b2_rows() -> Dict[tuple, Dict[str, Any]]:
    if not B2_JSON.is_file():
        return {}
    data = json.loads(B2_JSON.read_text(encoding="utf-8"))
    out: Dict[tuple, Dict[str, Any]] = {}
    for row in data.get("rows", []):
        if row.get("status") == "ok" and row.get("ft"):
            key = (str(row["subject"]), int(row["R"]))
            out[key] = row
    return out


def summarize(rows: List[Dict[str, Any]], b2: Dict[tuple, Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    by_R_a0: Dict[int, List[float]] = defaultdict(list)
    by_R_b2: Dict[int, List[float]] = defaultdict(list)
    by_R_delta: Dict[int, List[float]] = defaultdict(list)
    collapse_a0 = 0
    collapse_b2 = 0
    per_subject: List[Dict[str, Any]] = []

    for r in ok:
        R = int(r["R"])
        subj = str(r["subject"])
        a0_acc = float(r["online_trial_acc"])
        by_R_a0[R].append(a0_acc)
        b2_row = b2.get((subj, R))
        b2_acc = float(b2_row["online_trial_acc"]) if b2_row else float("nan")
        if b2_row:
            by_R_b2[R].append(b2_acc)
            by_R_delta[R].append(a0_acc - b2_acc)
        if float(r.get("online_max_class_frac", 0)) >= 0.85:
            collapse_a0 += 1
        if b2_row and float(b2_row.get("online_max_class_frac", 0)) >= 0.85:
            collapse_b2 += 1
        per_subject.append(
            {
                "subject": subj,
                "R": R,
                "a0_online": a0_acc,
                "b2_online": b2_acc,
                "delta_a0_minus_b2": a0_acc - b2_acc if b2_row else None,
                "a0_heldout": float(r["acc_after_heldout"]),
                "b2_heldout": float(b2_row["acc_after_heldout"]) if b2_row else None,
                "a0_online_max_frac": float(r.get("online_max_class_frac", 0)),
                "b2_online_max_frac": float(b2_row.get("online_max_class_frac", 0)) if b2_row else None,
            }
        )

    mean_by_R = {
        str(R): {
            "mean_a0_online": float(np.mean(by_R_a0[R])),
            "mean_b2_online": float(np.mean(by_R_b2[R])) if by_R_b2[R] else float("nan"),
            "mean_delta_a0_minus_b2": float(np.mean(by_R_delta[R])) if by_R_delta[R] else float("nan"),
            "n": len(by_R_a0[R]),
        }
        for R in sorted(by_R_a0.keys())
    }
    return {
        "mean_by_R": mean_by_R,
        "online_collapse_ge85_frac_a0": collapse_a0 / max(len(ok), 1),
        "online_collapse_ge85_frac_b2": collapse_b2 / max(len(ok), 1),
        "per_subject": per_subject,
        "n_ok": len(ok),
    }


def write_md(payload: Dict[str, Any], path: Path) -> None:
    summ = payload["summary"]
    lines = [
        "# 实验 29 · 无 replay 对照（A0 vs B2）",
        "",
        f"生成：{payload['generated_at']}",
        "",
        "A0 = 全模型 FT · **无 replay** · 5 epoch（对齐 Exp27 A0）",
        "B2 = T0 replay **0.10** · 5 epoch（主实验 ramp_leave_next）",
        "",
        "## 队列均值 · 在线 trial acc",
        "",
        "| R | A0 (无 replay) | B2 (t0+0.10) | Δ A0−B2 | 解读 |",
        "|---|----------------|--------------|---------|------|",
    ]
    for R, block in sorted(summ["mean_by_R"].items(), key=lambda x: int(x[0])):
        d = block["mean_delta_a0_minus_b2"]
        verdict = "A0更差" if d < -0.02 else ("A0更好" if d > 0.02 else "相当")
        lines.append(
            f"| {R} | {block['mean_a0_online']:.3f} | {block['mean_b2_online']:.3f} | "
            f"{d:+.3f} | {verdict} |"
        )
    lines += [
        "",
        f"- 在线塌缩（max_class_frac≥0.85）：A0 **{summ['online_collapse_ge85_frac_a0']:.0%}** · "
        f"B2 **{summ['online_collapse_ge85_frac_b2']:.0%}**",
        "",
        "## 结论模板",
        "",
        "- 若 A0 在线均值系统性低于 B2，且 A0 塌缩率更高 → **T0 replay 0.10 必要**",
        "- 若 heldout A0 > B2 但在线 A0 < B2 → 再次印证 **heldout ≠ 在线**",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="E29 no-replay A0 control")
    ap.add_argument("--R-values", default=",".join(str(r) for r in DEFAULT_R))
    ap.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]
    R_list = [0, 2] if args.smoke else [int(x.strip()) for x in args.R_values.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bank = load_bci2a_bank()
    b2 = load_b2_rows()
    rows: List[Dict[str, Any]] = []

    for subject in subjects:
        runs = subject_run_list(subject, bank)
        print(f"\n=== {subject} A0 ===", flush=True)
        for R in R_list:
            row = run_a0_point(subject, R, runs, bank, device=device, epochs=args.epochs)
            rows.append(row)
            if row.get("status") == "ok":
                b2_acc = b2.get((subject, R), {}).get("online_trial_acc")
                b2_str = f"{float(b2_acc):.3f}" if b2_acc is not None else "—"
                print(
                    f"  R{R} A0 online={row['online_trial_acc']:.3f} "
                    f"B2={b2_str} heldout A0={row['acc_after_heldout']:.3f}",
                    flush=True,
                )

    payload = {
        "experiment": "E29",
        "protocol": "leave_next_noreplay",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arm": "A0",
        "replay": "none",
        "compare_arm": "B2",
        "device": device,
        "epochs": args.epochs,
        "R_values": R_list,
        "rows": rows,
        "summary": summarize(rows, b2),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_md(payload, OUT_MD)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
