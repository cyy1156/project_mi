#!/usr/bin/env python3
"""实验 30 · BCI2a：E1f 融合 + 臂 W / S / C 聚合器对比。

W: 满 MI 多数票
S: 等 n+1 后 (n-1,n,n+1) 平滑 + τ 早停（方案 26 流式）
C: (n-2,n-1,n) 因果平滑 + 同款 τ 早停

用法::

    python -m experiment_game.tools.exp30_bci2a_aggregator_ablation
    python -m experiment_game.tools.exp30_bci2a_aggregator_ablation --smoke
    python -m experiment_game.tools.exp30_bci2a_aggregator_ablation --skip-forward
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.e1f import E1fRegistry, E1fStackConfig  # noqa: E402
from adapt_engine.readout import (  # noqa: E402
    majority_vote_from_probs,
    streaming_conf_stop_C,
    streaming_conf_stop_S,
)
from experiment_game.tools.exp29_bci2a_ramp_grid import (  # noqa: E402
    ALL_SUBJECTS,
    load_bci2a_bank,
    subject_run_list,
)

OUT_DIR = _REPO / "experiment_game" / "data" / "models" / "bci2a" / "exp30"
DEFAULT_E1F = _REPO / "experiment_game" / "config" / "e1f_four_member.json"
TAU_DEFAULT = 0.4
TAU_GRID_DEFAULT = (0.4, 0.45, 0.5, 0.55, 0.6)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _trial_key(subject_tag: str, trial_id: Any) -> str:
    # subject_tag like A01|run3|S1
    parts = str(subject_tag).split("|")
    subj = parts[0] if parts else "?"
    run = parts[1] if len(parts) > 1 else "?"
    return f"{subj}|{run}|{int(trial_id)}"


def _parse_run(key: str) -> str:
    return str(key).split("|")[1]


def _parse_subj(key: str) -> str:
    return str(key).split("|")[0]


def build_or_load_probs(
    *,
    e1f_config: Path,
    cache_dir: Path,
    batch_size: int,
    skip_forward: bool,
    smoke_n: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """返回 probs(N,3), y, subjects, trial_id, meta。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    probs_path = cache_dir / "e1f_fused_p_three.npy"
    meta_path = cache_dir / "cache_meta.json"
    bank = load_bci2a_bank()
    y = np.asarray(bank["y"])
    subjects = np.asarray(bank["subjects"], dtype=object)
    trial_id = np.asarray(bank["trial_id"])
    n_all = len(y)
    use_n = smoke_n if smoke_n > 0 else n_all

    if skip_forward and probs_path.is_file():
        probs = np.load(probs_path)
        if smoke_n > 0:
            probs = probs[:smoke_n]
            y = y[:smoke_n]
            subjects = subjects[:smoke_n]
            trial_id = trial_id[:smoke_n]
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        return probs, y, subjects, trial_id, meta

    stack = E1fStackConfig.load_json(e1f_config, repo_root=_REPO).resolve_paths(
        repo_root=_REPO
    )
    missing = stack.missing_paths(repo_root=_REPO)
    if missing:
        raise FileNotFoundError("\n".join(missing))

    device = _device()
    print(f"[exp30] loading E1f on {device} …")
    reg = E1fRegistry(stack, device=device)
    Xmm = bank["X"]
    windows = np.stack(
        [Xmm[i, 0].astype(np.float32) for i in range(use_n)], axis=0
    )
    print(f"[exp30] forward {use_n} windows (batch={batch_size}) …")
    probs = reg.forward_three_batch(windows, batch_size=batch_size)
    if smoke_n <= 0:
        np.save(probs_path, probs)
        meta = {
            "n": int(use_n),
            "e1f_config": str(e1f_config),
            "created": datetime.now().isoformat(timespec="seconds"),
            "device": device,
            "weights": list(stack.fusion.weights),
            "temperatures": list(stack.fusion.temperatures),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        y = y[:use_n]
        subjects = subjects[:use_n]
        trial_id = trial_id[:use_n]
        meta = {"n": int(use_n), "smoke": True}
    return probs, y, subjects, trial_id, meta


def group_trials(
    probs: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    trial_id: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    labels: Dict[str, int] = {}
    for i in range(len(y)):
        key = _trial_key(subjects[i], trial_id[i])
        buckets[key].append(i)
        if key not in labels:
            labels[key] = int(y[i])
    trials: Dict[str, Dict[str, Any]] = {}
    for key, idxs in buckets.items():
        idxs = sorted(idxs)
        p = probs[idxs]
        k = len(idxs)
        t_rels = [3.0 + 0.1 * j for j in range(k)]
        trials[key] = {
            "key": key,
            "subject": _parse_subj(key),
            "run": _parse_run(key),
            "label": labels[key],
            "probs": p,
            "t_rels": t_rels,
        }
    return trials


def decide_trial(trial: Dict[str, Any], arm: str, tau: float) -> Dict[str, Any]:
    probs = trial["probs"]
    t_rels = trial["t_rels"]
    if arm == "W":
        return majority_vote_from_probs(probs, t_rels=t_rels)
    if arm == "S":
        return streaming_conf_stop_S(probs, t_rels=t_rels, tau_conf=tau)
    if arm == "C":
        return streaming_conf_stop_C(probs, t_rels=t_rels, tau_conf=tau, min_windows=3)
    raise ValueError(arm)


def eval_trials(
    trials: Sequence[Dict[str, Any]],
    *,
    arms: Sequence[str],
    tau: float,
    labels: Sequence[int] = (1, 2),
) -> Dict[str, Any]:
    selected = [t for t in trials if int(t["label"]) in labels]
    out: Dict[str, Any] = {"n": len(selected), "arms": {}}
    for arm in arms:
        preds = []
        t_decs = []
        early = []
        for t in selected:
            d = decide_trial(t, arm, tau)
            preds.append(int(d["pred"]))
            t_decs.append(float(d["t_dec"]))
            early.append(bool(d.get("early")))
        y = [int(t["label"]) for t in selected]
        ok = sum(1 for p, yy in zip(preds, y) if p == yy)
        n = len(y)
        out["arms"][arm] = {
            "n": n,
            "n_correct": ok,
            "acc": (ok / n) if n else None,
            "t_dec_mean": float(np.mean(t_decs)) if t_decs else None,
            "early_frac": float(np.mean(early)) if early else None,
            "pred_hist": {
                int(k): int(v)
                for k, v in zip(*np.unique(preds, return_counts=True))
            }
            if preds
            else {},
        }
    if "S" in arms and "C" in arms and selected:
        out["delta_C_minus_S_acc"] = (
            out["arms"]["C"]["acc"] - out["arms"]["S"]["acc"]
            if out["arms"]["C"]["acc"] is not None and out["arms"]["S"]["acc"] is not None
            else None
        )
        out["delta_C_minus_S_t_dec"] = (
            out["arms"]["C"]["t_dec_mean"] - out["arms"]["S"]["t_dec_mean"]
            if out["arms"]["C"]["t_dec_mean"] is not None
            and out["arms"]["S"]["t_dec_mean"] is not None
            else None
        )
    return out


def select_shared_tau(
    val_trials: Sequence[Dict[str, Any]],
    *,
    tau_grid: Sequence[float],
) -> Tuple[float, Dict[str, float]]:
    """共享 τ*：在 val 上最大化 mean(Acc_S, Acc_C)。"""
    best_tau = float(tau_grid[0])
    best_score = -1.0
    scores: Dict[str, float] = {}
    for tau in tau_grid:
        ev = eval_trials(val_trials, arms=("S", "C"), tau=float(tau))
        a_s = ev["arms"]["S"]["acc"] or 0.0
        a_c = ev["arms"]["C"]["acc"] or 0.0
        score = 0.5 * (a_s + a_c)
        scores[f"{tau:.2f}"] = float(score)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau, scores


def run_protocol(
    trials_by_key: Dict[str, Dict[str, Any]],
    *,
    subjects: Sequence[str],
    bank_runs: Dict[str, List[str]],
    tau_fixed: float,
    tau_grid: Sequence[float],
    arms: Sequence[str],
) -> Dict[str, Any]:
    per_subj: Dict[str, Any] = {}
    pooled_all: List[Dict[str, Any]] = []
    pooled_test: List[Dict[str, Any]] = []
    test_rows_fixed: List[Dict[str, Any]] = []
    test_rows_star: List[Dict[str, Any]] = []

    for sid in subjects:
        runs = bank_runs.get(sid) or []
        if len(runs) < 2:
            continue
        val_runs = set(runs[:2])
        test_runs = set(runs[2:]) if len(runs) > 2 else set(runs[-1:])
        sub_trials = [t for t in trials_by_key.values() if t["subject"] == sid]
        val_t = [t for t in sub_trials if t["run"] in val_runs]
        test_t = [t for t in sub_trials if t["run"] in test_runs]
        pooled_all.extend(sub_trials)
        pooled_test.extend(test_t)

        tau_star, val_scores = select_shared_tau(val_t, tau_grid=tau_grid)
        fixed = eval_trials(sub_trials, arms=arms, tau=tau_fixed)
        test_fixed = eval_trials(test_t, arms=arms, tau=tau_fixed)
        test_star = eval_trials(test_t, arms=arms, tau=tau_star)
        all_star = eval_trials(sub_trials, arms=arms, tau=tau_star)
        per_subj[sid] = {
            "runs": runs,
            "val_runs": sorted(val_runs, key=lambda r: int(r.replace("run", "") or 0)),
            "test_runs": sorted(test_runs, key=lambda r: int(r.replace("run", "") or 0)),
            "tau_star": tau_star,
            "val_tau_scores": val_scores,
            "all_tau_fixed": fixed,
            "test_tau_fixed": test_fixed,
            "test_tau_star": test_star,
            "all_tau_star": all_star,
        }
        test_rows_fixed.append(test_fixed)
        test_rows_star.append(test_star)

    def _queue_mean(rows: List[Dict[str, Any]], arm: str, field: str) -> Optional[float]:
        vals = []
        for r in rows:
            v = (r.get("arms") or {}).get(arm, {}).get(field)
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    queue_fixed = {
        arm: {
            "acc_mean": _queue_mean(test_rows_fixed, arm, "acc"),
            "t_dec_mean": _queue_mean(test_rows_fixed, arm, "t_dec_mean"),
            "early_frac_mean": _queue_mean(test_rows_fixed, arm, "early_frac"),
        }
        for arm in arms
    }
    queue_star = {
        arm: {
            "acc_mean": _queue_mean(test_rows_star, arm, "acc"),
            "t_dec_mean": _queue_mean(test_rows_star, arm, "t_dec_mean"),
            "early_frac_mean": _queue_mean(test_rows_star, arm, "early_frac"),
        }
        for arm in arms
    }

    return {
        "per_subject": per_subj,
        "pooled_all_tau_fixed": eval_trials(pooled_all, arms=arms, tau=tau_fixed),
        "pooled_test_tau_fixed": eval_trials(pooled_test, arms=arms, tau=tau_fixed),
        "queue_test_tau_fixed": queue_fixed,
        "queue_test_tau_star": queue_star,
        "tau_fixed": tau_fixed,
        "tau_grid": list(tau_grid),
    }


def write_report(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 实验 30 · BCI2a 聚合器 W / S / C",
        "",
        f"生成：{summary.get('created')}",
        f"E1f：`{summary.get('e1f_config')}`",
        f"τ_fixed={summary.get('results', {}).get('tau_fixed')}",
        "",
        "## 队列 test（每人前 2 run val / 后 4 run test）· τ=0.4",
        "",
        "| 臂 | Acc_LR mean | t_dec mean | early mean |",
        "|----|-------------|------------|------------|",
    ]
    qf = summary["results"]["queue_test_tau_fixed"]
    for arm in ("W", "S", "C"):
        a = qf.get(arm) or {}
        lines.append(
            f"| {arm} | {_pct(a.get('acc_mean'))} | {_fmt(a.get('t_dec_mean'))} | {_pct(a.get('early_frac_mean'))} |"
        )
    lines += [
        "",
        "## 队列 test · 共享 τ*（val 上 max mean(Acc_S,Acc_C)）",
        "",
        "| 臂 | Acc_LR mean | t_dec mean | early mean |",
        "|----|-------------|------------|------------|",
    ]
    qs = summary["results"]["queue_test_tau_star"]
    for arm in ("W", "S", "C"):
        a = qs.get(arm) or {}
        lines.append(
            f"| {arm} | {_pct(a.get('acc_mean'))} | {_fmt(a.get('t_dec_mean'))} | {_pct(a.get('early_frac_mean'))} |"
        )
    lines += ["", "## 分被试 test · τ=0.4", ""]
    for sid, blob in summary["results"]["per_subject"].items():
        tf = blob["test_tau_fixed"]["arms"]
        lines.append(
            f"- **{sid}** τ*={blob['tau_star']:.2f} | "
            f"W={_pct(tf['W']['acc'])} S={_pct(tf['S']['acc'])} "
            f"C={_pct(tf['C']['acc'])} | "
            f"tS={_fmt(tf['S']['t_dec_mean'])} tC={_fmt(tf['C']['t_dec_mean'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(x: Optional[float]) -> str:
    return f"{100 * x:.1f}%" if x is not None else "n/a"


def _fmt(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1f-config", type=Path, default=DEFAULT_E1F)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tau", type=float, default=TAU_DEFAULT)
    ap.add_argument(
        "--tau-grid",
        default=",".join(str(t) for t in TAU_GRID_DEFAULT),
    )
    ap.add_argument("--arms", default="W,S,C")
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="仅前 500 窗冒烟")
    ap.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    args = ap.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    tau_grid = tuple(float(x) for x in args.tau_grid.split(",") if x.strip())
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    out_dir = args.out_dir
    cache_dir = out_dir / "probs_cache"
    smoke_n = 500 if args.smoke else 0

    probs, y, subjects_arr, trial_id, meta = build_or_load_probs(
        e1f_config=args.e1f_config,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        skip_forward=args.skip_forward,
        smoke_n=smoke_n,
    )
    print(f"[exp30] probs={probs.shape} caching meta keys={list(meta)}")

    trials = group_trials(probs, y, subjects_arr, trial_id)
    bank = load_bci2a_bank()
    bank_runs = {sid: subject_run_list(sid, bank) for sid in subjects}
    if args.smoke:
        # 冒烟：不做 run 划分，只报池化
        results = {
            "tau_fixed": args.tau,
            "tau_grid": list(tau_grid),
            "per_subject": {},
            "pooled_all_tau_fixed": eval_trials(
                list(trials.values()), arms=arms, tau=args.tau
            ),
            "pooled_test_tau_fixed": None,
            "queue_test_tau_fixed": {},
            "queue_test_tau_star": {},
            "smoke": True,
        }
    else:
        results = run_protocol(
            trials,
            subjects=subjects,
            bank_runs=bank_runs,
            tau_fixed=args.tau,
            tau_grid=tau_grid,
            arms=arms,
        )

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "e1f_config": str(args.e1f_config),
        "n_windows": int(probs.shape[0]),
        "n_trials": len(trials),
        "arms": list(arms),
        "cache_meta": meta,
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / ("results_e1f_smoke.json" if args.smoke else "results_e1f.json")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[exp30] wrote {json_path}")

    if not args.smoke:
        report_path = out_dir / "report.md"
        write_report(summary, report_path)
        print(f"[exp30] wrote {report_path}")
        qf = results["queue_test_tau_fixed"]
        print("\n=== queue test τ=0.4 ===")
        for arm in arms:
            a = qf.get(arm) or {}
            print(
                f"  {arm}: acc={_pct(a.get('acc_mean'))} "
                f"t_dec={_fmt(a.get('t_dec_mean'))} early={_pct(a.get('early_frac_mean'))}"
            )
        qs = results["queue_test_tau_star"]
        print("=== queue test τ* ===")
        for arm in arms:
            a = qs.get(arm) or {}
            print(
                f"  {arm}: acc={_pct(a.get('acc_mean'))} "
                f"t_dec={_fmt(a.get('t_dec_mean'))} early={_pct(a.get('early_frac_mean'))}"
            )
    else:
        pa = results["pooled_all_tau_fixed"]["arms"]
        print("\n=== smoke pooled ===")
        for arm in arms:
            a = pa[arm]
            print(
                f"  {arm}: acc={_pct(a['acc'])} t_dec={_fmt(a['t_dec_mean'])} "
                f"early={_pct(a['early_frac'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
