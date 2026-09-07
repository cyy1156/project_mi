#!/usr/bin/env python3
"""因果 vs 事后 E1f / 多数票：用已落盘的窗级融合 p_three 回放试次判定。

不重跑模型。输入：subjects/*/sessions/*/v3_trial_features.jsonl

用法::

    python -m experiment_game.tools.replay_causal_e1f_judge \\
        --subjects syj0828,xjh0828
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.readout import e1f_conf_stop_from_judgments  # noqa: E402
from experiment_game.experiment.judge_aggregate import (  # noqa: E402
    primary_judge_from_judgments,
)

SUBJECTS_ROOT = _REPO / "experiment_game" / "data" / "subjects"
LABEL_NAMES = {0: "Rest", 1: "Left", 2: "Right"}

def _mi_judgments(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    js = [j for j in (row.get("judgments") or []) if not j.get("signal_bad")]
    js = [j for j in js if j.get("p_three")]
    return sorted(js, key=lambda j: float(j.get("t_rel", 0.0)))

def causal_conf_stop(
    judgments: Sequence[Dict[str, Any]],
    *,
    lookback: int = 2,
    tau_conf: float = 0.4,
) -> Optional[Dict[str, Any]]:
    """在线因果：仅用当前窗及前 lookback 窗平滑；p_max≥τ 即提交。"""
    js = [j for j in judgments if not j.get("signal_bad") and j.get("p_three")]
    if not js:
        return None
    js = sorted(js, key=lambda j: float(j.get("t_rel", 0.0)))
    raw = [np.asarray(j["p_three"], dtype=np.float32) for j in js]
    picked = len(js) - 1
    smoothed_picked: Optional[np.ndarray] = None
    for i in range(len(js)):
        lo = max(0, i - int(lookback))
        p = np.mean(np.stack(raw[lo : i + 1], axis=0), axis=0).astype(np.float32)
        if float(np.max(p)) >= float(tau_conf):
            picked = i
            smoothed_picked = p
            break
    if smoothed_picked is None:
        lo = max(0, picked - int(lookback))
        smoothed_picked = np.mean(np.stack(raw[lo : picked + 1], axis=0), axis=0).astype(
            np.float32
        )
    pred = int(np.argmax(smoothed_picked))
    rep = dict(js[picked])
    rep["pred"] = pred
    rep["p_max"] = float(np.max(smoothed_picked))
    rep["p_three"] = [float(x) for x in smoothed_picked.ravel()]
    rep["rule"] = f"e1f_causal_lb{lookback}"
    rep["e1f_picked_t_rel"] = float(js[picked].get("t_rel", 0.0))
    rep["e1f_tau_conf"] = float(tau_conf)
    rep["e1f_lookback"] = int(lookback)
    return rep

def last_window_pred(judgments: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    js = _mi_judgments({"judgments": list(judgments)})
    if not js:
        return None
    j = js[-1]
    p = np.asarray(j["p_three"], dtype=np.float32)
    rep = dict(j)
    rep["pred"] = int(np.argmax(p))
    rep["p_max"] = float(np.max(p))
    rep["rule"] = "last_window"
    rep["e1f_picked_t_rel"] = float(j.get("t_rel", 0.0))
    return rep

def decide(
    judgments: List[Dict[str, Any]],
    *,
    mode: str,
    tau: float,
) -> Optional[Dict[str, Any]]:
    if mode == "majority":
        return primary_judge_from_judgments(judgments, mode="majority")
    if mode == "e1f_bidir":
        return e1f_conf_stop_from_judgments(
            judgments, smooth_radius=1, tau_conf=tau, primary_s=4.0
        )
    if mode == "causal_lb1":
        return causal_conf_stop(judgments, lookback=1, tau_conf=tau)
    if mode == "causal_lb2":
        return causal_conf_stop(judgments, lookback=2, tau_conf=tau)
    if mode == "last_window":
        return last_window_pred(judgments)
    raise ValueError(mode)

MODES = ("majority", "e1f_bidir", "causal_lb1", "causal_lb2", "last_window")

def iter_sessions(subject_ids: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for sid in subject_ids:
        root = SUBJECTS_ROOT / sid / "sessions"
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            f = d / "v3_trial_features.jsonl"
            if f.is_file():
                out.append((sid, f))
    return out

def load_trials(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def eval_session(
    rows: List[Dict[str, Any]],
    *,
    tau: float,
    labels: Sequence[int] = (1, 2),
) -> Dict[str, Any]:
    per_mode: Dict[str, Dict[str, Any]] = {}
    preds_by_mode: Dict[str, List[int]] = {m: [] for m in MODES}
    labs: List[int] = []
    decision_t: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        lab = int(row["label"])
        if lab not in labels:
            continue
        js = _mi_judgments(row)
        if not js:
            continue
        labs.append(lab)
        for mode in MODES:
            rep = decide(js, mode=mode, tau=tau)
            pred = int(rep["pred"]) if rep else -1
            preds_by_mode[mode].append(pred)
            if rep and rep.get("e1f_picked_t_rel") is not None:
                decision_t[mode].append(float(rep["e1f_picked_t_rel"]))
            elif rep and mode == "majority":
                decision_t[mode].append(4.0)  # 事后满 MI

    n = len(labs)
    for mode in MODES:
        preds = preds_by_mode[mode]
        ok = sum(1 for p, y in zip(preds, labs) if p == y)
        cm = Counter()
        for p, y in zip(preds, labs):
            cm[(y, p)] += 1
        per_mode[mode] = {
            "n": n,
            "acc": (ok / n) if n else None,
            "n_correct": ok,
            "pred_hist": dict(Counter(preds)),
            "mean_decision_t_rel": (
                float(np.mean(decision_t[mode])) if decision_t[mode] else None
            ),
            "early_frac_lt4": (
                float(np.mean([t < 3.999 for t in decision_t[mode]]))
                if decision_t[mode] and mode != "majority"
                else None
            ),
        }

    # pairwise disagree
    disagree = {}
    for a in MODES:
        for b in MODES:
            if a >= b:
                continue
            pa, pb = preds_by_mode[a], preds_by_mode[b]
            if not pa:
                continue
            disagree[f"{a}_vs_{b}"] = float(
                np.mean([x != y for x, y in zip(pa, pb)])
            )

    # when disagree with majority, who is right
    maj = preds_by_mode["majority"]
    for mode in ("e1f_bidir", "causal_lb1", "causal_lb2"):
        other = preds_by_mode[mode]
        both_wrong = maj_ok = oth_ok = 0
        n_dis = 0
        for m, o, y in zip(maj, other, labs):
            if m == o:
                continue
            n_dis += 1
            if m == y and o != y:
                maj_ok += 1
            elif o == y and m != y:
                oth_ok += 1
            else:
                both_wrong += 1
        per_mode[mode]["vs_majority_disagree"] = {
            "n": n_dis,
            "majority_correct": maj_ok,
            "other_correct": oth_ok,
            "both_wrong": both_wrong,
        }

    return {"n_lr": n, "modes": per_mode, "disagree": disagree}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="syj0828,xjh0828")
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument(
        "--tau-grid",
        default="",
        help="可选逗号分隔 τ 网格，仅扫 causal_lb2（附报）",
    )
    ap.add_argument(
        "--out",
        default="",
        help="写 JSON 汇总路径（默认打印到 stdout 旁写 tools 下）",
    )
    args = ap.parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    sessions = iter_sessions(subjects)
    if not sessions:
        print("no sessions found", file=sys.stderr)
        return 1

    summary: Dict[str, Any] = {
        "tau": args.tau,
        "modes": list(MODES),
        "sessions": [],
        "by_subject": {},
        "pooled": None,
    }

    pooled_rows: List[Dict[str, Any]] = []
    by_subj_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    print(f"tau={args.tau}  sessions={len(sessions)}")
    print(
        f"{'session':<42} {'n':>4} "
        + " ".join(f"{m:>12}" for m in MODES)
        + f" {'c_lb2_t':>8}"
    )

    for sid, path in sessions:
        rows = load_trials(path)
        pooled_rows.extend(rows)
        by_subj_rows[sid].extend(rows)
        ev = eval_session(rows, tau=args.tau)
        name = path.parent.name
        accs = []
        for m in MODES:
            a = ev["modes"][m]["acc"]
            accs.append(f"{a:.1%}" if a is not None else "  n/a")
        t_c = ev["modes"]["causal_lb2"]["mean_decision_t_rel"]
        t_s = f"{t_c:>8.2f}" if t_c is not None else f"{'n/a':>8}"
        print(
            f"{name:<42} {ev['n_lr']:>4} "
            + " ".join(f"{x:>12}" for x in accs)
            + f" {t_s}"
        )
        summary["sessions"].append({"subject": sid, "session": name, **ev})

    print("\n--- by subject (L/R) ---")
    for sid, rows in by_subj_rows.items():
        ev = eval_session(rows, tau=args.tau)
        summary["by_subject"][sid] = ev
        line = f"{sid}: n={ev['n_lr']}"
        for m in MODES:
            a = ev["modes"][m]["acc"]
            t = ev["modes"][m]["mean_decision_t_rel"]
            early = ev["modes"][m]["early_frac_lt4"]
            extra = ""
            if t is not None and m.startswith("causal"):
                extra = f" t_mean={t:.2f} early={early:.0%}"
            elif t is not None and m == "e1f_bidir":
                extra = f" t_mean={t:.2f} early={early:.0%}"
            line += f" | {m}={a:.1%}{extra}" if a is not None else f" | {m}=n/a"
        print(line)
        for m in ("e1f_bidir", "causal_lb1", "causal_lb2"):
            d = ev["modes"][m].get("vs_majority_disagree") or {}
            if d.get("n"):
                print(
                    f"  vs maj · {m}: disagree={d['n']} "
                    f"maj_ok={d['majority_correct']} "
                    f"{m}_ok={d['other_correct']} both_wrong={d['both_wrong']}"
                )

    pooled = eval_session(pooled_rows, tau=args.tau)
    summary["pooled"] = pooled
    print("\n--- pooled ---")
    print(f"n_lr={pooled['n_lr']}")
    for m in MODES:
        md = pooled["modes"][m]
        print(
            f"  {m:14s} acc={md['acc']:.1%} "
            f"({md['n_correct']}/{md['n']}) "
            f"pred={md['pred_hist']} "
            f"t_mean={md['mean_decision_t_rel']} early={md['early_frac_lt4']}"
        )
    print("disagree:", json.dumps(pooled["disagree"], indent=2))

    if args.tau_grid.strip():
        print("\n--- causal_lb2 tau grid (pooled) ---")
        grid = [float(x) for x in args.tau_grid.split(",") if x.strip()]
        for tau in grid:
            ev = eval_session(pooled_rows, tau=tau)
            md = ev["modes"]["causal_lb2"]
            print(
                f"  tau={tau:.2f} acc={md['acc']:.1%} "
                f"t_mean={md['mean_decision_t_rel']:.2f} early={md['early_frac_lt4']:.0%}"
            )

    out = Path(args.out) if args.out else (
        _REPO / "experiment_game" / "data" / "subjects" / "_replay_causal_e1f.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
