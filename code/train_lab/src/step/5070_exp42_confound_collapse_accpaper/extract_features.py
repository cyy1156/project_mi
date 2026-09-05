"""?????????? / ???? / d' / probe AUC?"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from io_sessions import load_eeg, load_eeg_meta, load_events, list_v3_sessions, parse_trials
from paths import ANALYSIS, FS, N_TIMES


def _car(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def _bandpower_welch(seg_ct: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """seg_ct: (C,T) -> log-bandpower (C,)?"""
    freqs, psd = signal.welch(seg_ct, fs=FS, nperseg=min(256, seg_ct.shape[1]))
    m = (freqs >= f0) & (freqs < f1)
    bp = psd[:, m].mean(axis=1) if m.any() else np.full(seg_ct.shape[0], 1e-12)
    return np.log(np.maximum(bp, 1e-12))


def _slice_seg(t: np.ndarray, x: np.ndarray, t0: float, t1: float) -> Optional[np.ndarray]:
    i0 = int(np.searchsorted(t, t0))
    i1 = int(np.searchsorted(t, t1))
    if i1 - i0 < int(1.0 * FS):
        return None
    return x[i0:i1].T  # (C,T)


def _trial_feat(seg_ct: np.ndarray) -> np.ndarray:
    """mu/beta ??????? (16,)?"""
    mu = _bandpower_welch(seg_ct, 8.0, 13.0)
    beta = _bandpower_welch(seg_ct, 13.0, 30.0)
    return np.concatenate([mu, beta], axis=0)


def _erd_li(seg_ct: np.ndarray) -> Tuple[float, float]:
    """?? C3/C4 ? mu ?????????"""
    mu = np.exp(_bandpower_welch(seg_ct, 8.0, 13.0))
    c3, c4 = float(mu[1]), float(mu[6])
    erd = float(0.5 * (c3 + c4))
    li = float((c3 - c4) / (c3 + c4 + 1e-12))
    return erd, li


def _dprime(Xa: np.ndarray, Xb: np.ndarray) -> float:
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    ma, mb = Xa.mean(axis=0), Xb.mean(axis=0)
    sa, sb = Xa.std(axis=0), Xb.std(axis=0)
    pooled = np.sqrt(0.5 * (sa**2 + sb**2)) + 1e-12
    return float(np.mean(np.abs(ma - mb) / pooled))


def _probe_auc(X: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 8:
        return float("nan")
    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    if min(counts.values()) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    n_splits = min(5, min(counts.values()))
    if n_splits < 2:
        return float("nan")
    try:
        scores = cross_val_score(
            clf,
            X,
            y,
            cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
            scoring="roc_auc_ovr",
        )
        return float(np.nanmean(scores))
    except Exception:
        try:
            scores = cross_val_score(
                clf,
                X,
                y,
                cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
                scoring="accuracy",
            )
            return float(np.nanmean(scores))
        except Exception:
            return float("nan")


def extract_session_features(session_dir: Path, member_id: str, ws: str) -> Dict[str, Any]:
    t, x = load_eeg(session_dir)
    meta = load_eeg_meta(session_dir)
    events = load_events(session_dir)
    trials = parse_trials(events)

    abs_x = np.abs(x)
    sat_frac = (
        float((abs_x >= 0.95 * np.nanpercentile(abs_x, 99.9)).mean()) if len(x) else float("nan")
    )
    rail = 400.0
    sat_rail = float((abs_x >= 0.95 * rail).mean())
    dt = np.diff(t)
    gap_frac = float((dt > 2.0 / FS).mean()) if len(dt) else float("nan")
    empty_pulls = float(meta.get("empty_pulls") or meta.get("n_empty_pulls") or 0)
    duration_s = float(t[-1] - t[0]) if len(t) > 1 else float("nan")
    empty_per_s = empty_pulls / duration_s if duration_s and duration_s > 0 else float("nan")

    try:
        cz_cpz_r = float(np.corrcoef(x[:, 3], x[:, 4])[0, 1])
    except Exception:
        cz_cpz_r = float("nan")

    x_car = _car(x)
    mid0 = len(x) // 3
    mid1 = min(len(x), mid0 + int(30 * FS))
    if mid1 - mid0 > int(5 * FS):
        seg = x_car[mid0:mid1].T
        mu_p = float(np.mean(np.exp(_bandpower_welch(seg, 8, 13))))
        beta_p = float(np.mean(np.exp(_bandpower_welch(seg, 13, 30))))
        delta_p = float(np.mean(np.exp(_bandpower_welch(seg, 0.5, 7))))
        tot = mu_p + beta_p + delta_p + 1e-12
        mu_rel, beta_rel, delta_rel = mu_p / tot, beta_p / tot, delta_p / tot
    else:
        mu_rel = beta_rel = delta_rel = float("nan")

    feats_mi: List[np.ndarray] = []
    labs_mi: List[int] = []
    feats_all: List[np.ndarray] = []
    labs_all: List[int] = []
    rest_mu: List[float] = []
    mi_erd: List[float] = []
    mi_li: List[float] = []

    for tr in trials:
        seg = _slice_seg(t, x_car, tr["t0"], tr["t1"])
        if seg is None:
            continue
        if seg.shape[1] >= N_TIMES:
            a = (seg.shape[1] - N_TIMES) // 2
            seg_w = seg[:, a : a + N_TIMES]
        else:
            continue
        f = _trial_feat(seg_w)
        feats_all.append(f)
        labs_all.append(int(tr["label"]))
        if tr["kind"] == "rest":
            rest_mu.append(float(np.mean(np.exp(_bandpower_welch(seg_w, 8, 13)))))
        else:
            feats_mi.append(f)
            labs_mi.append(int(tr["label"]))
            erd, li = _erd_li(seg_w)
            mi_erd.append(erd)
            mi_li.append(li)

    Xmi = np.stack(feats_mi, axis=0) if feats_mi else np.zeros((0, 16))
    ymi = np.asarray(labs_mi, dtype=np.int64)
    left = Xmi[ymi == 1] if len(ymi) else np.zeros((0, 16))
    right = Xmi[ymi == 2] if len(ymi) else np.zeros((0, 16))
    Xall = np.stack(feats_all, axis=0) if feats_all else np.zeros((0, 16))
    yall = np.asarray(labs_all, dtype=np.int64)
    mi_mask = yall > 0
    rest_X = Xall[~mi_mask] if len(yall) else np.zeros((0, 16))
    mi_X = Xall[mi_mask] if len(yall) else np.zeros((0, 16))

    d_lr = _dprime(left, right)
    d_mr = _dprime(mi_X, rest_X)
    if len(ymi) and len(np.unique(ymi)) == 2:
        auc_lr = _probe_auc(Xmi, ymi)
    else:
        auc_lr = float("nan")
    y_bin = (yall > 0).astype(np.int64) if len(yall) else np.zeros(0, dtype=np.int64)
    auc_mr = _probe_auc(Xall, y_bin) if len(y_bin) else float("nan")

    slope_rest_mu = float("nan")
    slope_abs_li = float("nan")
    if len(rest_mu) >= 6:
        xs = np.arange(len(rest_mu), dtype=np.float64)
        slope_rest_mu = float(np.polyfit(xs, np.asarray(rest_mu), 1)[0])
    if len(mi_li) >= 6:
        xs = np.arange(len(mi_li), dtype=np.float64)
        slope_abs_li = float(np.polyfit(xs, np.abs(np.asarray(mi_li)), 1)[0])

    return {
        "member_id": member_id,
        "ws": ws,
        "session_dir": session_dir.name,
        "n_trials_parsed": len(trials),
        "n_mi": int((ymi > 0).sum()) if len(ymi) else 0,
        "n_rest": int(len(rest_mu)),
        "sat_frac_rail": sat_rail,
        "sat_frac_p999": sat_frac,
        "gap_frac": gap_frac,
        "empty_per_s": empty_per_s,
        "cz_cpz_r": cz_cpz_r,
        "mu_rel": mu_rel,
        "beta_rel": beta_rel,
        "delta_rel": delta_rel,
        "dprime_lr": d_lr,
        "dprime_mi_rest": d_mr,
        "probe_auc_lr": auc_lr,
        "probe_auc_mi_rest": auc_mr,
        "slope_rest_mu": slope_rest_mu,
        "slope_abs_li": slope_abs_li,
        "mean_rest_mu": float(np.mean(rest_mu)) if rest_mu else float("nan"),
        "mean_abs_li": float(np.mean(np.abs(mi_li))) if mi_li else float("nan"),
    }


def _job(args: Tuple[str, str, str]) -> Dict[str, Any]:
    member_id, ws, path_s = args
    try:
        return extract_session_features(Path(path_s), member_id, ws)
    except Exception as exc:
        return {
            "member_id": member_id,
            "ws": ws,
            "session_dir": Path(path_s).name,
            "error": str(exc),
        }


def run_extract(cohort: Dict[str, Any], *, max_workers: int = 4) -> Path:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    jobs: List[Tuple[str, str, str]] = []
    for person in cohort["people"]:
        for mid in person["member_ids"]:
            by = list_v3_sessions(mid)
            for ws, path in sorted(by.items(), key=lambda kv: kv[0]):
                jobs.append((mid, ws, str(path)))

    rows: List[Dict[str, Any]] = []
    if not jobs:
        print("[extract] no sessions")
    else:
        print(f"[extract] jobs={len(jobs)} workers={max_workers}")
        # ThreadPool??? Windows spawn + ???????????numpy/scipy ?? GIL
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for fut in as_completed(futs):
                rows.append(fut.result())

    rows.sort(key=lambda r: (r.get("member_id", ""), r.get("ws", "")))
    out = {
        "schema": "exp42_session_features_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if "error" not in r),
        "rows": rows,
    }
    path = ANALYSIS / "session_features.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[extract] wrote {path} ok={out['n_ok']}/{out['n_rows']}")
    return path


if __name__ == "__main__":
    from cohort_map import build_cohort_map

    run_extract(build_cohort_map(), max_workers=4)
