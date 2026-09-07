"""A/B/C/D/E 五臂的 Day0 观测统计；A 为 Leave-Next 观测代理（完整 jackknife 重训另见 arm_A_full.py）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from paths import ANALYSIS, OUT_ROOT


def _load(name: str) -> Dict[str, Any]:
    return json.loads((ANALYSIS / name).read_text(encoding="utf-8"))


def _finite(xs: List[float]) -> np.ndarray:
    a = np.asarray(xs, dtype=np.float64)
    return a[np.isfinite(a)]


def arm_A_observational(ln: Dict[str, Any]) -> Dict[str, Any]:
    """基于 Leave-Next 解析结果的观测代理：

    - R_vol_proxy ~= last_mi - first_mi
    - R_recent_proxy ~= last_mi - mid_mi
    """
    deltas = []
    mid_deltas = []
    same_sign_vol = 0
    n = 0
    per = []
    for row in ln["rows"]:
        p = row["primary"]
        if p.get("error") or not p.get("stages"):
            continue
        stages = p["stages"]
        first, last = stages[0]["mi"], stages[-1]["mi"]
        mid = stages[len(stages) // 2]["mi"]
        d = float(last - first)
        dm = float(last - mid)
        deltas.append(d)
        mid_deltas.append(dm)
        n += 1
        if d > 0:
            same_sign_vol += 1
        per.append(
            {
                "person_id": row["person_id"],
                "primary_id": row["primary_id"],
                "n_rounds": len(stages),
                "first_mi": first,
                "mid_mi": mid,
                "last_mi": last,
                "R_vol_proxy": d,
                "R_recent_proxy": dm,
                "last_pass": p.get("last_pass"),
                "collapse_any": p.get("collapse_any"),
            }
        )
    arr = _finite(deltas)
    arr_m = _finite(mid_deltas)
    mean_vol = float(np.mean(arr)) if len(arr) else float("nan")
    frac_pos = same_sign_vol / n if n else float("nan")
    if mean_vol >= 0.03 and frac_pos >= 2 / 3:
        verdict = "volume_dominant_proxy"
    elif mean_vol < 0.01:
        verdict = "recent_state_dominant_proxy"
    else:
        verdict = "mixed_or_inconclusive_proxy"
    return {
        "arm": "A_observational",
        "note": "观测代理，非 A1−A2 头对头；同批 heldout 上的完整重训消融见 arm_A_full",
        "n_people": n,
        "R_vol_proxy_mean": mean_vol,
        "R_vol_proxy_median": float(np.median(arr)) if len(arr) else float("nan"),
        "frac_positive": frac_pos,
        "R_recent_proxy_mean": float(np.mean(arr_m)) if len(arr_m) else float("nan"),
        "verdict": verdict,
        "per_person": per,
    }


def arm_B_stats(feat: Dict[str, Any], ln: Dict[str, Any]) -> Dict[str, Any]:
    by_member: Dict[str, List[Dict[str, Any]]] = {}
    for row in feat["rows"]:
        if "error" in row:
            continue
        by_member.setdefault(row["member_id"], []).append(row)

    slope_rest_pos = 0
    slope_li_neg = 0
    n_slope = 0
    gains: List[float] = []
    sats: List[float] = []
    gaps: List[float] = []
    dprimes: List[float] = []
    day_decay: List[float] = []
    per = []

    for person in ln["rows"]:
        mid = person["primary_id"]
        sess = by_member.get(mid) or []
        prim = person["primary"]
        if prim.get("error"):
            continue
        s_rest = [s["slope_rest_mu"] for s in sess if np.isfinite(s.get("slope_rest_mu", np.nan))]
        s_li = [s["slope_abs_li"] for s in sess if np.isfinite(s.get("slope_abs_li", np.nan))]
        med_rest = float(np.median(s_rest)) if s_rest else float("nan")
        med_li = float(np.median(s_li)) if s_li else float("nan")
        if s_rest:
            n_slope += 1
            if med_rest > 0:
                slope_rest_pos += 1
            if s_li and med_li < 0:
                slope_li_neg += 1
        gain = float(prim.get("delta_mi") or np.nan)
        sat = float(np.nanmean([s["sat_frac_rail"] for s in sess])) if sess else float("nan")
        gap = float(np.nanmean([s["gap_frac"] for s in sess])) if sess else float("nan")
        dp = float(np.nanmedian([s["dprime_lr"] for s in sess])) if sess else float("nan")
        gains.append(gain)
        sats.append(sat)
        gaps.append(gap)
        dprimes.append(dp)
        sess_sorted = sorted(sess, key=lambda s: s.get("ws") or "")
        if len(sess_sorted) >= 2:
            for a, b in zip(sess_sorted, sess_sorted[1:]):
                if np.isfinite(a.get("dprime_lr", np.nan)) and np.isfinite(b.get("dprime_lr", np.nan)):
                    day_decay.append(float(b["dprime_lr"] - a["dprime_lr"]))
        per.append(
            {
                "person_id": person["person_id"],
                "primary_id": mid,
                "delta_mi": gain,
                "med_slope_rest_mu": med_rest,
                "med_slope_abs_li": med_li,
                "mean_sat_rail": sat,
                "mean_gap_frac": gap,
                "med_dprime_lr": dp,
            }
        )

    def _sp(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            return {"r": float("nan"), "p": float("nan"), "n": int(m.sum())}
        r, p = spearmanr(x[m], y[m])
        return {"r": float(r), "p": float(p), "n": int(m.sum())}

    dd = _finite(day_decay)
    if len(dd) >= 6:
        try:
            wstat = wilcoxon(dd)
            wil = {
                "stat": float(wstat.statistic),
                "p": float(wstat.pvalue),
                "n": len(dd),
                "median": float(np.median(dd)),
            }
        except Exception:
            wil = {"n": len(dd), "median": float(np.median(dd)), "p": float("nan")}
    else:
        wil = {
            "n": len(dd),
            "median": float(np.median(dd)) if len(dd) else float("nan"),
            "p": float("nan"),
        }

    fatigue_ok = (slope_rest_pos / n_slope >= 2 / 3) if n_slope else False
    return {
        "arm": "B",
        "n_people_slope": n_slope,
        "frac_rest_mu_slope_pos": slope_rest_pos / n_slope if n_slope else float("nan"),
        "frac_abs_li_slope_neg": slope_li_neg / n_slope if n_slope else float("nan"),
        "fatigue_signal": bool(fatigue_ok),
        "fatigue_verdict": (
            "session_fatigue_signal_present"
            if fatigue_ok
            else "no_clear_within_session_fatigue_under_this_paradigm"
        ),
        "spearman_gain_vs_sat": _sp(gains, sats),
        "spearman_gain_vs_gap": _sp(gains, gaps),
        "spearman_gain_vs_dprime": _sp(gains, dprimes),
        "adjacent_dprime_delta": wil,
        "per_person": per,
    }


def arm_C_homogeneity(feat: Dict[str, Any], ln: Dict[str, Any]) -> Dict[str, Any]:
    by_member: Dict[str, List[Dict[str, Any]]] = {}
    for row in feat["rows"]:
        if "error" in row:
            continue
        by_member.setdefault(row["member_id"], []).append(row)

    weak = mid = strong = 0
    n_sess = 0
    transfer_gaps: List[float] = []
    per = []
    for person in ln["rows"]:
        mid_id = person["primary_id"]
        sess = by_member.get(mid_id) or []
        dps = [s["dprime_lr"] for s in sess if np.isfinite(s.get("dprime_lr", np.nan))]
        aucs = [s["probe_auc_lr"] for s in sess if np.isfinite(s.get("probe_auc_lr", np.nan))]
        for dp in dps:
            n_sess += 1
            if dp >= 1.0:
                strong += 1
            elif dp >= 0.5:
                mid += 1
            else:
                weak += 1
        cv = float(np.std(dps) / (np.mean(dps) + 1e-12)) if len(dps) >= 2 else float("nan")
        if len(dps) >= 2:
            gaps = [abs(dps[i + 1] - dps[i]) for i in range(len(dps) - 1)]
            transfer_gaps.append(float(np.median(gaps)))
        per.append(
            {
                "person_id": person["person_id"],
                "primary_id": mid_id,
                "n_sess": len(sess),
                "med_dprime_lr": float(np.median(dps)) if dps else float("nan"),
                "med_probe_auc_lr": float(np.median(aucs)) if aucs else float("nan"),
                "dprime_cv": cv,
                "frac_weak": float(np.mean([1 if d < 0.5 else 0 for d in dps])) if dps else float("nan"),
            }
        )

    return {
        "arm": "C",
        "n_sessions": n_sess,
        "dprime_bins": {
            "strong_ge_1.0": strong,
            "mid_0.5_1.0": mid,
            "weak_lt_0.5": weak,
            "weak_frac": weak / n_sess if n_sess else float("nan"),
        },
        "median_adjacent_dprime_gap": float(np.median(transfer_gaps)) if transfer_gaps else float("nan"),
        "note_friedman": "个体内会话同质性 Friedman / KW 检验待补（当前 d' 为留一 CV 估计）；该行将在 Day0.5 补齐",
        "per_person": per,
    }


def arm_D_collapse(ln: Dict[str, Any], feat: Dict[str, Any]) -> Dict[str, Any]:
    by_member: Dict[str, List[Dict[str, Any]]] = {}
    for row in feat["rows"]:
        if "error" in row:
            continue
        by_member.setdefault(row["member_id"], []).append(row)

    counts = {"signal": 0, "optim": 0, "readout": 0, "none": 0, "mixed": 0}
    per = []
    for person in ln["rows"]:
        mid = person["primary_id"]
        prim = person["primary"]
        sess = by_member.get(mid) or []
        med_dp = float(np.nanmedian([s["dprime_lr"] for s in sess])) if sess else float("nan")
        med_auc = float(np.nanmedian([s["probe_auc_lr"] for s in sess])) if sess else float("nan")
        peak_mcf = float(prim.get("max_class_frac_peak") or np.nan)
        collapse = bool(prim.get("collapse_any"))
        tags = []
        if np.isfinite(med_dp) and med_dp < 0.5 and (not np.isfinite(med_auc) or med_auc < 0.65):
            tags.append("signal")
        if collapse or (np.isfinite(peak_mcf) and peak_mcf >= 0.85):
            tags.append("optim")
        last_mi = float(prim.get("last_mi") or np.nan)
        last_win = float("nan")
        if prim.get("stages"):
            last_win = float(prim["stages"][-1].get("win_smooth") or np.nan)
        if np.isfinite(last_mi) and last_mi < 0.4 and np.isfinite(last_win) and last_win > 0.45:
            tags.append("readout")
        if not tags:
            lab = "none"
            counts["none"] += 1
        elif len(tags) == 1:
            lab = tags[0]
            counts[lab] += 1
        else:
            lab = "mixed:" + "+".join(tags)
            counts["mixed"] += 1
        per.append(
            {
                "person_id": person["person_id"],
                "primary_id": mid,
                "med_dprime_lr": med_dp,
                "med_probe_auc_lr": med_auc,
                "max_class_frac_peak": peak_mcf,
                "collapse_any": collapse,
                "last_mi": last_mi,
                "label": lab,
            }
        )
    n = len(per) or 1
    return {
        "arm": "D",
        "counts": counts,
        "frac": {k: counts[k] / n for k in counts},
        "hypothesis_check": {
            "H1_signal_majority_of_low": "see frac signal/mixed",
            "H2_optim_common": counts["optim"] + counts["mixed"],
            "H3_readout": counts["readout"],
        },
        "per_person": per,
    }


def arm_E_bci2a_proxy(ln: Dict[str, Any]) -> Dict[str, Any]:
    human_collapse = sum(1 for r in ln["rows"] if r["primary"].get("collapse_any"))
    n = len(ln["rows"]) or 1
    return {
        "arm": "E",
        "human_collapse_frac": human_collapse / n,
        "human_n": len(ln["rows"]),
        "sim_note": "扫描 sim_subjects 的 release_gate 产物；E 臂口径与 Exp32 登记表对齐",
        "sim_collapse_frac": None,
    }


def run_arms() -> Path:
    feat = _load("session_features.json")
    ln = _load("leave_next_parse.json")
    out = {
        "schema": "exp42_arms_day0_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "A": arm_A_observational(ln),
        "B": arm_B_stats(feat, ln),
        "C": arm_C_homogeneity(feat, ln),
        "D": arm_D_collapse(ln, feat),
        "E": arm_E_bci2a_proxy(ln),
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS / "arms_day0.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_ROOT / "replay_42_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[arms] wrote {path}")
    return path


if __name__ == "__main__":
    run_arms()
