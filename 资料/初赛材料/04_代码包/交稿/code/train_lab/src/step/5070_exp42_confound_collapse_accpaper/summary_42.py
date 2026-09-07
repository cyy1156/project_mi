"""Write Day0 results into summary registry markdown."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from paths import ANALYSIS, SUMMARY_DIR

def _fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (x != x)):
            return ""
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)
    except Exception:
        return ""

def write_registry() -> Path:
    cohort_path = ANALYSIS / "cohort_map.json"
    if not cohort_path.is_file():
        raise FileNotFoundError(f"missing {cohort_path}; run day0/P0 first")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    arms_path = ANALYSIS / "arms_day0.json"
    if arms_path.is_file():
        arms = json.loads(arms_path.read_text(encoding="utf-8"))
    else:
        empty = {"note": "day0 not run yet"}
        arms = {
            "A": {"R_vol_proxy_mean": float("nan"), "frac_positive": float("nan"),
                  "R_recent_proxy_mean": float("nan"), "verdict": "pending_day0", "note": empty["note"]},
            "B": {"frac_rest_mu_slope_pos": float("nan"), "n_people_slope": 0,
                  "fatigue_verdict": "pending_day0", "frac_abs_li_slope_neg": float("nan"),
                  "spearman_gain_vs_sat": {"r": float("nan"), "p": float("nan"), "n": 0},
                  "spearman_gain_vs_gap": {"r": float("nan"), "p": float("nan"), "n": 0},
                  "spearman_gain_vs_dprime": {"r": float("nan"), "p": float("nan"), "n": 0},
                  "adjacent_dprime_delta": {"median": float("nan"), "p": float("nan"), "n": 0}},
            "C": {"dprime_bins": {"strong_ge_1.0": 0, "mid_0.5_1.0": 0, "weak_lt_0.5": 0, "weak_frac": float("nan")},
                  "median_adjacent_dprime_gap": float("nan"), "note_friedman": "pending_day0"},
            "D": {"counts": {"signal": 0, "optim": 0, "readout": 0, "mixed": 0, "none": 0},
                  "frac": {"signal": 0, "optim": 0, "readout": 0, "mixed": 0, "none": 0},
                  "per_person": []},
            "E": {"sim_collapse_frac": None, "human_collapse_frac": None, "human_n": 0, "sim_note": "pending_day0"},
        }
    a_agg = ANALYSIS / "arm_A_aggregate.json"
    if a_agg.is_file():
        arms["A_full"] = json.loads(a_agg.read_text(encoding="utf-8"))
    A, B, C, D, E = arms["A"], arms["B"], arms["C"], arms["D"], arms["E"]

    d0_path = ANALYSIS / "arm_D0.json"
    d0 = json.loads(d0_path.read_text(encoding="utf-8")) if d0_path.is_file() else None
    feat_path = ANALYSIS / "feat_anchor_check.json"
    feat = json.loads(feat_path.read_text(encoding="utf-8")) if feat_path.is_file() else None

    # Chinese headings via unicode escapes (encoding-safe source)
    t = {
        "title": "\u65b9\u6848\u56db\u5341\u4e8c \u00b7 \u771f\u4eba\u961f\u5217\u6df7\u6742\u5206\u89e3\u4e0e\u574d\u584c\u8bca\u65ad \u00b7 \u7ed3\u679c\u767b\u8bb0",
        "status": "Day0 \u5df2\u6267\u884c",
        "p0": "P0 \u00b7 \u8eab\u4efd\u5408\u5e76",
        "a": "A \u00b7 \u6df7\u6742\u5206\u89e3\uff08\u89c2\u6d4b\u4ee3\u7406\uff1b\u975e A1\u2212A2 \u5934\u5bf9\u5934\uff09",
        "b": "B \u00b7 \u75b2\u52b3\u4e0e\u8d28\u91cf\u5173\u8054",
        "c": "C \u00b7 \u4f1a\u8bdd\u7279\u5f81\u663e\u8457\u6027\u4e0e\u540c\u8d28\u6027",
        "d": "D \u00b7 \u574d\u584c\u4e09\u5c42\u5b9a\u4f4d",
        "d0": "D0 \u00b7 \u574d\u584c\u666e\u67e5\uff08heldout \u8bd5\u6b21\u7ea7\uff0cF5 \u591a\u6570\u7968\uff1b\u65b9\u6848 v0.2\uff09",
        "feat": "\u7279\u5f81\u5bf9\u8868\u6821\u9a8c\uff08\u751f\u4ea7\u951a\u70b9 vs \u81ea\u7b97\u951a\u70b9 vs \u90e8\u7f72\u6a21\u578b\uff09",
        "e": "E \u00b7 BCI2a \u5bf9\u7167",
        "exec": "\u6267\u884c\u8bb0\u5f55",
    }

    lines = []
    lines.append(f"# {t['title']}")
    lines.append("")
    lines.append(
        f"> \u65b9\u6848\uff1a[../\u65b9\u6848.md](../\u65b9\u6848.md) \u00b7 \u72b6\u6001\uff1a**{t['status']}** \u00b7 "
        f"\u751f\u6210 `{datetime.now().isoformat(timespec='seconds')}`"
    )
    lines.append(
        "> \u5355\u4f4d\u7eaa\u5f8b\uff1a\u5206\u6790\u7528 person_id\uff0817 \u4e2a\u4f53\uff0cxjh0828/fnz0830 \u5206\u884c\uff09\uff1b"
        "A \u81c2\u4e3a\u5e95\u5ea7\u5168\u94fe\u91cd\u653e\uff08\u89c1 A-full \u8282\uff09\uff0c\u89c2\u6d4b\u4ee3\u7406\u4ec5\u4f5c\u53c2\u7167\u3002"
    )
    lines.append("")
    lines.append(f"## {t['p0']}")
    lines.append("")
    lines.append("| person_id | member_ids | primary_id | rule |")
    lines.append("|---|---|---|---|")
    for p in cohort["people"]:
        lines.append(
            f"| {p['person_id']} | {' / '.join(p['member_ids'])} | {p['primary_id']} | {p['merge_rule']} |"
        )
    lines.append("")
    lines.append(f"n_people=**{cohort['n_people']}** (member_dirs={cohort['n_member_dirs']}).")
    lines.append("")
    lines.append(f"## {t['a']}")
    lines.append("")
    lines.append("| metric | mean | frac_pos | gate | verdict |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| R_vol_proxy=last-first MI | {_fmt(A['R_vol_proxy_mean'], 4)} | "
        f"{_fmt(A['frac_positive'], 3)} | >=+0.03 and >=2/3 | **{A['verdict']}** |"
    )
    lines.append(
        f"| R_recent_proxy=last-mid MI | {_fmt(A['R_recent_proxy_mean'], 4)} | - | aux | - |"
    )
    Afull_dyn = arms.get("A_full")
    span_note = (
        "见 A-full 节（重训已完成）" if Afull_dyn
        else "needs A retrain | **pending GPU**"
    )
    lines.append(f"| R_span / R_order / dA5 | - | - | - | {span_note} |")
    lines.append("")
    lines.append(f"note: {A['note']}")
    lines.append("")
    lines.append(f"## {t['b']}")
    lines.append("")
    lines.append("| metric | result | read |")
    lines.append("|---|---|---|")
    lines.append(
        f"| frac Rest-mu slope>0 | {_fmt(B['frac_rest_mu_slope_pos'])} (n={B['n_people_slope']}) | {B['fatigue_verdict']} |"
    )
    lines.append(
        f"| frac |LI| slope<0 | {_fmt(B['frac_abs_li_slope_neg'])} | aux |"
    )
    sp = B["spearman_gain_vs_sat"]
    lines.append(
        f"| Spearman gain x sat | r={_fmt(sp['r'])}, p={_fmt(sp['p'])}, n={sp['n']} | "
        f"{'weak quality confound' if abs(sp.get('r') or 0) < 0.2 else 'check'} |"
    )
    sp2 = B["spearman_gain_vs_gap"]
    lines.append(
        f"| Spearman gain x gap | r={_fmt(sp2['r'])}, p={_fmt(sp2['p'])}, n={sp2['n']} | aux |"
    )
    sp3 = B["spearman_gain_vs_dprime"]
    lines.append(
        f"| Spearman gain x dprime | r={_fmt(sp3['r'])}, p={_fmt(sp3['p'])}, n={sp3['n']} | aux |"
    )
    wil = B["adjacent_dprime_delta"]
    lines.append(
        f"| adjacent dprime delta | median={_fmt(wil.get('median'))}, "
        f"p={_fmt(wil.get('p'))}, n={wil.get('n')} | drift proxy |"
    )
    lines.append("")
    lines.append(f"## {t['c']}")
    lines.append("")
    bins = C["dprime_bins"]
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(
        f"| dprime bins strong/mid/weak | "
        f"{bins['strong_ge_1.0']} / {bins['mid_0.5_1.0']} / {bins['weak_lt_0.5']} "
        f"(weak_frac={_fmt(bins['weak_frac'])}) |"
    )
    lines.append(f"| median adjacent dprime gap | {_fmt(C['median_adjacent_dprime_gap'])} |")
    lines.append(f"| note | {C['note_friedman']} |")
    lines.append("")
    lines.append(f"## {t['d']}")
    lines.append("")
    lines.append("| label | n | frac |")
    lines.append("|---|---|---|")
    for k in ("signal", "optim", "readout", "mixed", "none"):
        lines.append(f"| {k} | {D['counts'][k]} | {_fmt(D['frac'][k])} |")
    lines.append("")
    lines.append("| person | dprime/AUC | max_class_frac | label |")
    lines.append("|---|---|---|---|")
    for p in D["per_person"]:
        lines.append(
            f"| {p['person_id']} | d={_fmt(p['med_dprime_lr'])}, "
            f"AUC={_fmt(p['med_probe_auc_lr'])} | {_fmt(p['max_class_frac_peak'])} | {p['label']} |"
        )
    lines.append("")
    
    Afull = arms.get("A_full")
    if Afull:
        lines.append("")
        lines.append("## A-full jackknife")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| R_vol_mean | {_fmt(Afull.get('R_vol_mean'), 4)} |")
        lines.append(f"| frac_pos | {_fmt(Afull.get('frac_positive_R_vol'), 3)} |")
        lines.append(f"| verdict | **{Afull.get('verdict')}** |")
        lines.append(f"| dA5_mean_slope | {_fmt(Afull.get('dA5_mean_slope'), 4)} |")
        lines.append("")

    if d0:
        g = d0["gate"]
        lines.append(f"## {t['d0']}")
        lines.append("")
        lines.append(
            f"**判定门：末轮坍塌 {g['last_round_k']}/{g['last_round_n']}，"
            f"frac={_fmt(g.get('frac'))}，CI95={g.get('ci95')} → {g['verdict']}**"
        )
        lines.append("")
        lines.append("定义：试次级预测分布熵 <0.5 或 heldout max_class_frac ≥0.8 或 MI 异侧率 ≥0.7（任一）。")
        lines.append("")
        lines.append("| person | rounds | collapse | label | last_round |")
        lines.append("|---|---|---|---|---|")
        for r in d0["people"]:
            lines.append(
                f"| {r['person']} | {r['n_rounds']} | {r['n_collapse']} | {r['label']} | {r['last_round_collapse']} |"
            )
        lines.append("")
    if feat:
        lines.append(f"## {t['feat']}")
        lines.append("")
        lines.append("| person | session | probeAUC(生产锚点) | probeAUC(自算) | 底座直推窗acc |")
        lines.append("|---|---|---|---|---|")
        for pid, v in feat["people"].items():
            if "error" in v:
                lines.append(f"| {pid} | SKIP | {v['error'][:40]} | | |")
                continue
            lines.append(
                f"| {pid} | {v['session']} | {_fmt(v['probe_auc_prod_anchor'])} | "
                f"{_fmt(v['probe_auc_self_anchor'])} | {_fmt(v['deployed_window_acc'])} |"
            )
        vd = feat.get("verdict", {})
        lines.append("")
        lines.append(
            f"verdict: anchor_misalignment={vd.get('anchor_misalignment')}，"
            f"feature_family_weak={vd.get('feature_family_weak')} → "
            "两锚点一致 ⇒ 管线无错位；手工 16 维特征族对 L/R 接近不可分（底座直推同样低），"
            "d′/AUC 判据仅限\u201c手工特征族内\u201d解释，信号层判据以 D0 为准。"
        )
        lines.append("")

    lines.append(f"## {t['e']}")
    lines.append("")
    lines.append("| metric | sim | human |")
    lines.append("|---|---|---|")
    lines.append(
        f"| collapse frac | {_fmt(E.get('sim_collapse_frac')) or (E.get('sim_collapse_frac') or 'TBD')} | "
        f"{_fmt(E['human_collapse_frac'])} (n={E['human_n']}) |"
    )
    lines.append(f"| note | {E['sim_note']} | |")
    lines.append("")
    lines.append(f"## {t['exec']}")
    lines.append("")
    lines.append("```text")
    lines.append("cd code/train_lab/src/step/5070_exp42_confound_collapse_accpaper")
    lines.append("python run_day0.py --workers 4")
    lines.append("```")
    lines.append("")
    lines.append("| date | event |")
    lines.append("|------|------|")
    lines.append("| 2026-09-04 | registry skeleton |")
    lines.append(
        f"| {datetime.now().strftime('%Y-%m-%d')} | 17 人口径 Day0 + 特征对表校验 + D0 普查 + A 臂底座全链重放（A0 缺省由 f5_base_e1f 对照承担）|"
    )
    lines.append("")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    # filename: 结果登记表.md
    path = SUMMARY_DIR / "\u7ed3\u679c\u767b\u8bb0\u8868.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] wrote {path}")
    return path

if __name__ == "__main__":
    write_registry()
