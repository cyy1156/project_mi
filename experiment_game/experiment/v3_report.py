"""v3 会话报告：分块准确率 + 特征 + 引导效应。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

LABEL_NAMES = {0: "Rest", 1: "Left", 2: "Right"}


def _acc_at_primary(records: List[Dict], primary_s: float) -> Dict[str, Any]:
    scored = [
        r for r in records
        if r.get("label") in (1, 2) and r.get("valid") and not r.get("signal_bad")
    ]
    if not scored:
        return {"n": 0, "acc_argmax": None, "acc_gated": None, "margin_mean": None}
    correct_a, correct_g, margins = [], [], []
    for r in scored:
        j = _primary_judge(r, primary_s)
        if j is None:
            continue
        lab = int(r["label"])
        correct_a.append(int(j.get("pred")) == lab)
        correct_g.append(int(j.get("gated_pred", j.get("pred"))) == lab)
        if j.get("margin") is not None:
            margins.append(float(j["margin"]))
    n = len(correct_a)
    if n == 0:
        return {"n": 0, "acc_argmax": None, "acc_gated": None, "margin_mean": None}
    return {
        "n": n,
        "acc_argmax": round(float(np.mean(correct_a)), 4),
        "acc_gated": round(float(np.mean(correct_g)), 4),
        "margin_mean": round(float(np.mean(margins)), 4) if margins else None,
    }


def _primary_judge(record: Dict, primary_s: float) -> Optional[Dict]:
    js = record.get("judgments") or []
    if not js:
        return None
    best = min(js, key=lambda j: abs(float(j.get("t_rel", -1)) - primary_s))
    return best


def _confusion(records: List[Dict], primary_s: float, *, gated: bool = False) -> List[List[int]]:
    mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for r in records:
        if r.get("label") not in (0, 1, 2) or r.get("signal_bad"):
            continue
        j = _primary_judge(r, primary_s)
        if j is None:
            continue
        pred = int(j.get("gated_pred" if gated else "pred", 0))
        mat[int(r["label"])][pred] += 1
    return mat


def _block_features(records: List[Dict]) -> Dict[str, Any]:
    lr = [
        r for r in records
        if r.get("label") in (1, 2) and r.get("features") and not r.get("signal_bad")
    ]
    if not lr:
        return {}
    erds = [float(r["features"].get("mu_erd_contra", 0)) for r in lr if "mu_erd_contra" in r["features"]]
    lats = [float(r["features"].get("laterality_pp", 0)) for r in lr if "laterality_pp" in r["features"]]
    bls = [float(r["features"].get("betal_erd_contra", 0)) for r in lr if "betal_erd_contra" in r["features"]]
    bhs = [float(r["features"].get("betah_erd_contra", 0)) for r in lr if "betah_erd_contra" in r["features"]]
    rms = [
        float(r["features"].get("rest_mu_frac", 0))
        for r in records
        if r.get("features") and not r.get("signal_bad")
    ]
    grades = [r["features"].get("grade", {}).get("grade") for r in lr if r.get("features")]
    weak = sum(1 for g in grades if g == "弱/不明显")
    return {
        "mu_erd_contra_mean": round(float(np.mean(erds)), 2) if erds else None,
        "laterality_pp_mean": round(float(np.mean(lats)), 2) if lats else None,
        "betal_erd_contra_mean": round(float(np.mean(bls)), 2) if bls else None,
        "betah_erd_contra_mean": round(float(np.mean(bhs)), 2) if bhs else None,
        "rest_mu_frac_mean": round(float(np.mean(rms)), 3) if rms else None,
        "weak_grade_frac": round(weak / len(grades), 3) if grades else None,
    }


def _quality_tier(acc: Optional[float], weak_frac: Optional[float]) -> str:
    if acc is None:
        return "unknown"
    if acc >= 0.60:
        return "zeroshot_ok"
    if acc >= 0.45:
        return "needs_calibration"
    if acc < 0.35:
        return "check_electrodes"
    if weak_frac is not None and weak_frac >= 0.5:
        return "weak_mi_candidate"
    return "needs_calibration"


def _signal_quality_section(
    block_records: Dict[str, List[Dict]],
    baseline: Optional[Dict] = None,
) -> Dict[str, Any]:
    hat = (baseline or {}).get("hat_check") or {}
    reason_hist: Dict[str, int] = {}
    n_signal_bad = 0
    for recs in block_records.values():
        for r in recs:
            if not r.get("signal_bad"):
                continue
            n_signal_bad += 1
            reason = r.get("signal_reason")
            if not reason:
                for j in r.get("judgments") or []:
                    if j.get("signal_bad") and j.get("reason"):
                        reason = j.get("reason")
                        break
            key = str(reason or "unknown")
            reason_hist[key] = reason_hist.get(key, 0) + 1
    return {
        "baseline_hat": hat,
        "n_signal_bad": n_signal_bad,
        "signal_bad_by_reason": reason_hist,
    }


def build_v3_report(
    *,
    block_order: List[str],
    block_records: Dict[str, List[Dict]],
    primary_judge_s: float,
    frozen: bool = True,
    invalid_streak_max: int = 0,
    baseline: Optional[Dict] = None,
) -> Dict[str, Any]:
    blocks: Dict[str, Any] = {}
    for cond in block_order:
        recs = block_records.get(cond, [])
        acc = _acc_at_primary(recs, primary_judge_s)
        feat = _block_features(recs)
        blocks[cond] = {
            "n_trials": len(recs),
            "accuracy": acc,
            "confusion_argmax": _confusion(recs, primary_judge_s, gated=False),
            "confusion_gated": _confusion(recs, primary_judge_s, gated=True),
            "features": feat,
        }

    delta: Dict[str, Any] = {}
    if "no_guide" in blocks and "guided" in blocks:
        for key in ("acc_argmax", "acc_gated", "margin_mean"):
            a = blocks["no_guide"]["accuracy"].get(key)
            b = blocks["guided"]["accuracy"].get(key)
            if a is not None and b is not None:
                delta[f"delta_{key}"] = round(b - a, 4)
        for fk in ("mu_erd_contra_mean", "laterality_pp_mean"):
            a = blocks["no_guide"]["features"].get(fk)
            b = blocks["guided"]["features"].get(fk)
            if a is not None and b is not None:
                delta[f"delta_{fk}"] = round(b - a, 2)

    guided_acc = blocks.get("guided", {}).get("accuracy", {}).get("acc_argmax")
    weak_frac = blocks.get("guided", {}).get("features", {}).get("weak_grade_frac")
    if guided_acc is None:
        guided_acc = blocks.get("no_guide", {}).get("accuracy", {}).get("acc_argmax")
        weak_frac = blocks.get("no_guide", {}).get("features", {}).get("weak_grade_frac")

    report = {
        "frozen": frozen,
        "block_order": block_order,
        "primary_judge_s": primary_judge_s,
        "blocks": blocks,
        "guidance_effect": delta,
        "quality_tier": _quality_tier(guided_acc, weak_frac),
        "invalid_streak_max": invalid_streak_max,
        "baseline": baseline or {},
        "signal_quality": _signal_quality_section(block_records, baseline),
    }
    return report


def write_v3_report(session_dir: Path, report: Dict[str, Any]) -> None:
    session_dir = Path(session_dir)
    (session_dir / "v3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md = _report_md(report)
    (session_dir / "v3_report.md").write_text(md, encoding="utf-8")


def _report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# v3 零样本探针报告",
        "",
        f"- 权重冻结：`{report.get('frozen')}`",
        f"- 块顺序：`{' → '.join(report.get('block_order') or [])}`",
        f"- 主判定点：{report.get('primary_judge_s')}s",
        f"- 底座质量定级：**{report.get('quality_tier')}**",
        "- 口径：特征卡/块统计的 ERD 相对块内滚动 Rest 基线；操作台功率条相对开场 60s 基线；信号质量不足的试次已从 acc/ERD 统计剔除",
        "",
        "## 分块准确率",
        "",
        "| 条件 | n | acc(argmax) | acc(gated) | 裕度 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond, blk in (report.get("blocks") or {}).items():
        acc = blk.get("accuracy") or {}
        lines.append(
            f"| {cond} | {acc.get('n', 0)} | "
            f"{_fmt_pct(acc.get('acc_argmax'))} | {_fmt_pct(acc.get('acc_gated'))} | "
            f"{acc.get('margin_mean') or '—'} |"
        )
    lines.extend(["", "## 分块特征", ""])
    for cond, blk in (report.get("blocks") or {}).items():
        f = blk.get("features") or {}
        lines.append(
            f"**{cond}**：对侧 mu ERD {f.get('mu_erd_contra_mean', '—')}% · "
            f"偏侧 {f.get('laterality_pp_mean', '—')}pp · "
            f"rest_mu_frac {f.get('rest_mu_frac_mean', '—')}"
        )
    delta = report.get("guidance_effect") or {}
    if delta:
        lines.extend(["", "## 引导效应 (guided − no_guide)", ""])
        for k, v in delta.items():
            lines.append(f"- {k}: {v}")
    sq = report.get("signal_quality") or {}
    hat = sq.get("baseline_hat") or {}
    if hat:
        lines.extend(["", "## 信号质量", ""])
        lines.append(f"- 基线帽检：**{hat.get('verdict', '—')}** · {hat.get('message', '—')}")
        lines.append(
            f"- 坏窗占比：{hat.get('bad_frac_pct', '—')}% "
            f"({hat.get('n_bad', 0)}/{hat.get('n_windows', 0)} 窗)"
        )
        rc = hat.get("reason_counts") or {}
        if rc:
            lines.append(
                "- 基线坏窗原因："
                + " · ".join(f"{k}={v}" for k, v in rc.items())
            )
        dead = hat.get("dead_channel_indices") or []
        if dead:
            lines.append(f"- 死通道索引：{dead}")
    n_bad = sq.get("n_signal_bad")
    hist = sq.get("signal_bad_by_reason") or {}
    if n_bad or hist:
        lines.append(f"- 试次 signal_bad 剔除：{n_bad or 0} 次")
        if hist:
            lines.append(
                "- 剔除原因分布："
                + " · ".join(f"{k}={v}" for k, v in hist.items())
            )
    return "\n".join(lines) + "\n"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.1f}%"
