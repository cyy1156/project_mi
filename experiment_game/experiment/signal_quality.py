"""EEG 信号质量门控：空帽/断线/饱和/大幅伪迹/共模时不进入模型计分。

在 3s 判定窗的原始 µV 数据（LSL RingBuffer）上检查，不依赖阻抗 API。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SignalQualityConfig:
    enabled: bool = True
    min_median_std_uv: float = 3.0
    min_peak_to_peak_uv: float = 8.0
    max_peak_uv: float = 1000.0
    min_per_channel_std_uv: float = 2.0
    min_active_channels: int = 3
    max_channel_std_ratio: float = 20.0
    flatline_max_abs_uv: float = 0.5
    flatline_frac_max: float = 0.85
    max_median_std_uv: float = 60.0
    max_ptp_uv: float = 400.0
    min_car_std_uv: float = 2.0
    max_common_mode_ratio: float = 0.85
    max_per_channel_std_uv: float = 60.0


def _normalize_tc(x: np.ndarray) -> tuple[Optional[np.ndarray], Optional[Dict]]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None, {"ok": False, "reason": "empty_window", "metrics": {}}
    if x.ndim != 2:
        return None, {"ok": False, "reason": "bad_shape", "metrics": {"shape": list(x.shape)}}
    if x.shape[0] == 8 and x.shape[1] != 8:
        x = x.T
    elif x.shape[1] != 8 and x.shape[0] != 8:
        return None, {"ok": False, "reason": "bad_channels", "metrics": {"shape": list(x.shape)}}
    if not np.all(np.isfinite(x)):
        return None, {"ok": False, "reason": "non_finite", "metrics": {}}
    return x, None


def assess_eeg_window(
    win_tc: np.ndarray,
    cfg: Optional[SignalQualityConfig] = None,
) -> Dict:
    """评估 (T, C) 或 (C, T) EEG 窗。返回 {ok, reason, metrics}。"""
    cfg = cfg or SignalQualityConfig()
    if not cfg.enabled:
        return {"ok": True, "reason": None, "metrics": {}}

    x, err = _normalize_tc(win_tc)
    if err is not None:
        return err

    ch_std = np.std(x, axis=0)
    peak = float(np.max(np.abs(x)))
    ptp = float(np.max(x) - np.min(x))
    med_std = float(np.median(ch_std))
    active = int(np.sum(ch_std >= cfg.min_per_channel_std_uv))

    x_car = x - x.mean(axis=1, keepdims=True)
    car_std = np.std(x_car, axis=0)
    car_min_std = float(np.min(car_std))
    dead_idx = int(np.argmin(car_std))

    ch_var = np.var(x, axis=0)
    med_ch_var = float(np.median(ch_var))
    common_var = float(np.var(x.mean(axis=1)))
    cm_ratio = common_var / med_ch_var if med_ch_var > 1e-12 else 0.0

    metrics = {
        "median_std_uv": round(med_std, 4),
        "peak_uv": round(peak, 4),
        "ptp_uv": round(ptp, 4),
        "active_channels": active,
        "max_ch_std_uv": round(float(np.max(ch_std)), 4),
        "car_min_std_uv": round(car_min_std, 4),
        "dead_channel_idx": dead_idx,
        "common_mode_ratio": round(float(cm_ratio), 4),
    }

    flat_frac = float(np.mean(np.max(np.abs(x), axis=1) < cfg.flatline_max_abs_uv))
    metrics["flatline_frac"] = round(flat_frac, 4)

    if flat_frac >= cfg.flatline_frac_max:
        return {"ok": False, "reason": "flatline", "metrics": metrics}
    if med_std < cfg.min_median_std_uv:
        return {"ok": False, "reason": "low_variance", "metrics": metrics}
    if ptp < cfg.min_peak_to_peak_uv:
        return {"ok": False, "reason": "low_dynamics", "metrics": metrics}
    if peak > cfg.max_peak_uv:
        return {"ok": False, "reason": "saturation", "metrics": metrics}
    if active < cfg.min_active_channels:
        return {"ok": False, "reason": "too_few_active_channels", "metrics": metrics}
    if med_std > 1e-9:
        ratio = float(np.max(ch_std) / med_std)
        metrics["channel_std_ratio"] = round(ratio, 4)
        if ratio > cfg.max_channel_std_ratio:
            return {"ok": False, "reason": "channel_imbalance", "metrics": metrics}

    if med_std > cfg.max_median_std_uv or ptp > cfg.max_ptp_uv:
        return {"ok": False, "reason": "artifact", "metrics": metrics}
    if car_min_std < cfg.min_car_std_uv:
        return {"ok": False, "reason": "dead_channel", "metrics": metrics}
    if cm_ratio > cfg.max_common_mode_ratio:
        return {"ok": False, "reason": "common_mode", "metrics": metrics}

    return {"ok": True, "reason": None, "metrics": metrics}


def summarize_baseline_hat_check(
    baseline_tc: np.ndarray,
    *,
    fs: float = 250.0,
    win_s: float = 3.0,
    cfg: Optional[SignalQualityConfig] = None,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """基线 60s 切成 3s 窗逐窗评估，汇总帽检结论。"""
    cfg = cfg or SignalQualityConfig()
    names = channel_names or [f"Ch{i}" for i in range(8)]
    x, err = _normalize_tc(baseline_tc)
    if err is not None:
        return {
            "verdict": "fail",
            "message": f"帽检未通过：{err['reason']}",
            "n_windows": 0,
            "n_bad": 0,
            "bad_frac": 1.0,
            "bad_frac_pct": 100.0,
            "reason_counts": {},
            "dead_channel_indices": [],
            "median_std_uv": None,
        }

    win_n = max(1, int(round(win_s * fs)))
    n = int(x.shape[0])
    if n < win_n:
        return {
            "verdict": "fail",
            "message": "帽检未通过：基线数据不足 3s",
            "n_windows": 0,
            "n_bad": 0,
            "bad_frac": 1.0,
            "bad_frac_pct": 100.0,
            "reason_counts": {},
            "dead_channel_indices": [],
            "median_std_uv": None,
        }

    results: List[Dict] = []
    med_stds: List[float] = []
    dead_indices: set[int] = set()
    for start in range(0, n - win_n + 1, win_n):
        qa = assess_eeg_window(x[start : start + win_n], cfg)
        results.append(qa)
        med_stds.append(float((qa.get("metrics") or {}).get("median_std_uv", 0.0)))
        if qa.get("reason") == "dead_channel":
            idx = (qa.get("metrics") or {}).get("dead_channel_idx")
            if idx is not None:
                dead_indices.add(int(idx))

    n_windows = len(results)
    bad = [r for r in results if not r.get("ok")]
    n_bad = len(bad)
    bad_frac = n_bad / n_windows if n_windows else 1.0
    reason_counts = dict(Counter(r.get("reason") for r in bad if r.get("reason")))

    median_std_uv = float(np.median(med_stds)) if med_stds else None
    dead_list = sorted(dead_indices)

    if bad_frac >= 0.5:
        verdict = "fail"
    elif bad_frac >= 0.2:
        verdict = "warn"
    else:
        verdict = "pass"

    message = _format_hat_message(
        verdict=verdict,
        bad_frac_pct=100.0 * bad_frac,
        reason_counts=reason_counts,
        n_windows=n_windows,
        dead_list=dead_list,
        channel_names=names,
        median_std_uv=median_std_uv,
    )

    return {
        "verdict": verdict,
        "message": message,
        "n_windows": n_windows,
        "n_bad": n_bad,
        "bad_frac": round(bad_frac, 4),
        "bad_frac_pct": round(100.0 * bad_frac, 1),
        "reason_counts": reason_counts,
        "dead_channel_indices": dead_list,
        "median_std_uv": round(median_std_uv, 2) if median_std_uv is not None else None,
    }


def _format_hat_message(
    *,
    verdict: str,
    bad_frac_pct: float,
    reason_counts: Dict[str, int],
    n_windows: int,
    dead_list: List[int],
    channel_names: List[str],
    median_std_uv: Optional[float],
) -> str:
    if verdict == "pass":
        med = median_std_uv if median_std_uv is not None else 0.0
        return f"帽检通过（中位 std {med:.1f} µV）"

    parts: List[str] = []
    if reason_counts.get("artifact"):
        pct = 100.0 * reason_counts["artifact"] / max(n_windows, 1)
        parts.append(f"大幅伪迹 {pct:.0f}%")
    if dead_list:
        dead_names = "、".join(
            channel_names[i] if i < len(channel_names) else f"Ch{i}" for i in dead_list
        )
        parts.append(f"{dead_names} 死通道")
    elif reason_counts.get("dead_channel"):
        parts.append("死通道")
    if reason_counts.get("common_mode"):
        parts.append("参考电极接触不良")

    other = [
        k
        for k in reason_counts
        if k not in ("artifact", "dead_channel", "common_mode")
    ]
    for k in other:
        parts.append(f"{k} {reason_counts[k]}窗")

    detail = "、".join(parts) if parts else f"坏窗 {bad_frac_pct:.0f}%"
    if verdict == "warn":
        return f"帽检警告：{detail}（坏窗 {bad_frac_pct:.0f}%）— 建议检查电极"
    return f"帽检未通过：{detail}——建议修好电极重跑"


_REASON_HINTS: Dict[str, str] = {
    "common_mode": "参考电极接触不良：检查 SRB2 + Bias，轻压耳后/乳突",
    "dead_channel": "去共模后无信号：补胶/按紧该电极",
    "artifact": "大幅伪迹：减少头动/咬牙，检查导线拉扯",
    "high_std": "单通道幅度过大：检查该电极接触与导电膏",
    "saturation": "饱和：检查电极是否翘起造成跳变",
    "low_std": "信号过平：电极可能未接触皮肤",
    "flatline": "信号过平：电极可能未接触皮肤",
    "channel_imbalance": "通道间幅度悬殊：检查松动电极或导电膏不均",
    "low_variance": "整体方差过低：检查帽是否戴好",
    "low_dynamics": "动态范围过小：检查电极接触",
    "too_few_active_channels": "有效通道不足：检查多个电极接触",
}


def _channel_diagnose(
    x: np.ndarray,
    cfg: SignalQualityConfig,
    channel_names: List[str],
) -> List[Dict[str, Any]]:
    """逐通道健康检查（v4 热力图）。"""
    ch_std = np.std(x, axis=0)
    x_car = x - x.mean(axis=1, keepdims=True)
    car_std = np.std(x_car, axis=0)
    med_std = float(np.median(ch_std)) if ch_std.size else 0.0
    out: List[Dict[str, Any]] = []
    n_ch = x.shape[1]
    for i in range(n_ch):
        name = channel_names[i] if i < len(channel_names) else f"Ch{i}"
        std_uv = float(ch_std[i])
        car_uv = float(car_std[i])
        peak_i = float(np.max(np.abs(x[:, i])))
        flat_frac = float(np.mean(np.abs(x[:, i]) < cfg.flatline_max_abs_uv))
        reason: Optional[str] = None
        if peak_i > cfg.max_peak_uv:
            reason = "saturation"
        elif car_uv < cfg.min_car_std_uv:
            reason = "dead_channel"
        elif std_uv > cfg.max_per_channel_std_uv:
            reason = "high_std"
        elif med_std > 1e-9 and std_uv / med_std > cfg.max_channel_std_ratio:
            reason = "imbalance"
        elif std_uv < cfg.min_per_channel_std_uv:
            reason = "low_std"
        elif flat_frac >= cfg.flatline_frac_max:
            reason = "flatline"
        out.append(
            {
                "idx": i,
                "name": name,
                "ok": reason is None,
                "std_uv": round(std_uv, 2),
                "car_std_uv": round(car_uv, 2),
                "peak_uv": round(peak_i, 2),
                "reason": reason,
            }
        )
    return out


def _build_problems(
    window_reason: Optional[str],
    per_channel: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> List[Dict[str, str]]:
    problems: List[Dict[str, str]] = []
    dead_names = [ch["name"] for ch in per_channel if ch.get("reason") == "dead_channel"]

    if window_reason == "common_mode":
        cm = float(metrics.get("common_mode_ratio", 0.0))
        problems.append(
            {
                "channel": "",
                "reason": "common_mode",
                "detail": f"共模比 {cm * 100:.0f}%",
                "hint": _REASON_HINTS["common_mode"],
            }
        )
    elif window_reason and window_reason not in ("dead_channel",):
        problems.append(
            {
                "channel": "",
                "reason": window_reason,
                "detail": window_reason,
                "hint": _REASON_HINTS.get(window_reason, "请检查电极与参考"),
            }
        )

    if dead_names:
        if len(dead_names) >= 3:
            hint = "多通道死通道：先查参考电极，再逐个按通道"
        else:
            hint = f"{dead_names[0]} {_REASON_HINTS['dead_channel']}"
        for name in dead_names:
            ch = next(c for c in per_channel if c["name"] == name)
            problems.append(
                {
                    "channel": name,
                    "reason": "dead_channel",
                    "detail": f"CAR std={ch['car_std_uv']:.1f}µV",
                    "hint": hint,
                }
            )

    for ch in per_channel:
        r = ch.get("reason")
        if not r or r == "dead_channel":
            continue
        problems.append(
            {
                "channel": ch["name"],
                "reason": r,
                "detail": f"std={ch['std_uv']:.1f}µV",
                "hint": _REASON_HINTS.get(r, "请检查该通道电极"),
            }
        )
    return problems


def live_channel_stats(
    win_tc: np.ndarray,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """v4 快速通道反馈（~0.5s 窗）：只看 std/peak，用于触诊时看哪格在动。"""
    names = channel_names or [f"Ch{i}" for i in range(8)]
    x, err = _normalize_tc(win_tc)
    if err is not None:
        return {"per_channel_live": [], "active_idx": None}
    ch_std = np.std(x, axis=0)
    ch_peak = np.max(np.abs(x), axis=0)
    per: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        if i >= len(ch_std):
            break
        per.append(
            {
                "idx": i,
                "name": name,
                "std_uv": round(float(ch_std[i]), 1),
                "peak_uv": round(float(ch_peak[i]), 1),
            }
        )
    active_idx = int(np.argmax(ch_std)) if ch_std.size else None
    return {"per_channel_live": per, "active_idx": active_idx}


def diagnose_eeg_window(
    win_tc: np.ndarray,
    cfg: Optional[SignalQualityConfig] = None,
    *,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """v4 专用：窗级 + 逐通道诊断与操作建议。"""
    cfg = cfg or SignalQualityConfig()
    names = channel_names or [f"Ch{i}" for i in range(8)]
    assess = assess_eeg_window(win_tc, cfg)
    metrics = dict(assess.get("metrics") or {})
    x, err = _normalize_tc(win_tc)
    if err is not None:
        return {
            "window_ok": False,
            "window_reason": assess.get("reason") or err.get("reason"),
            "metrics": metrics,
            "per_channel": [],
            "per_channel_ok": [],
            "problems": [
                {
                    "channel": "",
                    "reason": err.get("reason") or "bad_window",
                    "detail": str(err.get("reason")),
                    "hint": "窗口数据无效",
                }
            ],
        }

    per_channel = _channel_diagnose(x, cfg, names)
    per_channel_ok = [bool(ch["ok"]) for ch in per_channel]
    sat_n = sum(1 for ch in per_channel if ch.get("reason") == "saturation")
    window_ok = bool(assess.get("ok")) and all(per_channel_ok)
    window_reason = assess.get("reason")
    if not window_ok and window_reason is None:
        bad = next((ch for ch in per_channel if not ch["ok"]), None)
        window_reason = bad["reason"] if bad else "channel_fail"

    med_std = float(metrics.get("median_std_uv", 0.0))
    ptp = float(metrics.get("ptp_uv", 0.0))
    cm = float(metrics.get("common_mode_ratio", 0.0))
    active = int(metrics.get("active_channels", 0))

    problems = _build_problems(window_reason, per_channel, metrics)
    if sat_n >= 4:
        problems.insert(
            0,
            {
                "channel": "",
                "reason": "touch_artifact",
                "detail": f"饱和 {sat_n}/8 通道",
                "hint": (
                    "按压/摩擦电极会产生大幅跳变（饱和），格子不会变绿。"
                    "触诊请极轻、只看哪格数字跳动；达标请松手静坐 15s"
                ),
            },
        )

    return {
        "window_ok": window_ok,
        "window_reason": window_reason,
        "metrics": {
            **metrics,
            "median_std_ok": 3.0 <= med_std <= cfg.max_median_std_uv,
            "ptp_ok": cfg.min_peak_to_peak_uv <= ptp <= cfg.max_ptp_uv,
            "ch_ok": active >= cfg.min_active_channels,
            "cm_ok": cm <= cfg.max_common_mode_ratio,
        },
        "per_channel": per_channel,
        "per_channel_ok": per_channel_ok,
        "problems": problems,
    }


def summarize_v4_session(
    history: List[Dict[str, Any]],
    *,
    duration_s: float,
    pass_streak_required: int,
    achieved_stable: bool,
    time_to_stable_s: Optional[float],
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """v4 会话结束汇总。"""
    names = channel_names or [f"Ch{i}" for i in range(8)]
    total = len(history)
    pass_windows = sum(1 for h in history if h.get("window_ok"))
    fail_windows = total - pass_windows
    pass_rate = pass_windows / total if total else 0.0

    med_stds: List[float] = []
    cm_ratios: List[float] = []
    reason_counts: Counter[str] = Counter()
    dead_hits: Counter[str] = Counter()
    hot_hits: Counter[str] = Counter()

    for h in history:
        m = h.get("metrics") or {}
        if m.get("median_std_uv") is not None:
            med_stds.append(float(m["median_std_uv"]))
        if m.get("common_mode_ratio") is not None:
            cm_ratios.append(float(m["common_mode_ratio"]))
        if not h.get("window_ok"):
            wr = h.get("window_reason")
            if wr:
                reason_counts[str(wr)] += 1
        for ch in h.get("per_channel") or []:
            r = ch.get("reason")
            if r == "dead_channel":
                dead_hits[ch.get("name", "?")] += 1
            elif r in ("high_std", "imbalance", "saturation"):
                hot_hits[ch.get("name", "?")] += 1

    chronic_dead = [n for n, c in dead_hits.items() if c >= max(2, total // 3)]
    chronic_hot = [n for n, c in hot_hits.items() if c >= max(2, total // 3)]

    last_n = history[-pass_streak_required:] if pass_streak_required > 0 else []
    tail_all_ok = len(last_n) >= pass_streak_required and all(h.get("window_ok") for h in last_n)

    if achieved_stable:
        verdict = "pass"
    elif pass_rate >= 0.8 and tail_all_ok:
        verdict = "warn"
    else:
        verdict = "fail"

    if verdict == "pass":
        recommendation = "信号稳定，可以开始 v3 实验"
    elif verdict == "warn":
        recommendation = "接近达标但不稳定：建议再修电极或延长检测"
    else:
        parts: List[str] = []
        if reason_counts.get("common_mode"):
            parts.append("优先检查参考电极")
        if chronic_dead:
            parts.append("死通道 " + "、".join(chronic_dead))
        if chronic_hot:
            parts.append("高幅通道 " + "、".join(chronic_hot))
        recommendation = "未达标：" + ("；".join(parts) if parts else "请检查帽与电极")

    cm_p95 = float(np.percentile(cm_ratios, 95)) if cm_ratios else None
    med_median = float(np.median(med_stds)) if med_stds else None

    return {
        "verdict": verdict,
        "achieved_stable": achieved_stable,
        "duration_s": round(float(duration_s), 1),
        "total_windows": total,
        "pass_windows": pass_windows,
        "fail_windows": fail_windows,
        "pass_rate": round(pass_rate, 4),
        "time_to_stable_s": round(time_to_stable_s, 1) if time_to_stable_s is not None else None,
        "median_std_uv_median": round(med_median, 2) if med_median is not None else None,
        "common_mode_ratio_p95": round(cm_p95, 4) if cm_p95 is not None else None,
        "chronic_dead_channels": chronic_dead,
        "chronic_hot_channels": chronic_hot,
        "top_reasons": dict(reason_counts),
        "recommendation": recommendation,
        "channel_labels": list(names),
    }

