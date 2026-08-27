#!/usr/bin/env python3
"""用历史会话连续 EEG 离线回放 v4 / v3 / v2 关键栈。

epochs 目录（2s）与现行 3s 模型不兼容；本工具自动映射到对应 sessions，
从 continuous/eeg.csv + trial_table 按在线协议重切 3s 窗。

用法（cyy 环境）::

    python -m experiment_game.tools.replay_pipeline_offline \\
        data/epochs/sub02_ses01_20260723_180607_slide_w2s_h100ms \\
        data/epochs/sub03_ses01_20260723_185153
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from experiment_game.experiment.signal_quality import (  # noqa: E402
    SignalQualityConfig,
    assess_eeg_window,
    diagnose_eeg_window,
    summarize_baseline_hat_check,
    summarize_v4_session,
)
from experiment_game.experiment.v4_config import V4Config  # noqa: E402
from experiment_game.experiment.v4_quality import V4QualityMonitor  # noqa: E402
from experiment_game.offline.phase4_v2 import FS, WIN, HOP, T0_MIN, load_eeg  # noqa: E402
from experiment_game.experiment.feature_probe import (  # noqa: E402
    bandpowers_fft,
    erd,
    segment_to_hop_windows,
)
from experiment_game.experiment.inference_v2 import CHANNEL_ORDER, OnlinePreprocessor  # noqa: E402
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402
from src.common.steps.resample_zscore import trial_zscore  # noqa: E402

N_TIMES = int(round(WIN * FS))
IX = {n: i for i, n in enumerate(CHANNEL_ORDER)}


def resolve_session(path: Path) -> Path:
    path = path.resolve()
    if (path / "continuous" / "eeg.csv").is_file() or (path / "eeg.csv").is_file():
        return path
    meta = path / "meta.json"
    if meta.is_file():
        raw = json.loads(meta.read_text(encoding="utf-8"))
        sd = Path(raw.get("session_dir") or "")
        if sd.is_dir():
            return sd
        # 旧盘符路径 → 本仓库 sessions
        name = sd.name if sd.name else path.name.replace("_slide_w2s_h100ms", "").replace("_w2s", "")
        cand = _ROOT / "data" / "sessions" / name
        if cand.is_dir():
            return cand
    # epochs 名剥后缀
    name = path.name
    for suf in ("_slide_w2s_h100ms", "_slide_w3s_h100ms", "_w2s", "_w3s"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    cand = _ROOT / "data" / "sessions" / name
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"无法解析会话目录: {path}")


def load_trial_table(session_dir: Path) -> List[dict]:
    p = session_dir / "alignment" / "trial_table.csv"
    if not p.is_file():
        raise FileNotFoundError(f"缺少 trial_table: {p}")
    return list(csv.DictReader(p.open(encoding="utf-8")))


def slice_tc(t_lsl: np.ndarray, X: np.ndarray, t0: float, t1: float) -> np.ndarray:
    i0 = int(np.searchsorted(t_lsl, t0))
    i1 = int(np.searchsorted(t_lsl, t1))
    return X[i0:i1]


def run_v4(t_lsl: np.ndarray, X_raw: np.ndarray, *, cfg: V4Config) -> dict:
    sq = cfg.signal_quality_config()
    names = list(cfg.channel_labels)
    mon = V4QualityMonitor(cfg)
    history = []
    win_n = max(1, int(round(cfg.eval_window_s * FS)))
    hop_n = max(1, int(round(cfg.eval_interval_s * FS)))
    for start in range(0, X_raw.shape[0] - win_n + 1, hop_n):
        win = X_raw[start : start + win_n]
        diag = diagnose_eeg_window(win, sq, channel_names=names)
        elapsed = float(t_lsl[start] - t_lsl[0]) if len(t_lsl) else start / FS
        mon.update(diag, elapsed_s=elapsed)
        history.append({**diag, "pass_streak": mon.streak})
    duration = float(t_lsl[-1] - t_lsl[0]) if len(t_lsl) > 1 else X_raw.shape[0] / FS
    summary = summarize_v4_session(
        history,
        duration_s=duration,
        pass_streak_required=cfg.pass_streak_required,
        achieved_stable=mon.achieved_stable,
        time_to_stable_s=mon.time_to_stable_s,
        channel_names=names,
        unused_channels=list(cfg.unused_channels),
        scoring_channels=list(cfg.scoring_channels),
    )
    # 额外：前 90s（模拟一场 v4_session）是否能 PASS
    head_n = int(round(90.0 * FS))
    head = X_raw[: min(head_n, len(X_raw))]
    mon2 = V4QualityMonitor(cfg)
    hist2 = []
    for start in range(0, head.shape[0] - win_n + 1, hop_n):
        diag = diagnose_eeg_window(head[start : start + win_n], sq, channel_names=names)
        mon2.update(diag, elapsed_s=start / FS)
        hist2.append(diag)
    head_summary = summarize_v4_session(
        [{**d, "pass_streak": 0} for d in hist2],
        duration_s=head.shape[0] / FS,
        pass_streak_required=cfg.pass_streak_required,
        achieved_stable=mon2.achieved_stable,
        time_to_stable_s=mon2.time_to_stable_s,
        channel_names=names,
        unused_channels=list(cfg.unused_channels),
        scoring_channels=list(cfg.scoring_channels),
    )
    return {
        "full_session": summary,
        "first_90s": {
            "achieved_stable": mon2.achieved_stable,
            "time_to_stable_s": mon2.time_to_stable_s,
            "ok_rate": head_summary.get("ok_rate"),
            "n_windows": head_summary.get("n_windows"),
            "top_problems": head_summary.get("top_problems"),
        },
        "verdict": "PASS" if mon2.achieved_stable else "FAIL",
    }


def _cut_3s_windows(x_filt: np.ndarray, t_lsl: np.ndarray, t_a: float, t_b: float) -> List[np.ndarray]:
    """返回 (8,750) z-scored 窗列表。"""
    i0 = int(np.searchsorted(t_lsl, t_a))
    i1 = int(np.searchsorted(t_lsl, t_b))
    dur = (i1 - i0) / FS
    if dur < WIN + T0_MIN - 1e-6:
        return []
    outs = []
    t0 = T0_MIN
    while t0 + WIN <= dur + 1e-9:
        a = i0 + int(round(t0 * FS))
        b = i0 + int(round((t0 + WIN) * FS))
        outs.append(trial_zscore(x_filt[a:b]).T.astype(np.float32))
        t0 = round(t0 + HOP, 3)
    return outs


def _raw_3s_windows(X_raw: np.ndarray, t_lsl: np.ndarray, t_a: float, t_b: float) -> List[np.ndarray]:
    """原始 µV (T,8) 3s 窗，供信号质量。"""
    i0 = int(np.searchsorted(t_lsl, t_a))
    i1 = int(np.searchsorted(t_lsl, t_b))
    dur = (i1 - i0) / FS
    if dur < WIN + T0_MIN - 1e-6:
        return []
    outs = []
    t0 = T0_MIN
    while t0 + WIN <= dur + 1e-9:
        a = i0 + int(round(t0 * FS))
        b = i0 + int(round((t0 + WIN) * FS))
        outs.append(X_raw[a:b])
        t0 = round(t0 + HOP, 3)
    return outs


def _laterality(mi_seg: np.ndarray, rest_seg: np.ndarray, label: int, standards: dict) -> Tuple[Optional[float], Optional[float]]:
    """返回 (mu_erd_contra, laterality_pp)；数据不足则 None。"""
    mi_wins = segment_to_hop_windows(mi_seg, FS)
    rest_wins = segment_to_hop_windows(rest_seg, FS)
    if not mi_wins or not rest_wins:
        return None, None
    mu_mi, _, _ = zip(*[bandpowers_fft(w, FS, standards) for w in mi_wins])
    mu_rest, _, _ = zip(*[bandpowers_fft(w, FS, standards) for w in rest_wins])
    mu_mi = np.mean(np.stack(mu_mi), 0)
    mu_rest = np.mean(np.stack(mu_rest), 0)
    erd_c3 = erd(float(mu_mi[IX["C3"]]), float(mu_rest[IX["C3"]]))
    erd_c4 = erd(float(mu_mi[IX["C4"]]), float(mu_rest[IX["C4"]]))
    if label == 1:  # Left → contra C4
        contra, ipsi = erd_c4, erd_c3
    else:
        contra, ipsi = erd_c3, erd_c4
    return contra, ipsi - contra


def load_registry(task_ckpt: Path, three_ckpt: Path):
    from adapt_engine.registry import ModelRegistry

    return ModelRegistry(str(task_ckpt), str(three_ckpt))


def majority_acc(preds: List[int], label: int) -> Optional[bool]:
    if not preds:
        return None
    c = Counter(preds)
    maj, _ = c.most_common(1)[0]
    return maj == label


def run_v3_v2(
    t_lsl: np.ndarray,
    X_raw: np.ndarray,
    trials: List[dict],
    *,
    registry,
    phase_filter: Optional[set] = None,
) -> dict:
    from adapt_engine.readout import serial_gating

    x_filt = notch_and_bandpass(car_reference(X_raw), FS, l_freq=8.0, h_freq=30.0)
    sq = SignalQualityConfig()  # v3/v2 默认（min_active=3）
    standards = {"mu_hz": (8.0, 13.0), "beta_l_hz": (13.0, 20.0), "beta_h_hz": (20.0, 30.0)}
    pre = OnlinePreprocessor()

    rows = []
    phase_filter = phase_filter or {"acquire"}

    for r in trials:
        if r.get("rejected") == "1":
            continue
        phase = r.get("phase") or ""
        if phase not in phase_filter:
            continue
        lab = int(r["label"])
        if lab not in (0, 1, 2):
            continue
        if not r.get("t_mi_start") or not r.get("t_mi_end"):
            continue
        t_mi0, t_mi1 = float(r["t_mi_start"]), float(r["t_mi_end"])
        t_rest0 = float(r["t_rest_start"]) if r.get("t_rest_start") else None
        t_rest1 = float(r["t_rest_end"]) if r.get("t_rest_end") else None
        mi_dur = t_mi1 - t_mi0

        # —— v3：primary judge ≈ mi_start + 4s（若 MI 更短则用末档）——
        primary_t = min(4.0, max(0.6, mi_dur - 0.05))
        t_end = t_mi0 + primary_t
        i_end = int(np.searchsorted(t_lsl, t_end))
        i0 = i_end - N_TIMES
        v3_signal_bad = True
        v3_pred = None
        v3_gated = None
        v3_p3 = None
        if i0 >= 0 and i_end <= len(X_raw):
            raw_win = X_raw[i0:i_end]
            qa = assess_eeg_window(raw_win, sq)
            if qa["ok"]:
                v3_signal_bad = False
                tail_n = int(12.0 * FS)
                tail = X_raw[max(0, i_end - tail_n) : i_end]
                window = pre.process(tail)
                heads = registry.forward_heads(window)
                out = serial_gating(heads["p_task"], heads["p_three"], task_p_on=0.6)
                v3_pred = int(out["pred"])
                v3_gated = bool(out.get("gated", False))
                v3_p3 = [float(x) for x in np.asarray(heads["p_three"]).ravel()]
            else:
                v3_signal_bad = True

        mu_c, lat = (None, None)
        if lab in (1, 2) and t_rest0 is not None and t_rest1 is not None:
            mi_seg = slice_tc(t_lsl, x_filt, t_mi0, t_mi1)
            rest_seg = slice_tc(t_lsl, x_filt, t_rest0, t_rest1)
            mu_c, lat = _laterality(mi_seg, rest_seg, lab, standards)

        # —— v2 Acc_paper：MI 段 3s/hop100 多数表决 ——
        zs_wins = _cut_3s_windows(x_filt, t_lsl, t_mi0, t_mi1) if lab in (1, 2) else []
        raw_wins = _raw_3s_windows(X_raw, t_lsl, t_mi0, t_mi1) if lab in (1, 2) else []
        keep_preds = []
        keep_h1 = []
        n_bad = 0
        for zw, rw in zip(zs_wins, raw_wins):
            qa = assess_eeg_window(rw, sq)
            if not qa["ok"]:
                n_bad += 1
                continue
            heads = registry.forward_heads(zw)
            out = serial_gating(heads["p_task"], heads["p_three"], task_p_on=0.6)
            pred = int(out["pred"])
            keep_preds.append(pred)
            # H1 质量门：仅 MI；用试次级 laterality/ERD 近似（与窗级 H1 同阈值）
            if mu_c is not None and lat is not None and mu_c <= -15 and lat >= 8:
                keep_h1.append(pred)

        acc_h0 = majority_acc(keep_preds, lab) if lab in (1, 2) else None
        acc_h1 = majority_acc(keep_h1, lab) if lab in (1, 2) else None
        abstain_h1 = lab in (1, 2) and not keep_h1

        rows.append(
            {
                "trial_id": int(r["trial_id"]),
                "phase": phase,
                "label": lab,
                "mi_dur": round(mi_dur, 3),
                "v3_signal_bad": v3_signal_bad,
                "v3_pred": v3_pred,
                "v3_gated": v3_gated,
                "v3_correct": (v3_pred == lab) if (v3_pred is not None and lab in (1, 2)) else None,
                "mu_erd_contra": None if mu_c is None else round(mu_c, 2),
                "laterality_pp": None if lat is None else round(lat, 2),
                "h1_pass": bool(mu_c is not None and lat is not None and mu_c <= -15 and lat >= 8),
                "n_windows": len(zs_wins),
                "n_signal_bad_windows": n_bad,
                "acc_paper_h0": acc_h0,
                "acc_paper_h1": acc_h1,
                "abstain_h1": abstain_h1,
                "p_three": v3_p3,
            }
        )

    mi = [x for x in rows if x["label"] in (1, 2)]
    rest = [x for x in rows if x["label"] == 0]

    def _rate(xs, key):
        vals = [x[key] for x in xs if x[key] is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    v3_ok = [x for x in mi if not x["v3_signal_bad"] and x["v3_pred"] is not None]
    hat = summarize_baseline_hat_check(
        X_raw[: int(60 * FS)] if len(X_raw) >= int(60 * FS) else X_raw,
        fs=FS,
    )

    return {
        "n_trials": len(rows),
        "n_mi": len(mi),
        "n_rest": len(rest),
        "mi_dur_mean": float(np.mean([x["mi_dur"] for x in mi])) if mi else None,
        "v3_hat_check": hat,
        "v3": {
            "n_scored": len(v3_ok),
            "n_signal_bad": sum(1 for x in mi if x["v3_signal_bad"]),
            "acc_mi": _rate(v3_ok, "v3_correct"),
            "pred_hist": dict(Counter(x["v3_pred"] for x in v3_ok)),
            "note": "primary ≈ mi_start+min(4s, mi_dur)；旧会话 MI≈4s，现行协议 imagine_s=6s",
        },
        "v2_acc_paper": {
            "h0_acc": _rate(mi, "acc_paper_h0"),
            "h1_acc": _rate([x for x in mi if not x["abstain_h1"]], "acc_paper_h1"),
            "h1_abstain_rate": float(np.mean([x["abstain_h1"] for x in mi])) if mi else None,
            "h1_pass_rate": float(np.mean([x["h1_pass"] for x in mi])) if mi else None,
            "mean_windows_per_mi": float(np.mean([x["n_windows"] for x in mi])) if mi else None,
            "signal_bad_window_frac": (
                float(np.sum([x["n_signal_bad_windows"] for x in mi]) / max(1, np.sum([x["n_windows"] for x in mi])))
                if mi
                else None
            ),
            "note": "Acc_paper = 试次内 3s/hop100 窗多数表决；H1=试次级 ERD≤-15 & lat≥8",
        },
        "features": {
            "mu_erd_contra_mean": float(np.mean([x["mu_erd_contra"] for x in mi if x["mu_erd_contra"] is not None]))
            if any(x["mu_erd_contra"] is not None for x in mi)
            else None,
            "laterality_mean": float(np.mean([x["laterality_pp"] for x in mi if x["laterality_pp"] is not None]))
            if any(x["laterality_pp"] is not None for x in mi)
            else None,
            "lat_ge8_rate": float(np.mean([1 if (x["laterality_pp"] or -999) >= 8 else 0 for x in mi])) if mi else None,
            "mu_le_m15_rate": float(np.mean([1 if (x["mu_erd_contra"] or 999) <= -15 else 0 for x in mi])) if mi else None,
        },
        "trials": rows,
    }


def epoch_compat_note(epoch_dir: Optional[Path]) -> dict:
    if epoch_dir is None or not (epoch_dir / "meta.json").is_file():
        return {}
    meta = json.loads((epoch_dir / "meta.json").read_text(encoding="utf-8"))
    shape = (meta.get("summary") or {}).get("X_shape") or meta.get("X_shape")
    win = meta.get("win_sec")
    n_times = meta.get("n_times")
    ok_3s = bool(n_times == 750 or (win is not None and abs(float(win) - 3.0) < 1e-6))
    return {
        "epoch_dir": str(epoch_dir),
        "X_shape": shape,
        "win_sec": win,
        "n_times": n_times,
        "compatible_with_v2v3_3s_model": ok_3s,
        "action": "已改从 continuous EEG 重切 3s 窗" if not ok_3s else "可直接用 epochs",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline replay v4/v3/v2 on game sessions")
    ap.add_argument("inputs", nargs="+", type=Path, help="epochs 或 sessions 目录")
    ap.add_argument("-o", "--out-dir", type=Path, default=None)
    ap.add_argument("--phase", default="acquire", help="逗号分隔 phase 过滤，默认 acquire")
    args = ap.parse_args()

    task_ckpt = _REPO / "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/run_20260822_094942/task/fold0/best_task.pt"
    three_ckpt = _REPO / "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/run_20260822_094942/three/fold0/best_three.pt"
    if not task_ckpt.is_file() or not three_ckpt.is_file():
        raise SystemExit(f"缺少 S3 权重:\n  {task_ckpt}\n  {three_ckpt}")

    print("加载 ModelRegistry …")
    registry = load_registry(task_ckpt, three_ckpt)
    v4_cfg = V4Config.load_yaml()
    phases = {p.strip() for p in args.phase.split(",") if p.strip()}
    out_dir = args.out_dir or (_ROOT / "data" / "offline_replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []
    for inp in args.inputs:
        epoch_dir = inp.resolve() if (inp / "X.npy").is_file() else None
        session_dir = resolve_session(inp)
        print(f"\n======== {inp.name} → {session_dir.name} ========")
        print(json.dumps(epoch_compat_note(epoch_dir), ensure_ascii=False))

        t_lsl, X_raw = load_eeg(session_dir)
        trials = load_trial_table(session_dir)
        print(f"EEG {X_raw.shape} · trials {len(trials)} · phases filter={sorted(phases)}")

        print("… v4 帽检回放")
        v4 = run_v4(t_lsl, X_raw, cfg=v4_cfg)
        print(f"  v4 first_90s: {v4['verdict']}  stable={v4['first_90s']['achieved_stable']}  ok_rate={v4['first_90s'].get('ok_rate')}")

        print("… v3/v2 零样本推理")
        v32 = run_v3_v2(t_lsl, X_raw, trials, registry=registry, phase_filter=phases)
        print(
            f"  v3 MI acc={v32['v3']['acc_mi']}  signal_bad={v32['v3']['n_signal_bad']}/{v32['n_mi']}  "
            f"hat={v32['v3_hat_check'].get('verdict') or v32['v3_hat_check'].get('ok')}"
        )
        print(
            f"  v2 Acc_paper H0={v32['v2_acc_paper']['h0_acc']}  "
            f"H1={v32['v2_acc_paper']['h1_acc']}  abstain={v32['v2_acc_paper']['h1_abstain_rate']}"
        )

        report = {
            "input": str(inp.resolve()),
            "session_dir": str(session_dir),
            "epoch_compat": epoch_compat_note(epoch_dir),
            "v4": v4,
            "v3_v2": {k: v for k, v in v32.items() if k != "trials"},
            "trial_rows": v32["trials"],
            "weights": {"task": str(task_ckpt), "three": str(three_ckpt)},
        }
        out_path = out_dir / f"{session_dir.name}_pipeline_replay.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {out_path}")
        all_reports.append(report)

    # 控制台汇总表
    print("\n======== 汇总 ========")
    print(f"{'session':<36} {'v4_90s':<8} {'v3_acc':<8} {'v3_bad':<8} {'v2_H0':<8} {'v2_H1':<8} {'H1_abs':<8}")
    for r in all_reports:
        name = Path(r["session_dir"]).name
        v4v = r["v4"]["verdict"]
        v3a = r["v3_v2"]["v3"]["acc_mi"]
        v3b = r["v3_v2"]["v3"]["n_signal_bad"]
        h0 = r["v3_v2"]["v2_acc_paper"]["h0_acc"]
        h1 = r["v3_v2"]["v2_acc_paper"]["h1_acc"]
        ab = r["v3_v2"]["v2_acc_paper"]["h1_abstain_rate"]
        def fmt(x):
            return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        print(f"{name:<36} {v4v:<8} {fmt(v3a):<8} {v3b:<8} {fmt(h0):<8} {fmt(h1):<8} {fmt(ab):<8}")


if __name__ == "__main__":
    main()
