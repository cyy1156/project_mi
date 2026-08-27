#!/usr/bin/env python3
"""Stieger 伪在线流 → experiment_game v4 / v3 / v2 关键栈离线回放。

数据源：`code/preprocess_lab/out/stieger_3s_hop100`（3s/hop100 窗，非连续 LSL）。
- v2 Acc_paper：试次级 3s 窗多数表决 + H1 门控（X_noz ERD/laterality）
- v3：每 MI 试次取末窗作 primary judge（Stieger 已滤波窗，不做原始 µV 信号 QC）
- v4：逐窗 QC（**代理**：X_noz=CAR+陷波+8–30，非原始 µV；无 .mat 时无法真连续流）

用法::

    python -m experiment_game.tools.replay_stieger_pipeline --subjects S1,S2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
_STIEGER_STEP = _REPO / "code" / "train_lab" / "src" / "step" / "stieger_pseudo_online_3s_hop100"
_GAME_STEP = _REPO / "code" / "train_lab" / "src" / "step" / "game_pseudo_online_hop100"
for p in (str(_REPO), str(_REPO / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiment_game.experiment.signal_quality import (  # noqa: E402
    SignalQualityConfig,
    assess_eeg_window,
    diagnose_eeg_window,
)
from experiment_game.experiment.v4_config import V4Config  # noqa: E402

FS = 250.0
N_TIMES = 750


def _import_stieger():
    saved = list(sys.path)
    sys.path.insert(0, str(_STIEGER_STEP))
    try:
        from data import iter_subject_streams  # type: ignore
        from config import OPENBMI_CHANS  # type: ignore
        return iter_subject_streams, list(OPENBMI_CHANS)
    finally:
        sys.path[:] = saved


def _import_gate():
    saved = list(sys.path)
    sys.path.insert(0, str(_GAME_STEP))
    try:
        from online_gate import build_gate_keeps  # type: ignore
        return build_gate_keeps
    finally:
        sys.path[:] = saved


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _win_ct(x: np.ndarray) -> np.ndarray:
    """(1,8,T) or (8,T) → (T,8) float64."""
    w = np.asarray(x)
    if w.ndim == 3:
        w = w[0]
    if w.shape[0] == 8:
        w = w.T
    return np.asarray(w, dtype=np.float64)


def _win_model(x: np.ndarray) -> np.ndarray:
    """(1,8,T) or (8,T) → (8,T) float32."""
    w = np.asarray(x)
    if w.ndim == 3:
        w = w[0]
    if w.shape[0] != 8:
        w = w.T
    return np.ascontiguousarray(w, dtype=np.float32)


def load_registry(task_ckpt: Path, three_ckpt: Path):
    from adapt_engine.registry import ModelRegistry

    return ModelRegistry(str(task_ckpt), str(three_ckpt))


def acc_paper_segment(preds: List[int], label: int) -> Optional[bool]:
    """train_lab Acc_paper：窗级 pred==label 比例 > 0.5。"""
    if not preds:
        return None
    return float(np.mean([p == label for p in preds])) > 0.5


def _import_metrics():
    saved = list(sys.path)
    sys.path.insert(0, str(_GAME_STEP.parent))
    sys.path.insert(0, str(_GAME_STEP))
    try:
        from eval_metrics import aggregate_windows_to_segments  # type: ignore
        from gated_segment_metrics import aggregate_windows_to_segments_gated  # type: ignore
        return aggregate_windows_to_segments, aggregate_windows_to_segments_gated
    finally:
        sys.path[:] = saved


def run_s07_reference(stream, three_ckpt: Path, build_gate_keeps, ch_names: List[str]) -> dict:
    """与 S07-01/03 对齐：three 头 argmax + Acc_paper 段级聚合。"""
    import torch
    from braindecode.models import ShallowFBCSPNet

    agg_fn, agg_gated_fn = _import_metrics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(three_ckpt), map_location="cpu", weights_only=False)
    n_out = int(ckpt.get("n_outputs", 3))
    model = ShallowFBCSPNet(n_chans=8, n_outputs=n_out, n_times=N_TIMES, drop_prob=0.5)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    X = stream.X
    if X.ndim == 4 and X.shape[1] == 1:
        Xb = np.asarray(X[:, 0], dtype=np.float32)
    else:
        Xb = np.asarray(X, dtype=np.float32)
    preds: list[np.ndarray] = []
    bs = 128
    with torch.no_grad():
        for i in range(0, len(Xb), bs):
            batch = torch.from_numpy(Xb[i : i + bs]).to(device)
            logits = model(batch)
            if logits.ndim > 2:
                logits = logits.reshape(logits.shape[0], -1)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)

    h0 = agg_fn(stream.y_three, y_pred, stream.seg_keys, n_classes=3)
    keeps_h1 = build_gate_keeps(
        stream.X_noz,
        stream.y_three,
        stream.cue_ids,
        stream.segs,
        ch_names=ch_names,
        modes=("H1",),
    )["H1"]
    h1 = agg_gated_fn(stream.y_three, y_pred, stream.seg_keys, keeps_h1, n_classes=3)
    return {
        "three_head_argmax": True,
        "h0_acc_paper": float(h0["segment_metrics"]["acc_paper"]),
        "h1_acc_paper": float(h1["segment_metrics"]["acc_paper"]),
        "h1_abstain_rate": float(h1["segment_metrics"]["abstain_rate"]),
        "note": "对齐 S07：无 serial_gating；fold0 three ckpt",
    }


def run_v4_proxy(X_noz: np.ndarray, *, cfg: V4Config) -> dict:
    """逐 3s 窗 QC；输入为预处理无 z-score 窗（非真原始 µV）。"""
    sq = cfg.signal_quality_config()
    names = list(cfg.channel_labels)
    history = []
    for i in range(len(X_noz)):
        win = _win_ct(X_noz[i])
        diag = diagnose_eeg_window(win, sq, channel_names=names)
        history.append(diag)
    ok = sum(1 for d in history if d.get("window_ok"))
    n = len(history)
    problems = Counter()
    for d in history:
        if not d.get("window_ok"):
            problems[str(d.get("window_reason") or "unknown")] += 1
    mi_only = None
    return {
        "mode": "per_window_proxy",
        "note": "X_noz=CAR+陷波+8–30µV；非 LSL 原始流，阈值仅作参考",
        "n_windows": n,
        "ok_rate": float(ok / n) if n else None,
        "top_problems": dict(problems.most_common(5)),
        "verdict": "PASS" if n and ok / n >= 0.85 else "FAIL",
        "mi_only": mi_only,
    }


def run_v3_v2(stream, *, registry, ch_names: List[str], build_gate_keeps, apply_signal_qc: bool) -> dict:
    from adapt_engine.readout import serial_gating

    sq_v3 = SignalQualityConfig() if apply_signal_qc else None
    X = stream.X
    X_noz = stream.X_noz
    y_three = stream.y_three
    segs = stream.segs
    seg_keys = stream.seg_keys
    trial_ids = stream.trial_ids
    cue_ids = stream.cue_ids

    if X_noz is None:
        raise RuntimeError(f"{stream.subject_id}: 缺少 X_noz")

    keeps_h1 = build_gate_keeps(
        X_noz,
        y_three,
        cue_ids,
        segs,
        ch_names=ch_names,
        modes=("H0", "H1"),
    )["H1"]

    # 按 MI 试次聚合窗
    by_seg: Dict[str, List[int]] = defaultdict(list)
    for i, sk in enumerate(seg_keys):
        if segs[i] == "mi":
            by_seg[str(sk)].append(i)

    rows = []
    for sk, idxs in sorted(by_seg.items()):
        lab = int(y_three[idxs[0]])
        if lab not in (1, 2):
            continue
        tid = int(sk.split(":")[0])

        # v3 primary ≈ 试次末窗（反馈段内最晚 3s 窗）
        last_i = idxs[-1]
        v3_signal_bad = False
        if sq_v3 is not None:
            raw_win = _win_ct(X_noz[last_i])
            v3_signal_bad = not assess_eeg_window(raw_win, sq_v3)["ok"]
        v3_pred = None
        v3_gated = None
        if not v3_signal_bad:
            heads = registry.forward_heads(_win_model(X[last_i]))
            out = serial_gating(heads["p_task"], heads["p_three"], task_p_on=0.6)
            v3_pred = int(out["pred"])
            v3_gated = bool(out.get("gated", False))

        keep_preds = []
        keep_h1 = []
        n_bad = 0
        for i in idxs:
            if sq_v3 is not None:
                rw = _win_ct(X_noz[i])
                if not assess_eeg_window(rw, sq_v3)["ok"]:
                    n_bad += 1
                    continue
            heads = registry.forward_heads(_win_model(X[i]))
            out = serial_gating(heads["p_task"], heads["p_three"], task_p_on=0.6)
            pred = int(out["pred"])
            keep_preds.append(pred)
            if keeps_h1[i]:
                keep_h1.append(pred)

        rows.append(
            {
                "trial_id": tid,
                "seg_key": sk,
                "label": lab,
                "n_windows": len(idxs),
                "n_signal_bad_windows": n_bad,
                "v3_signal_bad": v3_signal_bad,
                "v3_pred": v3_pred,
                "v3_gated": v3_gated,
                "v3_correct": (v3_pred == lab) if v3_pred is not None else None,
                "acc_paper_h0": acc_paper_segment(keep_preds, lab),
                "acc_paper_h1": acc_paper_segment(keep_h1, lab),
                "abstain_h1": not keep_h1,
            }
        )

    def _rate(xs, key):
        vals = [x[key] for x in xs if x[key] is not None]
        return float(np.mean(vals)) if vals else None

    v3_ok = [x for x in rows if not x["v3_signal_bad"] and x["v3_pred"] is not None]
    return {
        "n_mi_trials": len(rows),
        "v3": {
            "n_scored": len(v3_ok),
            "n_signal_bad": sum(1 for x in rows if x["v3_signal_bad"]),
            "acc_mi": _rate(v3_ok, "v3_correct"),
            "note": "primary=MI 试次末 3s 窗；真在线为 tail12s+OnlinePreprocessor",
        },
        "v2_acc_paper": {
            "h0_acc": _rate(rows, "acc_paper_h0"),
            "h1_acc": _rate([x for x in rows if not x["abstain_h1"]], "acc_paper_h1"),
            "h1_abstain_rate": float(np.mean([x["abstain_h1"] for x in rows])) if rows else None,
            "mean_windows_per_mi": float(np.mean([x["n_windows"] for x in rows])) if rows else None,
            "signal_bad_window_frac": (
                float(np.sum([x["n_signal_bad_windows"] for x in rows]) / max(1, np.sum([x["n_windows"] for x in rows])))
                if rows
                else None
            ),
            "note": "experiment_game v2：serial_gating + Acc_paper(窗正确率>0.5)；H1=online_gate",
        },
        "trials": rows,
    }


def main() -> None:
    iter_subject_streams, ch_names = _import_stieger()
    build_gate_keeps = _import_gate()

    ap = argparse.ArgumentParser(description="Stieger → v4/v3/v2 offline replay")
    ap.add_argument("--subjects", default="S1,S2", help="逗号分隔被试 id")
    ap.add_argument("-o", "--out-dir", type=Path, default=None)
    ap.add_argument("--run-stamp", default="run_20260821_190504", help="OpenBMI S3 权重 run")
    ap.add_argument("--fold", type=int, default=0, help="fold 编号")
    ap.add_argument(
        "--signal-qc",
        action="store_true",
        help="对 X_noz 做原始 µV 信号 QC（默认关：Stieger 已滤波，阈值不适用）",
    )
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    weight_root = (
        _REPO
        / "code/train_lab/out/5060_baseline_openbmi_3s_hop100_accpaper"
        / "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100"
        / args.run_stamp
    )
    fold = int(args.fold)
    task_ckpt = weight_root / f"task/fold{fold}/best_task.pt"
    three_ckpt = weight_root / f"three/fold{fold}/best_three.pt"
    if not task_ckpt.is_file() or not three_ckpt.is_file():
        raise SystemExit(f"缺少 S3 权重 fold{fold}:\n  {task_ckpt}\n  {three_ckpt}")

    print(f"加载 ModelRegistry ({args.run_stamp} fold{fold}) …")
    registry = load_registry(task_ckpt, three_ckpt)
    v4_cfg = V4Config.load_yaml()
    out_dir = args.out_dir or (_ROOT / "data" / "offline_replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 07 登记表参考（three H0）
    ref_07 = {"S1": 0.5299, "S2": 0.4143}

    all_reports: List[dict] = []
    for stream in iter_subject_streams(subjects=subjects):
        print(f"\n======== {stream.subject_id} n={len(stream.X)} ========")
        v4 = run_v4_proxy(stream.X_noz, cfg=v4_cfg)
        print(f"  v4 proxy ok_rate={v4['ok_rate']:.3f}  verdict={v4['verdict']}")

        v32 = run_v3_v2(
            stream,
            registry=registry,
            ch_names=ch_names,
            build_gate_keeps=build_gate_keeps,
            apply_signal_qc=bool(args.signal_qc),
        )
        h0 = v32["v2_acc_paper"]["h0_acc"]
        ref = ref_07.get(stream.subject_id)
        delta = f" (vs S07-01 {ref:.4f}, Δ{h0 - ref:+.4f})" if ref and h0 is not None else ""
        print(
            f"  v3 MI acc={v32['v3']['acc_mi']}  bad={v32['v3']['n_signal_bad']}/{v32['n_mi_trials']}"
        )
        print(
            f"  v2 H0={h0}{delta}  H1={v32['v2_acc_paper']['h1_acc']}  "
            f"abstain={v32['v2_acc_paper']['h1_abstain_rate']}"
        )

        s07 = run_s07_reference(stream, three_ckpt, build_gate_keeps, ch_names)
        print(f"  S07-ref H0={s07['h0_acc_paper']:.4f}  H1={s07['h1_acc_paper']:.4f}")

        report = {
            "subject_id": stream.subject_id,
            "meta": stream.meta,
            "v4": v4,
            "v3_v2": {k: v for k, v in v32.items() if k != "trials"},
            "s07_reference": s07,
            "ref_s07_01_three_h0": ref,
            "trial_rows": v32["trials"],
            "weights": {"task": str(task_ckpt), "three": str(three_ckpt)},
        }
        out_path = out_dir / f"stieger_{stream.subject_id}_pipeline_replay.json"
        out_path.write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {out_path}")
        all_reports.append(report)

    print("\n======== 汇总 ========")
    print(f"{'subject':<8} {'v4_ok%':<8} {'v3_acc':<8} {'v2_H0':<8} {'S07_H0':<8} {'S07ref':<8} {'v2_H1':<8}")
    for r in all_reports:
        sid = r["subject_id"]
        v4r = r["v4"]["ok_rate"]
        v3a = r["v3_v2"]["v3"]["acc_mi"]
        h0 = r["v3_v2"]["v2_acc_paper"]["h0_acc"]
        h1 = r["v3_v2"]["v2_acc_paper"]["h1_acc"]
        ref = r.get("ref_s07_01_three_h0")
        s07h0 = r.get("s07_reference", {}).get("h0_acc_paper")

        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"

        print(
            f"{sid:<8} {fmt(v4r):<8} {fmt(v3a):<8} {fmt(h0):<8} "
            f"{fmt(ref):<8} {fmt(s07h0):<8} {fmt(h1):<8}"
        )


if __name__ == "__main__":
    main()
