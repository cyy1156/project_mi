"""M2 验收：用已录会话回放，比较在线/离线两条推理路径的一致性。

离线路径 = 整段连续 CAR+notch+8-30Hz → 按 cue 切 3s 窗 → z-score → 双头推理
在线路径 = 每判定点独立取 12s 尾段 → 同款滤波 → 末 750 点 → z-score → 双头推理
判据：argmax 不一致率 <1%（两头分别统计）+ max|Δz| 报告值。

用法：
  python experiment_game/experiment/m2_acceptance.py <session_dir> [--fold 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # D:/MI（experiment_game 包）
sys.path.insert(0, str(_HERE.parents[2] / "code"))              # adapt_engine
sys.path.insert(0, str(_HERE.parents[2] / "code" / "preprocess_lab"))

from experiment_game.experiment import inference_v2 as inf  # noqa: E402

# 2026-08-29 冻结：设备序 = 模型序 = channel_layout 全局统一序
RAW_COLS = ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"]
FROZEN = ["FC3", "C3", "CP3", "CZ", "CPZ", "FC4", "C4", "CP4"]
REORDER = [RAW_COLS.index(c.upper()) for c in FROZEN]  # 恒等 [0..7]


def load_session_eeg(session_dir: Path):
    """合并 by_phase/*/eeg.csv → (t_lsl(N,), X(N,8) 冻结序)。"""
    frames, times = [], []
    for d in sorted((session_dir / "by_phase").glob("*")):
        f = d / "eeg.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            rd = csv.reader(fh)
            header = next(rd)
            cols = [header.index(c) for c in RAW_COLS]
            rows = [r for r in rd if len(r) == len(header)]
        arr = np.asarray(rows, dtype=np.float64)
        times.append(arr[:, header.index("lsl_time")])
        frames.append(arr[:, cols])
    if not frames:
        raise RuntimeError(f"{session_dir} 无 eeg.csv")
    return np.concatenate(times), np.concatenate(frames)[:, REORDER]


def load_cues(session_dir: Path):
    with open(session_dir / "alignment" / "trial_table.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(int(r["trial_id"]), int(r["label"]), float(r["t_cue"])) for r in rows if r["t_cue"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tail-s", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    t_lsl, X_raw = load_session_eeg(session_dir)
    cues = load_cues(session_dir)
    print(f"会话 {session_dir.name}：{len(t_lsl)} 样本（{len(t_lsl)/250:.1f}s），{len(cues)} 个 cue")

    # —— 权重（S3 fold{N} 双头）——
    root = Path(r"D:/MI/code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper/"
                r"shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100/run_20260822_094942")
    from adapt_engine.registry import ModelRegistry

    reg = ModelRegistry(root / "task" / f"fold{args.fold}" / "best_task.pt",
                        root / "three" / f"fold{args.fold}" / "best_three.pt")

    # —— 逐 cue 比较两条路径 ——
    stats = {"max_dz": 0.0, "mean_dz": [], "task_dis": 0, "three_dis": 0, "n_win": 0,
             "p_diff_max": 0.0}
    per_trial = []
    for trial_id, label, cue_t in cues:
        i_cue = int(np.searchsorted(t_lsl, cue_t))
        if i_cue < int(6.0 * 250) or i_cue + int(6.0 * 250) > len(t_lsl):
            per_trial.append({"trial": trial_id, "skip": True})
            continue
        off = inf.offline_windows(X_raw, i_cue)
        on = inf.online_windows(X_raw, i_cue, tail_s=args.tail_s)
        dz = np.abs(off - on)
        stats["max_dz"] = max(stats["max_dz"], float(dz.max()))
        stats["mean_dz"].append(float(dz.mean()))
        t_agree = th_agree = 0
        for wo, wn in zip(off, on):
            po = reg.forward_heads(wo)
            pn = reg.forward_heads(wn)
            if int(np.argmax(po["p_task"])) == int(np.argmax(pn["p_task"])):
                t_agree += 1
            else:
                stats["task_dis"] += 1
            if int(np.argmax(po["p_three"])) == int(np.argmax(pn["p_three"])):
                th_agree += 1
            else:
                stats["three_dis"] += 1
            stats["p_diff_max"] = max(
                stats["p_diff_max"],
                float(np.abs(po["p_three"] - pn["p_three"]).max()),
            )
            stats["n_win"] += 1
        per_trial.append({"trial": trial_id, "label": label, "task_agree": t_agree,
                          "three_agree": th_agree, "max_dz": float(dz.max())})

    n = max(1, stats["n_win"])
    task_rate = 1 - stats["task_dis"] / n
    three_rate = 1 - stats["three_dis"] / n

    lines = [
        f"# M2 验收报告 · {session_dir.name}",
        "",
        f"- 数据：{len(t_lsl)} 样本 / {len(cues)} cue / 判定窗 {stats['n_win']} 个（t=3/4/5/6s）",
        f"- 权重：S3 fold{args.fold} 双头（`run_20260822_094942`）",
        f"- 通道重排：设备序 {RAW_COLS} → 冻结序 {FROZEN}",
        f"- 在线尾段：{args.tail_s}s",
        "",
        "## 结果",
        "",
        f"| 指标 | 值 | 判据 |",
        f"|------|-----|------|",
        f"| Task 头 argmax 一致率 | **{task_rate:.4f}** | ≥0.99 |",
        f"| Three 头 argmax 一致率 | **{three_rate:.4f}** | ≥0.99 |",
        f"| max\\|Δz\\|（z-score 后窗差） | {stats['max_dz']:.4f} | 报告值 |",
        f"| mean\\|Δz\\| | {np.mean(stats['mean_dz']):.5f} | 报告值 |",
        f"| max\\|Δp_three\\| | {stats['p_diff_max']:.4f} | 报告值 |",
        "",
        "## 逐试次",
        "",
        "| trial | label | task一致 | three一致 | maxΔz |",
        "|-------|-------|---------|-----------|-------|",
    ]
    for r in per_trial:
        if r.get("skip"):
            lines.append(f"| {r['trial']} | — | 跳过（前后数据不足） | | |")
        else:
            lines.append(f"| {r['trial']} | {r['label']} | {r['task_agree']}/4 | {r['three_agree']}/4 | {r['max_dz']:.4f} |")
    verdict = "✅ 通过" if (task_rate >= 0.99 and three_rate >= 0.99) else "❌ 未过"
    lines += ["", f"**判定：{verdict}**（argmax 一致率两头均 ≥99%）"]

    out = Path(args.out) if args.out else _HERE.parents[1] / "docs" / f"m2_acceptance_{session_dir.name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:18]))
    print(f"\n报告已写入 {out}")
    return 0 if verdict.startswith("✅") else 1


if __name__ == "__main__":
    raise SystemExit(main())
