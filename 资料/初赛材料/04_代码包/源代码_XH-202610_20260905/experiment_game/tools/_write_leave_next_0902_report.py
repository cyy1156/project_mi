# -*- coding: utf-8 -*-
"""Write Leave-Next F5 detailed report for 0902 cohort (plan A)."""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SUBJECTS_ROOT = Path(__file__).resolve().parents[1] / "data" / "subjects"
OUT_ROOT = SUBJECTS_ROOT / "_analysis"
SUBJECTS = ["zyj0902", "djh0902", "zcy0902"]

NOTES = {
    "zyj0902": "w03 缺 eeg / record_excluded，爬坡跳过 w03；R5 持有 w07。当日操作台 FT 多为非 Leave-Next（heldout=[]）。",
    "djh0902": "除 w03 外多场 CZ 饱和/大 DC，ft_eligible=false；离线 Leave-Next 仍纳入做质量对照。",
    "zcy0902": "w04/w05 CZ 饱和告警仍纳入完整 R1–R5；对照在线窗级准确率。",
}


def latest_summary(sid: str) -> Optional[Path]:
    ft = SUBJECTS_ROOT / sid / "models" / "ft_runs"
    if not ft.is_dir():
        return None
    cands = sorted(
        ft.glob(f"*{sid}*leave_next_all4_f5_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def short_sid(s: str) -> str:
    for p in str(s).split("_"):
        pl = p.lower()
        if pl.startswith("ws") and pl[2:].isdigit():
            return pl
        if pl.startswith("w") and not pl.startswith("ws") and pl[1:].isdigit():
            return pl
        if "+" in pl:
            return pl
    return str(s)[-20:]


def train_tag(row: Dict[str, Any]) -> str:
    train = row.get("train") or []
    if train and isinstance(train[0], str) and train[0].startswith("w"):
        # already short keys sometimes
        keys = [short_sid(x) for x in train]
    else:
        keys = [short_sid(x) for x in train]
    hold = short_sid(str(row.get("heldout") or "?"))
    # prefer r_stage train_keys if present
    if row.get("train_keys"):
        keys = [str(x) for x in row["train_keys"]]
    if row.get("hold_key"):
        hold = str(row["hold_key"])
    return f"{'+'.join(keys)}→{hold}"


def f5_pack(pack: Optional[Dict[str, Any]]) -> str:
    if not pack:
        return "—"
    bl = pack.get("by_label") or {}
    left, right, rest = bl.get("Left") or {}, bl.get("Right") or {}, bl.get("Rest") or {}
    sc, sm = pack.get("score"), pack.get("score_max")
    scs = f"{float(sc):.1f}/{float(sm):.1f}" if sc is not None and sm is not None else "—"
    return (
        f"L {int(left.get('ok', 0))}/{int(left.get('n', 0))} · "
        f"R {int(right.get('ok', 0))}/{int(right.get('n', 0))} · "
        f"Rest {int(rest.get('ok', 0))}/{int(rest.get('n', 0))} · {scs}"
    )


def f5_score(pack: Optional[Dict[str, Any]]) -> Optional[float]:
    if not pack or pack.get("score") is None:
        return None
    return float(pack["score"])


def mu_sigma(vals: List[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"


def mu_sigma_f5(vals: List[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{statistics.mean(vals):.1f}±{statistics.stdev(vals):.1f}"


def online_table(sid: str) -> List[str]:
    idx_path = SUBJECTS_ROOT / sid / "index.json"
    if not idx_path.is_file():
        return ["- 无 index.json"]
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    lines = [
        "| 场 | primary(acc_argmax) | window(acc_window) | ft_eligible | electrode_ok | 备注 |",
        "|----|---------------------|------------------|-------------|--------------|------|",
    ]
    for sess in idx.get("sessions") or []:
        if sess.get("phase_mode") != "v3_session":
            continue
        warns = sess.get("electrode_warnings") or []
        note = "；".join(str(w) for w in warns) if warns else ""
        if sess.get("record_excluded"):
            note = (note + "；" if note else "") + "record_excluded"
        pa, wa = sess.get("primary_acc"), sess.get("window_acc")
        pa_s = f"{pa:.4f}" if isinstance(pa, (int, float)) else "—"
        wa_s = f"{wa:.4f}" if isinstance(wa, (int, float)) else "—"
        lines.append(
            f"| {sess.get('session_id')} | {pa_s} | {wa_s} | "
            f"{sess.get('ft_eligible')} | {sess.get('electrode_ok')} | {note or '—'} |"
        )
    return lines


def main() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines: List[str] = [
        f"# Leave-Next F5 详细报告 · 0902 被试队列 · {stamp}",
        "",
        "口径：[`docs/统计口径方案A_20260831.md`](../../docs/统计口径方案A_20260831.md)",
        "",
        "- FT：E1f 四成员融合 `ft_scope=all4`",
        "- 窗级：`heldout_acc` = **smooth**（展示）；`heldout_acc_raw` = **raw**（门控）",
        "- F5：试次级因果平滑 lookback=2 + 多数票；MI +1 / Rest +0.5（满分常 45）",
        "- 未自动晋升 `models/current`（仅分析）",
        "- 当日操作台增量 FT（`heldout=[]`）**不是**标准 Leave-Next，本报告以离线 ramp summary 为准",
        "",
        "## 1. 实验口径与爬坡",
        "",
        "```",
        "标准（djh/zcy）：R1 w01→w02 … R5 …+w05→w06（R1–R3 replay on；R4–R5 off）",
        "zyj0902：跳过 w03；R5 …+w06→w07",
        "```",
        "",
        "## 2. 在线采集速览（方案 A 字段）",
        "",
    ]
    for sid in SUBJECTS:
        lines.append(f"### {sid}")
        lines.extend(online_table(sid))
        lines.append("")

    lines.extend(
        [
            "## 3. 跨被试总览（Leave-Next 末档 + μ±σ）",
            "",
            "| 被试 | PASS | smooth μ±σ | raw μ±σ | F5 μ±σ | 末档 smooth/raw | 末档 F5 FT |",
            "|------|------|------------|---------|--------|-----------------|------------|",
        ]
    )

    payloads: Dict[str, Dict[str, Any]] = {}
    paths: Dict[str, Path] = {}
    for sid in SUBJECTS:
        sp = latest_summary(sid)
        if sp is None:
            lines.append(f"| {sid} | — | — | — | — | — | — |")
            continue
        paths[sid] = sp
        payload = json.loads(sp.read_text(encoding="utf-8"))
        payloads[sid] = payload
        rows = payload.get("rows") or []
        n_pass = sum(1 for r in rows if r.get("release_pass"))
        sm = [
            float(r["heldout_acc"])
            for r in rows
            if isinstance(r.get("heldout_acc"), (int, float))
        ]
        raw = [
            float(r.get("heldout_acc_raw", r.get("heldout_acc")))
            for r in rows
            if isinstance(r.get("heldout_acc_raw", r.get("heldout_acc")), (int, float))
        ]
        f5s = [v for r in rows if (v := f5_score(r.get("f5_ft"))) is not None]
        last = rows[-1] if rows else {}
        lsm = last.get("heldout_acc")
        lraw = last.get("heldout_acc_raw", lsm)
        lf5 = f5_pack(last.get("f5_ft")) if last else "—"
        lines.append(
            f"| **{sid}** | **{n_pass}/{len(rows)}** | {mu_sigma(sm)} | {mu_sigma(raw)} | "
            f"{mu_sigma_f5(f5s)} | "
            f"{(f'{lsm:.3f}/{lraw:.3f}' if isinstance(lsm, (int, float)) else '—')} | {lf5} |"
        )

    lines.extend(["", "## 4. 分被试详细结果", ""])
    for sid in SUBJECTS:
        lines.append(f"### {sid}")
        lines.append(f"- 备注：{NOTES.get(sid, '—')}")
        sp = paths.get(sid)
        if sp is None:
            lines.append("- **无 Leave-Next summary**（尚未跑完或失败）")
            lines.append("")
            continue
        payload = payloads[sid]
        rows = payload.get("rows") or []
        lines.append(f"- summary：`{sp}`")
        lines.append(f"- 设备：`{payload.get('device')}` · lookback={payload.get('causal_lookback')}")
        n_pass = sum(1 for r in rows if r.get("release_pass"))
        lines.append(f"- gate PASS：**{n_pass}/{len(rows)}**")
        lines.append("")
        lines.append("| R | train→hold | rep | smooth | raw | gate |")
        lines.append("|---|------------|-----|--------|-----|------|")
        for row in rows:
            r = row.get("r_stage") or "?"
            rep = "on" if row.get("use_replay") else "off"
            sm = row.get("heldout_acc")
            raw = row.get("heldout_acc_raw", sm)
            gate = "PASS" if row.get("release_pass") else "FAIL"
            sm_s = f"{sm:.3f}" if isinstance(sm, (int, float)) else "—"
            raw_s = f"{raw:.3f}" if isinstance(raw, (int, float)) else "—"
            # train tag
            tk = row.get("train_keys") or row.get("train") or []
            hk = row.get("hold_key") or row.get("heldout") or "?"
            if isinstance(tk, list) and tk and not str(tk[0]).startswith("v3_"):
                tag = f"{'+'.join(str(x) for x in tk)}→{hk}"
            else:
                tag = train_tag(row)
            lines.append(f"| R{r} | {tag} | {rep} | {sm_s} | {raw_s} | {gate} |")
        lines.append("")
        lines.append("| R | model | F5 | Left | Right | Rest | pred |")
        lines.append("|---|-------|----|------|-------|------|------|")
        for row in rows:
            r = row.get("r_stage") or "?"
            pred = row.get("pred_labels") or {}
            if not pred:
                rg = row.get("release_gate") or {}
                pred = rg.get("pred_labels") or {}
            pred_s = ", ".join(f"{k}:{v}" for k, v in pred.items()) if pred else "—"
            for model_key, label in (("f5_ft", "**FT all4**"), ("f5_base_three", "base3")):
                pack = row.get(model_key) or {}
                bl = pack.get("by_label") or {}
                left, right, rest = bl.get("Left") or {}, bl.get("Right") or {}, bl.get("Rest") or {}
                sc, smx = pack.get("score"), pack.get("score_max")
                scs = f"{float(sc):.1f}/{float(smx):.1f}" if sc is not None and smx is not None else "—"
                lines.append(
                    f"| R{r} | {label} | {scs} | "
                    f"{int(left.get('ok', 0))}/{int(left.get('n', 0))} | "
                    f"{int(right.get('ok', 0))}/{int(right.get('n', 0))} | "
                    f"{int(rest.get('ok', 0))}/{int(rest.get('n', 0))} | {pred_s if model_key=='f5_ft' else ''} |"
                )
        lines.append("")

    lines.extend(
        [
            "## 5. 读表要点（初稿，跑完后请据实修订）",
            "",
            "1. 先看 **末档 F5** 与 **gate PASS 比例**，再看 smooth 爬坡是否单调。",
            "2. CZ 告警场次：若 max_class_frac 高 / Rest 塌缩，优先判为信号质量而非算法失败。",
            "3. 与历史强被试（syj0828 末档 F5≈41.5/45）对比时，勿混用旧 raw heldout 批次。",
            "",
        ]
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"leave_next_f5_cohort_0902_详细报告_{stamp}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
