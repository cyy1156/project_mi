# -*- coding: utf-8 -*-
"""Generate detailed Leave-Next F5 cohort markdown report (0828/0830)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SUBJECTS_ROOT = Path(__file__).resolve().parents[1] / "data" / "subjects"
OUT = SUBJECTS_ROOT / "_analysis" / "leave_next_f5_cohort_0828_0830_详细报告_20260830.md"
SUBJECTS = ["syj0828", "xjh0828", "cyy0830", "fnz0830", "wzr0830", "xj0830"]
STAMP_HINT = {
    "syj0828": "20260830_230014",
    "xjh0828": "20260830_230057",
    "cyy0830": "20260830_230116",
    "fnz0830": "20260831_103500",
    "wzr0830": "20260831_103536",
    "xj0830": "20260831_103607",
}
NOTES = {
    "syj0828": "全档 PASS；末档持有集极强，F5 显著高于底座。",
    "xjh0828": "缺 v3 ws01（仅 v4 首测）；ws02–ws07 共 6 场 v3，Leave-Next 5 档；ws07 于 2026-08-30 重测覆盖断流版。",
    "cyy0830": "w02/w03 电极质量差；R3–R4 预测塌成几乎只出 Left。",
    "fnz0830": "legacy 切窗 Rest 窗偏多；末档易偏 Rest，MI 弱。",
    "wzr0830": "0830 中相对最好；R4–R5 过门控。",
    "xj0830": "Rest 预测偏多、MI 弱；全档 FAIL。",
}

def latest_summary(sid: str) -> Path:
    hint = STAMP_HINT.get(sid, "")
    ft = SUBJECTS_ROOT / sid / "models" / "ft_runs"
    p = ft / f"{hint}_{sid}_e1f_task_leave_next_f5_summary.json"
    if p.is_file():
        return p
    cands = sorted(
        ft.glob(f"*{sid}*leave_next*f5_summary.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(sid)
    return cands[0]

def short_sid(s: str) -> str:
    for p in str(s).split("_"):
        if p.startswith("ws") and p[2:].isdigit():
            return p
        if p.startswith("w") and len(p) >= 3 and p[1:].isdigit():
            return p
    return s[-24:]

def train_tag(row: Dict[str, Any]) -> str:
    train = [short_sid(x) for x in (row.get("train") or [])]
    hold = short_sid(str(row.get("heldout") or "?"))
    return f"{'+'.join(train)}→{hold}"

def f5_mi_rest(pack: Optional[Dict[str, Any]]) -> str:
    if not pack:
        return "—"
    bl = pack.get("by_label") or {}
    left, right, rest = bl.get("Left") or {}, bl.get("Right") or {}, bl.get("Rest") or {}
    n_mi = int(left.get("n") or 0) + int(right.get("n") or 0)
    ok_mi = int(left.get("ok") or 0) + int(right.get("ok") or 0)
    n_r = int(rest.get("n") or 0)
    ok_r = int(rest.get("ok") or 0)
    sc, sm = pack.get("score"), pack.get("score_max")
    if sc is not None and sm is not None:
        return f"MI {ok_mi}/{n_mi} · Rest {ok_r}/{n_r} · {sc:.1f}/{sm:.1f}"
    return f"MI {ok_mi}/{n_mi} · Rest {ok_r}/{n_r}"

def f5_by_class(pack: Optional[Dict[str, Any]]) -> str:
    if not pack:
        return "—"
    bl = pack.get("by_label") or {}
    bits: List[str] = []
    for name in ("Left", "Right", "Rest"):
        d = bl.get(name) or {}
        n = int(d.get("n") or 0)
        ok = int(d.get("ok") or 0)
        if n:
            bits.append(f"{name} {ok}/{n} ({100.0 * ok / n:.0f}%)")
    sc, sm = pack.get("score"), pack.get("score_max")
    if sc is not None and sm is not None:
        bits.append(f"总分 {sc:.1f}/{sm:.1f}")
    return "；".join(bits) if bits else "—"

def pred_s(row: Dict[str, Any]) -> str:
    pl = row.get("pred_labels") or {}
    return ", ".join(f"{k}:{v}" for k, v in pl.items()) if pl else "—"

def load_release(out_dir: str) -> Dict[str, Any]:
    p = Path(out_dir) / "release_gate.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def gate_detail(row: Dict[str, Any]) -> Tuple[str, str]:
    """Return (PASS/FAIL, check summary)."""
    status = "PASS" if row.get("release_pass") else "FAIL"
    rg = load_release(str(row.get("out_dir") or ""))
    checks = rg.get("checks") or {}
    parts: List[str] = []
    if isinstance(checks, dict) and checks:
        for k, v in checks.items():
            if isinstance(v, dict):
                ok = v.get("ok")
                if ok is True:
                    parts.append(f"{k}:OK")
                elif ok is False:
                    parts.append(f"{k}:FAIL")
                else:
                    parts.append(f"{k}:{v}")
            elif v is True:
                parts.append(f"{k}:OK")
            elif v is False:
                parts.append(f"{k}:FAIL")
            else:
                parts.append(f"{k}:{v}")
    else:
        if row.get("heldout_acc") is not None:
            parts.append(f"heldout_smooth={float(row['heldout_acc']):.3f}")
        if row.get("heldout_acc_raw") is not None:
            parts.append(f"heldout_raw={float(row['heldout_acc_raw']):.3f}")
        if row.get("max_class_frac") is not None:
            parts.append(f"max_class_frac={float(row['max_class_frac']):.3f}")
        if row.get("train_minus_heldout") is not None:
            parts.append(f"train_gap={float(row['train_minus_heldout']):.3f}")
    return status, "; ".join(parts) if parts else "—"

def main() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payloads: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
    for sid in SUBJECTS:
        sp = latest_summary(sid)
        payloads[sid] = (sp, json.loads(sp.read_text(encoding="utf-8")))

    lines: List[str] = [
        "# Leave-Next F5 详细报告 · 0828 / 0830 被试队列",
        "",
        "> **口径权威**：[`docs/统计口径方案A_20260831.md`](../../docs/统计口径方案A_20260831.md)",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "- 方案：**A**（展示/早停=smooth；门控=raw）",
        "- 被试：syj0828、xjh0828、cyy0830、fnz0830、wzr0830、xj0830",
        "- 未自动晋升 `models/current`（仅分析）",
        "- 简表源：`leave_next_f5_restfix_20260830_231249.md`",
        "",
        "---",
        "",
        "## 1. 实验口径",
        "",
        "| 项 | 说明 |",
        "|----|------|",
        "| 协议 | Leave-Next：前 k 场训练 → 下一场 heldout |",
        "| 模型 | shallow three（主）+ task（辅）；底座 OpenBMI 3s hop100 |",
        "| Rest | Cue 前静息 = Rest（label=0），来自 trial_table `t_rest_*` |",
        "| F5 | 因果平滑 lookback=2 + 多数票 |",
        "| F5 计分 | Left/Right 各 1 分；Rest 0.5 分 |",
        "| Replay | 前几档 ratio=0.1；末几档关闭（noreplay） |",
        "| 窗级展示 | **heldout_acc = 因果平滑**（对齐在线 acc_window） |",
        "| 早停 | 因果平滑 lookback=2（与 F5 同） |",
        "| 门控 | **raw** `heldout_acc_raw` + max_class_frac + train_gap + three_classes |",
        "",
        "历史批次（20260830）表中「win heldout」若为 raw，以 JSON 内 `heldout_acc_raw` / 批次 stamp 为准。",
        "",
        "### Ramp（典型 6 场）",
        "",
        "```",
        "R1: w01        → w02   (replay on)",
        "R2: w01+w02    → w03   (replay on)",
        "R3: …+w03      → w04   (replay on)",
        "R4: …+w04      → w05   (replay off)",
        "R5: …+w05      → w06   (replay off)",
        "```",
        "",
        "> xjh0828 缺 w01，从 ws02 起 ramp。",
        "",
        "---",
        "",
        "## 2. 跨被试总览（末档）",
        "",
        "| 被试 | 末档 | win heldout | 门控 | F5 FT | F5 base3 | FT−base | 预测分布 |",
        "|------|------|-------------|------|-------|----------|---------|----------|",
    ]

    for sid in SUBJECTS:
        _, payload = payloads[sid]
        rows = payload.get("rows") or []
        if not rows:
            lines.append(f"| {sid} | — | — | — | — | — | — | — |")
            continue
        row = rows[-1]
        ft = row.get("f5_ft") or {}
        b3 = row.get("f5_base_three") or {}
        sc, bc = ft.get("score"), b3.get("score")
        delta = f"{sc - bc:+.1f}" if sc is not None and bc is not None else "—"
        gate, _ = gate_detail(row)
        lines.append(
            f"| **{sid}** | {train_tag(row)} | {float(row.get('heldout_acc') or 0):.3f} | "
            f"{gate} | {f5_mi_rest(ft)} | {f5_mi_rest(b3)} | {delta} | {pred_s(row)} |"
        )

    lines += [
        "",
        "### 读表要点",
        "",
        "1. **syj0828** 末档 heldout **0.916**、F5 **42.0/45**，显著强于底座，且全档过门控。",
        "2. **wzr0830** 是 0830 中唯一末档过门控者；F5 仍受 Rest 偏多拖累。",
        "3. **fnz0830 / xj0830** 修 Rest 后预测里已有 Rest，但 MI 判决偏弱，易 Rest 塌缩。",
        "4. **cyy0830** 中段预测塌成 Left-only，与电极问题一致，不宜当正常微调结论。",
        "5. **xjh0828** 有 Rest、有一定 MI，但门控持续 FAIL。",
        "",
        "---",
        "",
        "## 3. 分被试详细结果",
        "",
    ]

    for sid in SUBJECTS:
        sp, payload = payloads[sid]
        rows = payload.get("rows") or []
        lines += [
            f"### {sid}",
            "",
            f"- 摘要 JSON：`{sp}`",
            f"- 备注：{NOTES.get(sid, '—')}",
            f"- 设备：`{payload.get('device', '—')}` · lookback={payload.get('causal_lookback', '—')}",
            "",
            "| R | train→eval | replay | train_acc | heldout(smooth) | task_heldout | max_class | train_gap(raw) | 门控 | 门控明细 |",
            "|---|------------|--------|-----------|---------|--------------|-----------|-----------|------|----------|",
        ]
        for i, row in enumerate(rows, start=1):
            gate, gdetail = gate_detail(row)
            rep = "on" if row.get("use_replay") else "off"
            ta = row.get("train_acc")
            ha = row.get("heldout_acc")
            tha = row.get("task_heldout_acc")
            mcf = row.get("max_class_frac")
            gap = row.get("train_minus_heldout")
            ok_nums = all(x is not None for x in (ta, ha, tha, mcf, gap))
            if ok_nums:
                metrics = (
                    f"{float(ta):.3f} | {float(ha):.3f} | {float(tha):.3f} | "
                    f"{float(mcf):.3f} | {float(gap):.3f}"
                )
            else:
                metrics = "— | — | — | — | —"
            lines.append(
                f"| R{i} | {train_tag(row)} | {rep} | {metrics} | {gate} | {gdetail} |"
            )

        lines += [
            "",
            "| R | F5 FT（按类） | F5 base3（按类） | F5 e1f 底座（按类） | 窗级预测 | 产出目录 |",
            "|---|---------------|------------------|---------------------|----------|----------|",
        ]
        for i, row in enumerate(rows, start=1):
            out = row.get("out_dir") or "—"
            out_short = Path(out).name if out != "—" else "—"
            e1f = row.get("f5_base_e1f")
            lines.append(
                f"| R{i} | {f5_by_class(row.get('f5_ft'))} | "
                f"{f5_by_class(row.get('f5_base_three'))} | "
                f"{f5_by_class(e1f) if e1f else '—'} | "
                f"{pred_s(row)} | `{out_short}` |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 4. 结论与建议",
        "",
        "| 优先级 | 建议 |",
        "|--------|------|",
        "| 上线候选 | **syj0828** 末档（或 R5 权重）可作强参考；**wzr0830** R4/R5 过门控可备选 |",
        "| 慎用 | xjh0828 / fnz0830 / xj0830：有 Rest，但 MI 或门控不稳 |",
        "| 排除/复采 | **cyy0830** 中段电极异常，应排除坏场或重采后再 FT |",
        "| 切窗 | 已确认 Cue 前静息进 Rest；后续 FT 勿回退到「无 Rest」旧路径 |",
        "",
        "### 相对底座",
        "",
        "- syj0828：FT 相对 base3 大幅提升（末档约 +24.5 分）。",
        "- wzr0830：末档 FT 略优于 base3，且过门控。",
        "- cyy / xj：部分档位 FT 不优于甚至弱于底座。",
        "",
        "---",
        "",
        "## 5. 产物索引",
        "",
        "| 被试 | summary JSON |",
        "|------|--------------|",
    ]
    for sid in SUBJECTS:
        sp, _ = payloads[sid]
        lines.append(f"| {sid} | `{sp}` |")

    lines += [
        "",
        f"- 本报告：`{OUT}`",
        "- 简表：`experiment_game/data/subjects/_analysis/leave_next_f5_restfix_20260830_231249.md`",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    return OUT

if __name__ == "__main__":
    main()
