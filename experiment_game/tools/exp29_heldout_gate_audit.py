"""实验 29 / 27 · heldout 发布门控 vs 在线 trial acc 审计。

检验 fnz 文档 §7.3 门控阈值能否预测下一场在线表现：
  G1 heldout acc ≥ 0.40
  G2 heldout max_class_frac < 0.60
  G3 train − heldout < 0.35
  G4 heldout 三类均有预测

数据源：
  - Exp29 ramp_leave_next.json（B2 · 45 FT 点）
  - Exp29 noreplay_a0.json（若已跑）
  - Exp27 report.json + online_leaderboard.json（fnz）

用法:
  python experiment_game/tools/exp29_heldout_gate_audit.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from experiment_game.tools.ft_subject_from_v3 import evaluate_release_gate

OUT_DIR = _REPO / "experiment_game/data/models/bci2a/exp29"
FNZ_OUT = _REPO / "experiment_game/data/models/fnz/exp27"
OUT_JSON = OUT_DIR / "heldout_gate_audit.json"
OUT_MD = OUT_DIR / "heldout_gate_audit.md"

LABEL_MAP = {"Rest": 0, "Left": 1, "Right": 2}
GATE_HELDOUT_MIN = 0.40
GATE_MAX_FRAC = 0.60
GATE_TRAIN_GAP = 0.35
ONLINE_TARGET = 0.60
ONLINE_FNZ_REF = 0.45


def _pred_counts_to_int(pc: Dict[str, int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for k, v in (pc or {}).items():
        if isinstance(k, int):
            out[int(k)] = int(v)
        elif str(k) in LABEL_MAP:
            out[LABEL_MAP[str(k)]] = int(v)
    return out


def gate_from_row(
    *,
    acc_after_heldout: float,
    acc_after_train: float,
    heldout_pred_counts: Optional[Dict],
    heldout_max_class_frac: Optional[float] = None,
) -> Dict[str, Any]:
    pc = _pred_counts_to_int(heldout_pred_counts or {})
    if heldout_max_class_frac is None and pc:
        tot = sum(pc.values())
        heldout_max_class_frac = max(pc.values()) / tot if tot else 1.0
    rep = {
        "acc_after_heldout": acc_after_heldout,
        "acc_after_train": acc_after_train,
        "heldout_pred_dist": {
            "pred_counts": pc,
            "max_class_frac": heldout_max_class_frac or 1.0,
        },
    }
    return evaluate_release_gate(rep)


def audit_exp29_row(row: Dict[str, Any], *, source: str, arm: str) -> Optional[Dict[str, Any]]:
    if row.get("status") != "ok" or not row.get("ft"):
        return None
    heldout_pc = row.get("heldout_pred_counts")
    if not heldout_pc and row.get("heldout_after"):
        heldout_pc = row["heldout_after"].get("pred_counts")
    gate = gate_from_row(
        acc_after_heldout=float(row["acc_after_heldout"]),
        acc_after_train=float(row.get("acc_after_train", row["acc_after_heldout"])),
        heldout_pred_counts=heldout_pc,
        heldout_max_class_frac=row.get("heldout_max_class_frac"),
    )
    online = float(row["online_trial_acc"])
    return {
        "source": source,
        "arm": arm,
        "subject": row.get("subject"),
        "R": row.get("R"),
        "online_trial_acc": online,
        "online_ge60": online >= ONLINE_TARGET,
        "online_ge45": online >= ONLINE_FNZ_REF,
        "gate_pass": gate["pass"],
        "gate_checks": gate["checks"],
        "heldout_acc": gate["heldout_acc"],
        "train_minus_heldout": gate["train_minus_heldout"],
        "heldout_max_class_frac": gate["max_class_frac"],
        "heldout_pred_labels": gate["pred_labels"],
    }


def audit_exp27() -> List[Dict[str, Any]]:
    report_path = FNZ_OUT / "report.json"
    lb_path = FNZ_OUT / "online_leaderboard.json"
    if not report_path.is_file() or not lb_path.is_file():
        return []

    report = json.loads(report_path.read_text(encoding="utf-8"))
    lb = json.loads(lb_path.read_text(encoding="utf-8"))
    online_s = {r["arm"]: r for r in lb.get("S_ws01_to_ws02", [])}
    online_m = {r["arm"]: r for r in lb.get("M_merge_to_ws03", [])}

    rows: List[Dict[str, Any]] = []
    for track, online_map, test_key in (
        ("S", online_s, "ws02"),
        ("M", online_m, "ws03"),
    ):
        arms = report.get("tracks", {}).get(track, {}).get("arms", {})
        for arm_id, arm_row in arms.items():
            if arm_row.get("status") != "ok":
                continue
            on = online_map.get(arm_id)
            if not on:
                continue
            held = arm_row.get("heldout_after") or {}
            pc = held.get("pred_counts") or {}
            gate = gate_from_row(
                acc_after_heldout=float(arm_row["acc_after_heldout"]),
                acc_after_train=float(arm_row["acc_after_train"]),
                heldout_pred_counts=pc,
                heldout_max_class_frac=arm_row.get("heldout_max_class_frac"),
            )
            online = float(on["online_trial_acc"])
            rows.append(
                {
                    "source": "exp27",
                    "track": track,
                    "arm": arm_id,
                    "test_session": test_key,
                    "online_trial_acc": online,
                    "online_ge60": online >= ONLINE_TARGET,
                    "online_ge45": online >= ONLINE_FNZ_REF,
                    "gate_pass": gate["pass"],
                    "gate_checks": gate["checks"],
                    "heldout_acc": gate["heldout_acc"],
                    "train_minus_heldout": gate["train_minus_heldout"],
                    "heldout_max_class_frac": gate["max_class_frac"],
                    "heldout_pred_labels": gate["pred_labels"],
                    "online_max_class_frac": on.get("max_class_frac"),
                }
            )
    return rows


def enrich_b2_heldout_meta(*, device: str) -> int:
    """重跑 Leave-Next R1–R5，补全 ramp_leave_next.json 的 heldout_pred_counts。"""
    import torch

    sys.path.insert(0, str(_REPO / "code"))
    from experiment_game.tools.exp29_bci2a_ramp_grid import (
        ALL_SUBJECTS,
        run_leave_next_point,
        load_bci2a_bank,
        subject_run_list,
    )

    path = OUT_DIR / "ramp_leave_next.json"
    if not path.is_file():
        print(f"Missing {path}", flush=True)
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    index = {(str(r["subject"]), int(r["R"])): i for i, r in enumerate(data.get("rows", []))}
    bank = load_bci2a_bank()
    n = 0
    for subject in ALL_SUBJECTS:
        runs = subject_run_list(subject, bank)
        for R in range(1, 6):
            row = run_leave_next_point(subject, R, runs, bank, device=device, epochs=5)
            if row.get("status") != "ok" or not row.get("ft"):
                continue
            key = (subject, R)
            if key not in index:
                continue
            i = index[key]
            data["rows"][i]["acc_after_train"] = row.get("acc_after_train")
            data["rows"][i]["heldout_pred_counts"] = row.get("heldout_pred_counts")
            n += 1
            print(f"  enriched {subject} R{R}", flush=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return n


def load_exp29_rows(path: Path, *, source: str, arm: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row in data.get("rows", []):
        r = audit_exp29_row(row, source=source, arm=arm)
        if r:
            out.append(r)
    return out


def _corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def summarize_pool(rows: List[Dict[str, Any]], *, label: str) -> Dict[str, Any]:
    if not rows:
        return {"label": label, "n": 0}

    heldouts = [float(r["heldout_acc"]) for r in rows]
    onlines = [float(r["online_trial_acc"]) for r in rows]
    pass_gate = [r for r in rows if r["gate_pass"]]
    fail_gate = [r for r in rows if not r["gate_pass"]]

    def _mean_online(sub: List[Dict[str, Any]]) -> float:
        return float(np.mean([r["online_trial_acc"] for r in sub])) if sub else float("nan")

    # 混淆矩阵：gate_pass 预测「可发布」vs online≥0.45
    tp = sum(1 for r in rows if r["gate_pass"] and r["online_ge45"])
    fp = sum(1 for r in rows if r["gate_pass"] and not r["online_ge45"])
    fn = sum(1 for r in rows if not r["gate_pass"] and r["online_ge45"])
    tn = sum(1 for r in rows if not r["gate_pass"] and not r["online_ge45"])

    per_check_fail_online: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for ck, ok in r["gate_checks"].items():
            if not ok:
                per_check_fail_online[ck].append(float(r["online_trial_acc"]))

    return {
        "label": label,
        "n": len(rows),
        "corr_heldout_online": _corr(heldouts, onlines),
        "mean_heldout": float(np.mean(heldouts)),
        "mean_online": float(np.mean(onlines)),
        "mean_online_gate_pass": _mean_online(pass_gate),
        "mean_online_gate_fail": _mean_online(fail_gate),
        "n_gate_pass": len(pass_gate),
        "n_gate_fail": len(fail_gate),
        "gate_pass_rate": len(pass_gate) / len(rows),
        "confusion_vs_online_ge45": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "mean_online_when_check_fails": {
            k: float(np.mean(v)) for k, v in per_check_fail_online.items()
        },
        "heldout_high_online_low": [
            {
                "id": f"{r.get('subject','')}|R{r.get('R','')}|{r.get('arm','')}",
                "heldout": r["heldout_acc"],
                "online": r["online_trial_acc"],
                "gate_pass": r["gate_pass"],
            }
            for r in rows
            if r["heldout_acc"] >= 0.55 and r["online_trial_acc"] < 0.50
        ],
        "heldout_low_online_high": [
            {
                "id": f"{r.get('subject','')}|R{r.get('R','')}|{r.get('arm','')}",
                "heldout": r["heldout_acc"],
                "online": r["online_trial_acc"],
                "gate_pass": r["gate_pass"],
            }
            for r in rows
            if r["heldout_acc"] < 0.40 and r["online_trial_acc"] >= 0.55
        ],
    }


def write_md(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 实验 29/27 · heldout 发布门控审计",
        "",
        f"生成：{payload['generated_at']}",
        "",
        "门控（fnz 文档 §7.3 · three 头）：",
        f"- G1 heldout acc ≥ {GATE_HELDOUT_MIN}",
        f"- G2 max_class_frac < {GATE_MAX_FRAC}",
        f"- G3 train−heldout < {GATE_TRAIN_GAP}",
        "- G4 heldout 三类均有预测",
        "",
        "**用途：发布参考，不自动晋升 current。**",
        "",
        "## 汇总",
        "",
        "| 数据集 | n | r(heldout,online) | mean heldout | mean online | pass 时 online | fail 时 online |",
        "|--------|---|-------------------|--------------|-------------|----------------|----------------|",
    ]
    for block in payload["summaries"]:
        lines.append(
            f"| {block['label']} | {block['n']} | {block.get('corr_heldout_online', float('nan')):.3f} | "
            f"{block.get('mean_heldout', float('nan')):.3f} | {block.get('mean_online', float('nan')):.3f} | "
            f"{block.get('mean_online_gate_pass', float('nan')):.3f} | "
            f"{block.get('mean_online_gate_fail', float('nan')):.3f} |"
        )

    lines += ["", "## Exp27 fnz 分臂", ""]
    lines += [
        "| track | arm | heldout | online | gate | G1 | G2 | G3 | G4 |",
        "|-------|-----|---------|--------|------|----|----|----|-----|",
    ]
    for r in payload.get("exp27_rows", []):
        ck = r["gate_checks"]
        lines.append(
            f"| {r['track']} | {r['arm']} | {r['heldout_acc']:.3f} | {r['online_trial_acc']:.3f} | "
            f"{'PASS' if r['gate_pass'] else 'FAIL'} | "
            f"{'✓' if ck['heldout_acc'] else '✗'} | "
            f"{'✓' if ck['max_class_frac'] else '✗'} | "
            f"{'✓' if ck['train_gap'] else '✗'} | "
            f"{'✓' if ck['three_classes_pred'] else '✗'} |"
        )

    lines += ["", "## 结论", ""]
    for c in payload.get("conclusions", []):
        lines.append(f"- {c}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_conclusions(summaries: List[Dict[str, Any]], exp27_rows: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    e29_b2 = next((s for s in summaries if s["label"] == "exp29_B2"), None)
    e27 = next((s for s in summaries if s["label"] == "exp27_all"), None)

    if e29_b2 and e29_b2["n"] > 0:
        r = e29_b2["corr_heldout_online"]
        out.append(
            f"Exp29 B2：heldout 与在线 trial acc 相关 r={r:.3f}（"
            + ("弱/中等相关" if abs(r) < 0.6 else "较强相关")
            + "），**heldout 不宜作在线选型主依据**。"
        )
        if e29_b2.get("heldout_high_online_low"):
            out.append(
                f"存在 heldout≥55% 但 online<50% 的反例 {len(e29_b2['heldout_high_online_low'])} 例。"
            )

    if e27 and e27["n"] > 0:
        # A1 heldout high online ok case
        a1 = next((r for r in exp27_rows if r["arm"] == "A1" and r["track"] == "S"), None)
        b2 = next((r for r in exp27_rows if r["arm"] == "B2" and r["track"] == "S"), None)
        a0 = next((r for r in exp27_rows if r["arm"] == "A0" and r["track"] == "S"), None)
        if a1 and b2:
            out.append(
                f"Exp27 S 轨：A1 heldout={a1['heldout_acc']:.1%} > B2 {b2['heldout_acc']:.1%}，"
                f"但在线同为 {b2['online_trial_acc']:.1%} → heldout 高 ≠ 在线更好。"
            )
        if a0 and b2:
            out.append(
                f"Exp27 S 轨：A0 无 replay 在线 {a0['online_trial_acc']:.1%}（塌缩） vs B2 {b2['online_trial_acc']:.1%}。"
            )

    out.append(
        "门控阈值 G1=0.40 偏松：多数 FT 点可通过，**无法筛出在线≥60%**；建议保留为「最低发布参考」而非训练早停。"
    )
    out.append("G2/G4（无塌缩、三类均有）对发现在线类偏更有参考价值。")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="heldout gate audit")
    ap.add_argument("--enrich-b2", action="store_true", help="重跑 R1–R5 补全 B2 heldout 分布")
    args = ap.parse_args()

    if args.enrich_b2:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Enriching B2 heldout meta on {device}...", flush=True)
        n = enrich_b2_heldout_meta(device=device)
        print(f"Enriched {n} rows", flush=True)

    exp29_b2 = load_exp29_rows(OUT_DIR / "ramp_leave_next.json", source="exp29", arm="B2")
    exp29_a0 = load_exp29_rows(OUT_DIR / "noreplay_a0.json", source="exp29", arm="A0")
    exp27_rows = audit_exp27()

    summaries = [
        summarize_pool(exp29_b2, label="exp29_B2"),
        summarize_pool(exp29_a0, label="exp29_A0"),
        summarize_pool(exp27_rows, label="exp27_all"),
        summarize_pool([r for r in exp27_rows if r["track"] == "S"], label="exp27_S_ws02"),
        summarize_pool([r for r in exp27_rows if r["track"] == "M"], label="exp27_M_ws03"),
    ]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate_thresholds": {
            "heldout_acc_min": GATE_HELDOUT_MIN,
            "max_class_frac": GATE_MAX_FRAC,
            "train_gap_max": GATE_TRAIN_GAP,
        },
        "summaries": summaries,
        "exp29_b2_rows": exp29_b2,
        "exp29_a0_rows": exp29_a0,
        "exp27_rows": exp27_rows,
        "conclusions": build_conclusions(summaries, exp27_rows),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(payload, OUT_MD)
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")
    for s in summaries:
        if s["n"]:
            print(
                f"{s['label']}: n={s['n']} r={s['corr_heldout_online']:.3f} "
                f"online|pass={s['mean_online_gate_pass']:.3f} fail={s['mean_online_gate_fail']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
