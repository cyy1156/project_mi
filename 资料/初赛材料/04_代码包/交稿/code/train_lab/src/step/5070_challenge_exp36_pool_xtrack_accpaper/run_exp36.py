# -*- coding: utf-8 -*-
"""Exp36 编排。

用法：
  python run_exp36.py --stage day0 --update-registry
  python run_exp36.py --stage day1 --update-registry
  python run_exp36.py --stage full --update-registry
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_STEP = Path(__file__).resolve().parent
from exp36_config import exp36_out, scheme_doc  # noqa: E402


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(_STEP / script), *(extra or [])]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_STEP))


def _update_registry() -> None:
    day0_p = exp36_out() / "replay" / "day0_latest.json"
    day1_p = exp36_out() / "replay" / "day1_latest.json"
    reg = scheme_doc() / "总结" / "结果登记表.md"
    text = reg.read_text(encoding="utf-8")

    def _rows_day0(doc: dict) -> str:
        arms, gates = doc.get("arms") or {}, doc.get("gates") or {}
        s0 = arms.get("S0") or {}
        lines = [
            "| 臂 | Val | vs S0 | Wilcoxon p | Val 拟合参数量 | 过线 | 状态 |",
            "|----|-----|-------|------------|----------------|------|------|",
            f"| S0 | {s0.get('val_acc_mean', float('nan')):.3f}±{s0.get('val_acc_std', 0):.3f} | — | — | — | — | ✅ |",
        ]
        for aid in ("M7", "MC0", "M7c"):
            if aid not in arms:
                continue
            a, g = arms[aid], gates.get(aid) or {}
            if aid == "MC0":
                mean = (a.get("nested") or {}).get("val_acc_mean", a.get("val_acc_mean"))
                std = (a.get("nested") or {}).get("val_acc_std", a.get("val_acc_std"))
                nparam = "3 (nested)"
            else:
                mean, std = a.get("val_acc_mean"), a.get("val_acc_std", 0)
                nparam = str(a.get("n_fit_params", ""))
            p = g.get("wilcoxon_p")
            p_s = f"{p:.3f}" if isinstance(p, float) and p == p else "—"
            lines.append(
                f"| {aid} | {mean:.3f}±{std:.3f} | {g.get('delta_vs_s0', 0)*100:+.1f}pp | {p_s} | {nparam} | "
                f"{'✅' if g.get('pass_replace') else '否'} | ✅ |"
            )
        return "\n".join(lines)

    def _rows_day1(doc: dict) -> str:
        arms, gates = doc.get("arms") or {}, doc.get("gates") or {}
        lines = [
            "| 臂 | Val | vs S0 | Wilcoxon p | 过线 | 状态 |",
            "|----|-----|-------|------------|------|------|",
        ]
        for aid, a in arms.items():
            g = gates.get(aid) or {}
            p = g.get("wilcoxon_p")
            p_s = f"{p:.3f}" if isinstance(p, float) and p == p else "—"
            lines.append(
                f"| {aid} | {a.get('val_acc_mean', float('nan')):.3f}±{a.get('val_acc_std', 0):.3f} | "
                f"{g.get('delta_vs_s0', 0)*100:+.1f}pp | {p_s} | "
                f"{'✅' if g.get('pass_replace') else '否'} | ✅ |"
            )
        return "\n".join(lines)

    if day0_p.is_file():
        d0 = json.loads(day0_p.read_text(encoding="utf-8"))
        table = "## 表 1 · Day0 零训练\n\n" + _rows_day0(d0) + "\n"
        text, _ = re.subn(
            r"## 表 1 · Day0 零训练\n\n\|[\s\S]*?(?=\n## 表 2)",
            table + "\n",
            text,
            count=1,
        )

    any_pass = False
    if day1_p.is_file():
        d1 = json.loads(day1_p.read_text(encoding="utf-8"))
        any_pass = bool(d1.get("any_pass_replace"))
        table = "## 表 2 · Day1 扩池 / 配方\n\n" + _rows_day1(d1) + "\n"
        text, _ = re.subn(
            r"## 表 2 · Day1 扩池 / 配方\n\n\|[\s\S]*?(?=\n## 表 3)",
            table + "\n",
            text,
            count=1,
        )
        # decision
        best = None
        for aid, g in (d1.get("gates") or {}).items():
            if best is None or g.get("val_acc", -1) > best[1].get("val_acc", -1):
                best = (aid, g)
        if any_pass:
            decision = f"**有过线臂**（含 {best[0] if best else '?'}）→ 可写主 CSV / 勾选 §12.1。"
            status = "**Day1 完成 · 有过线候选**"
        else:
            decision = (
                f"Day1 最高 {best[0]}={best[1].get('val_acc'):.3f}（p={best[1].get('wilcoxon_p')}）"
                f"仍未同时满足 Val≥0.568∧p<0.05。C1 待赛规确认；本方案可按收手纪律结案或挂 C1。"
            )
            status = "**Day0+Day1 完成 · 暂无过线**"

        text = re.sub(r"> 状态：.*", f"> 状态：{status} · device=5070  ", text, count=1)
        text = re.sub(r"## 决策\n\n[\s\S]*", f"## 决策\n\n{decision}\n", text, count=1)

    reg.write_text(text, encoding="utf-8")
    print("updated", reg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("day0", "day1", "day2", "full"), default="full")
    ap.add_argument("--update-registry", action="store_true")
    ap.add_argument("--arms", default="B1,B2,B3,B4", help="Day1 训练臂")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--day2-limit-mats", type=int, default=None)
    args = ap.parse_args()

    if args.stage in ("day0", "full"):
        _run("replay_m7_margin.py")

    if args.stage in ("day1", "full"):
        train_extra = ["--arms", args.arms, "--max-folds", str(args.max_folds)]
        if args.run_tag:
            train_extra += ["--run-tag", args.run_tag]
        _run("train_day1.py", train_extra)
        _run("replay_day1.py")

    if args.stage == "day2":
        c1_extra: list[str] = ["--stage", "all"]
        if args.run_tag:
            c1_extra += ["--run-tag", args.run_tag]
        if args.day2_limit_mats is not None:
            c1_extra += ["--limit-mats", str(args.day2_limit_mats)]
        if args.max_folds:
            c1_extra += ["--max-folds", str(args.max_folds)]
        _run("train_c1.py", c1_extra)
        _run("replay_day2.py")

    if args.update_registry:
        _update_registry()

    print("DONE", args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
