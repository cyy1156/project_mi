# -*- coding: utf-8 -*-
"""Exp37 编排。

用法：
  python run_exp37.py --update-registry
  python run_exp37.py --write-csv   # 仅 N7 过线时
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_STEP = Path(__file__).resolve().parent
from exp37_config import exp37_out, scheme_doc  # noqa: E402


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(_STEP / script), *(extra or [])]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_STEP))


def _update_registry() -> None:
    latest = exp37_out() / "replay" / "nested_latest.json"
    if not latest.is_file():
        raise SystemExit(f"missing {latest}")
    doc = json.loads(latest.read_text(encoding="utf-8"))
    gates = doc.get("gates") or {}
    arms = doc.get("arms") or {}

    def row(aid: str, role: str) -> str:
        g = gates.get(aid) or {}
        a = arms.get(aid) or {}
        mean = g.get("val_acc_mean", a.get("val_acc_mean"))
        std = g.get("val_acc_std", a.get("val_acc_std", 0))
        if mean is None:
            return f"| {aid} | {role} | — | — | — | — | — | — | ⬜ |"
        if aid == "N0":
            return (
                f"| N0 | 锚 | {mean:.3f}±{std:.3f} | — | — | — | — | — | ✅ |"
            )
        delta = g.get("nested_delta_vs_n0", 0) * 100
        pw = g.get("wilcoxon_p")
        pw_s = f"{pw:.4f}" if isinstance(pw, float) and pw == pw else "—"
        f0d = g.get("fold0_delta")
        f0d_s = f"{f0d*100:+.1f}pp" if isinstance(f0d, float) else "—"
        coll = "是" if g.get("fold0_collapsed") else "否"
        passed = "✅" if g.get("pass_replace") else "否"
        return (
            f"| {aid} | {role} | {mean:.3f}±{std:.3f} | {delta:+.1f}pp | {pw_s} | "
            f"{f0d_s} | {coll} | {passed} | ✅ |"
        )

    table1 = "\n".join(
        [
            "## 表 1 · 嵌套主结果",
            "",
            "| 臂 | 层级 | 嵌套 Val | vs N0 | Wilcoxon p | fold0 Δ | fold0 塌缩 | 过线 | 状态 |",
            "|----|------|----------|-------|------------|---------|------------|------|------|",
            row("N0", "锚"),
            row("N7", "**主**"),
            row("N7b", "附报"),
            row("N7_AC", "附报"),
            row("N7_ABC", "附报"),
            "",
        ]
    )

    def aux_row(aid: str) -> str:
        g = gates.get(aid) or {}
        if aid == "N0" or "mcnemar" not in g:
            return ""
        m = g["mcnemar"]
        b = g.get("cluster_bootstrap") or {}
        sus = (b.get("suspect_single_subject") or {}).get("flag")
        ci = b.get("ci95")
        ci_s = f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]pp" if ci else "—"
        return (
            f"| {aid} | {m.get('p_exact', float('nan')):.4f} | {ci_s} | "
            f"{'是' if sus else '否'} | b={m.get('b_a_ok_b_wrong')}, c={m.get('c_a_wrong_b_ok')} |"
        )

    table2_lines = [
        "## 表 2 · 辅证",
        "",
        "| 臂 | McNemar p | cluster-boot Δ 95% CI | 单被试风险 | 备注 |",
        "|----|-----------|------------------------|------------|------|",
    ]
    for aid in ("N7", "N7b", "N7_AC", "N7_ABC"):
        r = aux_row(aid)
        if r:
            table2_lines.append(r)
    table2 = "\n".join(table2_lines) + "\n"

    decision = doc.get("decision")
    status = doc.get("status")
    n7 = gates.get("N7") or {}
    if decision == "REPLACE_with_N7":
        decision_md = (
            f"**过线：换主 CSV 为 N7 栈。** nested={n7.get('val_acc_mean'):.3f} "
            f"（Δ={n7.get('nested_delta_vs_n0', 0)*100:+.1f}pp，Wilcoxon p={n7.get('wilcoxon_p')}）。"
            f" upgrade_ABC={doc.get('upgrade_discuss_n7_abc')}。"
        )
        status_line = "**N7 过线 · 可换主 CSV**"
    elif status == "DELTA_OK_WILCOXON_FAIL":
        decision_md = (
            f"**弱阳性：嵌套 Δ≥1pp 但 Wilcoxon 未过 → 不换 CSV。** "
            f"N7 nested={n7.get('val_acc_mean'):.3f} Δ={n7.get('nested_delta_vs_n0', 0)*100:+.1f}pp "
            f"p={n7.get('wilcoxon_p')}。"
        )
        status_line = "**弱阳性 · 维持 S0**"
    else:
        decision_md = (
            f"**不换 CSV（嵌套涨点缩水或未过线）。** "
            f"N7 nested={n7.get('val_acc_mean')} Δ={n7.get('nested_delta_vs_n0')} "
            f"p={n7.get('wilcoxon_p')}。Exp36 折内 +4.7~6.8pp 含显著 in-sample 水分之结论成立。"
        )
        status_line = "**未过线 · 维持 S0**"

    reg = scheme_doc() / "总结" / "结果登记表.md"
    text = f"""# 实验 37 · 结果登记

> 方案：[../方案.md](../方案.md) · **v0.2**  
> 状态：{status_line} · device=5070 · `{doc.get('generated_at')}`  
> 对照锚：**N0 = nested-S0**  
> 过线：**N7** 嵌套 Δ≥+1pp vs N0 **且** Wilcoxon p&lt;0.05  
> 产物：`code/train_lab/out/5070_challenge_exp37_nested_mcnemar_accpaper/`

## 表 0 · 锚与尺子

| 项 | 值 |
|----|-----|
| 交卷锚 CSV（过线前） | Exp34 S0 |
| 对照 | N0 nested-S0 = {gates.get('N0', {}).get('val_acc_mean')} |
| 主确认臂 | N7 |
| 主检验 | 嵌套 6 折 Wilcoxon |
| 辅证 | McNemar + 被试 cluster bootstrap（n_boot={doc.get('n_boot')}） |
| C1 dump | `{doc.get('c1_run')}` |

{table1}
{table2}
## 表 3 · Exp36 折内对照（只读）

| 臂 | Val | 逐折 Δpp | Wilcoxon | 配对 t |
|----|-----|----------|----------|--------|
| M7 | 0.604 | [0, +6.7, +4.7, +11.3, +4.0, +1.3] | 0.0625 | ≈0.037 |
| M7_ABC | 0.626 | [0, +9.3, +4.7, +19.3, +6.0, +1.3] | 0.0625 | ≈0.064 |

## 决策

{decision_md}

decision=`{decision}` · status=`{status}` · upgrade_ABC=`{doc.get('upgrade_discuss_n7_abc')}`
"""
    reg.write_text(text, encoding="utf-8")
    print("updated", reg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-registry", action="store_true")
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-replay", action="store_true")
    args = ap.parse_args()

    if not args.skip_replay:
        _run("replay_nested.py", ["--n-boot", str(args.n_boot)])

    if args.update_registry or not args.skip_replay:
        _update_registry()

    if args.write_csv:
        latest = exp37_out() / "replay" / "nested_latest.json"
        doc = json.loads(latest.read_text(encoding="utf-8"))
        if not doc.get("n7_pass_replace"):
            print("N7 未过线，拒绝写 CSV（按 §5）")
            return 1
        print("write-csv：N7 已过线，但 submission 导出脚本尚未接入（需接 Exp34 predict 管线）。")
        print("请手动：用 N7 嵌套协议在 test 上重放并写 submission_exp37_*。")
        return 0

    print("DONE exp37")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
