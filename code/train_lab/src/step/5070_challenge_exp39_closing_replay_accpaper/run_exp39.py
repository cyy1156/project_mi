# -*- coding: utf-8 -*-
"""Exp39 编排。

用法：
  python run_exp39.py --update-registry
  python run_exp39.py --write-csv   # 仅工程轨要求新文件时
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_STEP = Path(__file__).resolve().parent
from exp39_config import exp39_out, scheme_doc  # noqa: E402


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(_STEP / script), *(extra or [])]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_STEP))


def _fmt(mean, std) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def _update_registry() -> None:
    latest = exp39_out() / "replay" / "ranking_latest.json"
    if not latest.is_file():
        raise SystemExit(f"missing {latest}")
    doc = json.loads(latest.read_text(encoding="utf-8"))
    arms = doc.get("arms") or {}
    gates = doc.get("gates") or {}
    sci = doc.get("decision_science") or {}
    eng = doc.get("decision_engineering") or {}
    uni = doc.get("uni50_enter") or {}

    def arm_row(aid: str, note: str = "") -> str:
        a = arms.get(aid) or {}
        g = gates.get(aid) or {}
        mean = a.get("val_acc_mean", g.get("val_acc_mean"))
        std = a.get("val_acc_std", g.get("val_acc_std"))
        if mean is None:
            return f"| {aid} | — | — | — | — | — | ⬜ |"
        if aid == "N0":
            return f"| N0 | {_fmt(mean, std)} | — | — | — | 锚 | ✅ |"
        d_n0 = g.get("nested_delta_vs_base")
        # R-B8_vs_V1 另表；此处 vs N0
        if aid == "R-B8" and g.get("base_arm") == "N0":
            pass
        delta_s = f"{d_n0*100:+.1f}pp" if isinstance(d_n0, (int, float)) else "—"
        pw = g.get("wilcoxon_p")
        pw_s = f"{pw:.4f}" if isinstance(pw, float) and pw == pw else "—"
        ins = "是" if a.get("in_sample_theta") or g.get("in_sample_theta") else "否"
        return (
            f"| {aid} | {_fmt(mean, std)} | {delta_s} | {pw_s} | {ins} | {note} | ✅ |"
        )

    rb8_v1 = gates.get("R-B8_vs_V1") or {}
    ranking_lines = [
        "| 名次 | 臂 | 嵌套 Val | in-sample-θ |",
        "|------|----|----------|-------------|",
    ]
    for i, r in enumerate(doc.get("honest_ranking") or [], 1):
        ranking_lines.append(
            f"| {i} | {r['arm']} | {r['nested_mean']:.3f}±{r['nested_std']:.3f} | "
            f"{'是' if r.get('in_sample_theta') else '否'} |"
        )

    write_csv = "是" if eng.get("write_new_csv") else "否"
    eng_id = eng.get("decision_engineering", "—")
    sci_id = sci.get("decision_science", "—")

    text = f"""# 实验 39 · 结果登记

> 方案：[../方案.md](../方案.md) · **v0.1**  
> 状态：**已结案** · `{doc.get('generated_at')}` · device=5070  
> 科学轨：`{sci_id}` · 工程轨：`{eng_id}`（{eng.get('label', '')}）  
> 产物：`code/train_lab/out/5070_challenge_exp39_closing_replay_accpaper/`

## 表 0 · 双轨门控

| 轨 | 规则 | 本跑结果 |
|----|------|----------|
| 科学 | 默认 KEEP_S0_discipline；Wilcoxon 不作洗白闸 | `{sci_id}` |
| 工程 | nested 均值最高；Δ&lt;1pp → 最简；标注非显著 | `{eng_id}` |
| 写新 CSV | 仅工程轨非 S0_csv | {write_csv} |

## 表 1 · 诚实排行榜（leave-fold；排除 pool）

{chr(10).join(ranking_lines)}

## 表 2 · 主测臂明细

| 臂 | 嵌套 Val | vs N0 | Wilcoxon p | in-sample-θ | 备注 | 状态 |
|----|----------|-------|------------|-------------|------|------|
{arm_row("N0", "复用/重算 nested-S0")}
{arm_row("F1", "a59_conformer")}
{arm_row("V1", "b8_shallow_b")}
{arm_row("R-B8", "**主缺口** nested E1f-B8-ft")}
{arm_row("R-uni50", "固定 50/50")}
{arm_row("R-pool-S0", "诊断；不进冠军键")}
{arm_row("R-pool-B8", "诊断；不进冠军键")}

## 表 3 · Q2 · R-B8 vs V1

| 项 | 值 |
|----|-----|
| R-B8 nested | {arms.get('R-B8', {}).get('val_acc_mean')} |
| V1 nested | {arms.get('V1', {}).get('val_acc_mean')} |
| Δ (R-B8 − V1) | {rb8_v1.get('nested_delta_vs_base', float('nan'))*100:+.2f}pp |
| Wilcoxon p | {rb8_v1.get('wilcoxon_p')} |

## 表 4 · R-uni50 进入门槛（§5.3）

| 项 | 值 |
|----|-----|
| ≥5/6 折 ≥ F1 | {uni.get('n_folds_ge_f1')} / 6 |
| pooled Acc | {uni.get('pooled_acc')} |
| 门槛 pooled≥0.53 | {uni.get('pooled_line')} |
| 进入工程候选 | {'是' if uni.get('enter_engineering_set') else '否'} |

## 表 5 · 工程裁决明细

| 项 | 值 |
|----|-----|
| 候选集 | {eng.get('candidates')} |
| 平手池（Δ&lt;1pp） | {eng.get('tie_pool')} |
| scored | `{json.dumps(eng.get('scored'), ensure_ascii=False)}` |
| `decision_science` | `{sci_id}` |
| `decision_engineering` | `{eng_id}` |
| 是否写 `submission_exp39_*` | {write_csv} |
| 主 CSV 科学叙事 | Exp34 S0（纪律） |
| 主 CSV 工程终态 | {eng_id if eng.get('write_new_csv') else '维持 submission_exp34_e1f_a59_sens_full_20260902_1930.csv'} |

## 决策

- **科学**：{sci.get('note', sci_id)}  
- **工程**：选 `{eng_id}` · {eng.get('label')}  
- R-B8 vs N0：Δ={sci.get('rb8_delta_vs_n0', 0)*100:+.2f}pp · Wilcoxon p={sci.get('rb8_wilcoxon_p')}  
- **不开 Exp40 算法臂**；官方算法线收工。
"""
    reg = scheme_doc() / "总结" / "结果登记表.md"
    reg.parent.mkdir(parents=True, exist_ok=True)
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
        _run("replay_closing.py", ["--n-boot", str(args.n_boot)])

    if args.update_registry or not args.skip_replay:
        _update_registry()

    if args.write_csv:
        _run("write_engineering_csv.py")

    print("DONE exp39")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
