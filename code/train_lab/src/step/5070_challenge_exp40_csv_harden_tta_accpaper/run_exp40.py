# -*- coding: utf-8 -*-
"""Exp40 编排。

用法：
  python run_exp40.py --update-registry
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_STEP = Path(__file__).resolve().parent
from exp40_config import exp40_out, scheme_doc  # noqa: E402


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(_STEP / script), *(extra or [])]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_STEP))


def _fmt(mean, std) -> str:
    if mean is None:
        return "—"
    return f"{mean:.3f}±{std:.3f}"


def _update_registry() -> None:
    latest = exp40_out() / "replay" / "harden_latest.json"
    if not latest.is_file():
        raise SystemExit(f"missing {latest}")
    doc = json.loads(latest.read_text(encoding="utf-8"))
    arms = doc.get("arms") or {}
    eng = doc.get("decision_engineering") or {}
    h0 = doc.get("h0_risk") or {}
    final = eng.get("decision_engineering_final", "—")

    def row(aid: str) -> str:
        a = arms.get(aid) or {}
        if aid == "R-B8_raw":
            return (
                f"| R-B8_raw | {_fmt(a.get('val_acc_mean'), a.get('val_acc_std'))} | — | — | 是 | ✅ |"
            )
        if not a:
            return f"| {aid} | — | — | — | — | ⬜ |"
        d = a.get("delta_vs_rb8")
        ds = f"{d*100:+.1f}pp" if isinstance(d, (int, float)) else "—"
        nge = a.get("n_folds_ge", "—")
        ent = "是" if a.get("enter_candidate") else "否"
        return (
            f"| {aid} | {_fmt(a.get('val_acc_mean'), a.get('val_acc_std'))} | {ds} | "
            f"{nge}/6 | {ent} | ✅ |"
        )

    sub_meta = exp40_out() / "submissions" / "final_decision_latest.json"
    csv_path = "—"
    if sub_meta.is_file():
        csv_path = json.loads(sub_meta.read_text(encoding="utf-8")).get("path", "—")

    sm = h0.get("s0_margin") or {}
    rm = h0.get("rb8_margin") or {}
    ceil = h0.get("acc_ceiling_if_404040") or {}

    text = f"""# 实验 40 · 结果登记

> 方案：[../方案.md](../方案.md) · **v0.2**  
> 状态：**已结案 · 算法线冻结** · `{doc.get('generated_at')}` · device=5070  
> 底座：R-B8 = 0.540 · 工程终态：`{final}`  
> 产物：`code/train_lab/out/5070_challenge_exp40_csv_harden_tta_accpaper/`

## 表 0 · 门控（v0.2）

| 项 | 规则 | 本跑 |
|----|------|------|
| MC 进候选 | nested≥0.540 ∧ ≥4/6 | {"是" if (arms.get("MC-B8") or {}).get("enter_candidate") else "否"} |
| TTA 进候选 | Δ≥+0.5pp ∧ ≥4/6 | {"是" if (arms.get("TTA-B8") or {}).get("enter_candidate") else "否"} |
| 终态 | nested 最高；&lt;1pp 更简 | `{final}` |
| 科学叙事 | KEEP_S0_discipline | `{doc.get("decision_science")}` |

## 表 1 · H0 风险基线

| 项 | 值 |
|----|-----|
| S0↔R-B8 Hamming | {h0.get("hamming_s0_rb8")} |
| S0 边际 L/R/Rest | {sm.get("L")} / {sm.get("R")} / {sm.get("Rest")} |
| R-B8 边际 L/R/Rest | {rm.get("L")} / {rm.get("R")} / {rm.get("Rest")} |
| test≈40/40/40 边际上限 | S0 {ceil.get("S0")} · R-B8 {ceil.get("R-B8")} |

## 表 2 · 加固臂

| 臂 | 嵌套 Val | vs R-B8 | n_folds_ge | 进终态候选 | 状态 |
|----|----------|---------|------------|------------|------|
{row("R-B8_raw")}
{row("MC-B8")}
{row("TTA-B8")}
{row("MC∘TTA")}

## 表 3 · 终态

| 项 | 值 |
|----|-----|
| `decision_engineering_final` | `{final}` |
| 候选集 | {eng.get("candidates")} |
| 平手池 | {eng.get("tie_pool")} |
| scored | `{json.dumps(eng.get("scored"), ensure_ascii=False)}` |
| 终态 CSV | `{csv_path}` |
| S0 回退 | `submission_exp34_e1f_a59_sens_full_20260902_1930.csv` |
| 算法线 | **冻结** |

## 风险注

- {(doc.get("risk_notes") or {}).get("mc")}
- {(doc.get("risk_notes") or {}).get("q2")}
- {(doc.get("risk_notes") or {}).get("prior")}

## Day-M 勾选

- [ ] Excel 指定集行回填  
- [ ] 数据集使用说明 · OpenBMI 申报句  
- [ ] 技术报告指定集节 + 乐观偏差专节  
- [ ] 重生成三件套；无 0.604/0.626 误引  
- [ ] 删除占位注 3  

## 决策

- **科学**：`KEEP_S0_discipline`  
- **工程终态**：`{final}` · {eng.get("label")}  
- **官方算法线宣布冻结**；剩余预算 → Day-M + 轨 C 扩人。
"""
    reg = scheme_doc() / "总结" / "结果登记表.md"
    reg.write_text(text, encoding="utf-8")
    print("updated", reg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-registry", action="store_true")
    ap.add_argument("--skip-margin", action="store_true")
    ap.add_argument("--skip-tta", action="store_true")
    ap.add_argument("--skip-decide", action="store_true")
    ap.add_argument("--skip-csv", action="store_true")
    args = ap.parse_args()

    if not args.skip_margin:
        _run("replay_margin_b8.py")
    if not args.skip_tta:
        _run("replay_tta_b8.py")
    if not args.skip_decide:
        _run("combine_and_decide.py")
    if not args.skip_csv:
        _run("write_final_csv.py")

    if args.update_registry or True:
        _update_registry()

    print("DONE exp40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
