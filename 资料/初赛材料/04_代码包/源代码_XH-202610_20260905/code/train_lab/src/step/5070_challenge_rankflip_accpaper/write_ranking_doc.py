# -*- coding: utf-8 -*-
"""轨 R：写跨域排名对照文档。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from exp35_config import scheme_doc, ranking_doc_permanent

OPENBMI_3S = [
    ("T-shallow", 0.5886),
    ("shallow S3", 0.5839),
    ("conformer", 0.5767),
    ("eegnet", 0.5629),
]
OPENBMI_2S = [
    ("shallow", 0.5404),
    ("conformer", 0.5375),
    ("eegnet", 0.5322),
]
A59 = [
    ("conformer", 0.511),
    ("shallow", 0.453),
    ("shallow_b", 0.450),
    ("eegnet", 0.386),
]
B8_SCRATCH = [
    ("conformer", 0.502),
    ("shallow≈", 0.450),
    ("eegnet", 0.362),
]
B8_FT = [
    ("shallow_b", 0.528),
    ("conformer", 0.527),
    ("shallow", 0.523),
    ("eegnet", 0.507),
]


def _rank_table(rows: list[tuple[str, float]]) -> str:
    lines = ["| 名次 | 成员 | Acc |", "|------|------|-----|"]
    for i, (n, a) in enumerate(rows, 1):
        lines.append(f"| {i} | {n} | {a:.3f} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-json", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = scheme_doc() / "总结" / f"跨域三分类成员排名对照表_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    e1f_weights = ""
    if args.replay_json and args.replay_json.is_file():
        payload = json.loads(args.replay_json.read_text(encoding="utf-8"))
        f0 = (payload.get("results") or {}).get("F0")
        if f0:
            lines = [
                "",
                "## Exp35 F0 每折权重（复现）",
                "",
                "| fold | w_shallow | w_shallow_b | w_eegnet | w_conformer | val_acc |",
                "|------|-----------|-------------|----------|-------------|---------|",
            ]
            for fr in f0.get("folds") or []:
                cfg = fr["config"]
                names = cfg["member_names"]
                w = {n: cfg["weights"][i] for i, n in enumerate(names)}
                lines.append(
                    f"| {fr['fold']} | {w.get('shallow', 0):.2f} | {w.get('shallow_b', 0):.2f} | "
                    f"{w.get('eegnet', 0):.2f} | {w.get('conformer', 0):.2f} | "
                    f"{fr['val_metrics']['acc']:.3f} |"
                )
            e1f_weights = "\n".join(lines)

    md = f"""# 跨域三分类成员排名对照表

> **常驻文档**（轨 R 升格 · Exp35 v0.2）· 生成：{datetime.now().isoformat(timespec="seconds")}  
> 口径：OpenBMI=Acc_paper Three；官方=Exp34 LOSO6 Val Acc  
> **禁止**把不同协议绝对值混比夺冠；本表只比**名次形态**。  
> 维护脚本：`code/train_lab/src/step/5070_challenge_rankflip_accpaper/write_ranking_doc.py`

## OpenBMI 3s hop100（方案 24/26）

{_rank_table(OPENBMI_3S)}

融合 E1f test **0.617**。

## OpenBMI 2s（正式十一模型 Three）

{_rank_table(OPENBMI_2S)}

## 官方 A59 从零（Exp34）

{_rank_table(A59)}

融合 E1f-A59 Val **0.558**。

## 官方 B8 从零 / FT（Exp34）

### scratch

{_rank_table(B8_SCRATCH)}

### FT（OpenBMI 热启）

{_rank_table(B8_FT)}

## 协议差异矩阵

| 项 | OpenBMI 3s | 官方 A59 | 官方 B8 |
|----|------------|----------|---------|
| 被试划分 | Subject 5-fold | LOSO6（S01–S06） | 同左 |
| 时间聚合 | hop100 多窗 | 1 trial=1 窗 | 同左 |
| 通道 | 8 | 59 | 8（Pz→CPz） |
| 训练起点 | 从零 / 预训练库 | 从零 | 从零或 OpenBMI FT |

## 一句话

OpenBMI：**shallow 家族 > conformer > eegnet**；  
官方从零：**conformer ≫ shallow > eegnet**；  
官方 8ch FT：排名被拉回接近打平。
{e1f_weights}
"""
    out.write_text(md, encoding="utf-8")
    stable = scheme_doc() / "总结" / "跨域三分类成员排名对照表.md"
    stable.write_text(md, encoding="utf-8")
    permanent = ranking_doc_permanent()
    permanent.parent.mkdir(parents=True, exist_ok=True)
    permanent.write_text(md, encoding="utf-8")
    print("wrote", out)
    print("wrote", stable)
    print("wrote", permanent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
