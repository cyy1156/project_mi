# -*- coding: utf-8 -*-
"""Write exp31 registry + subject analysis (one-shot)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARIES = {
    "syj0828": REPO
    / "experiment_game/data/subjects/syj0828/models/ft_runs"
    / "20260829_155626_syj0828_e1f_task_leave_next_f5_summary.json",
    "xjh0828": REPO
    / "experiment_game/data/subjects/xjh0828/models/ft_runs"
    / "20260829_155712_xjh0828_e1f_task_leave_next_f5_summary.json",
}
OUT_DIR = REPO / "资料/模型训练/31_旁路_被试LeaveNext_F5读出_syj_fnz_openbmi_accpaper"

def fmt_lab(bl: dict, name: str) -> str:
    b = (bl or {}).get(name) or {}
    n = int(b.get("n") or 0)
    ok = int(b.get("ok") or 0)
    pts = float(b.get("points") or 0)
    pmax = float(b.get("points_max") or 0)
    acc = (ok / n * 100) if n else float("nan")
    return f"{ok}/{n}={acc:.1f}% ({pts:.1f}/{pmax:.1f})"

def main() -> None:
    (OUT_DIR / "总结").mkdir(parents=True, exist_ok=True)
    out: list[str] = [
        "# 实验 31 · 结果登记（Leave-Next + F5 · syj0828 / xjh0828）",
        "",
        "> 日期：2026-08-29  ",
        "> **状态：正式现行（已 promote 至 `subjects/*/models/current`）**  ",
        "> 通道序：`FC3, C3, CP3, CZ, CPZ, FC4, C4, CP4`  ",
        "> 读出 **F5**：因果平滑 lookback=2 + 试次多数票（L/R +1，Rest +0.5）  ",
        "> FT 底座：E1f 四成员中的 shallow fold0 `task+three`；Leave-Next；前段 replay=0.10  ",
        "",
        "## 0. 现行 vs 历史",
        "",
        "| 标记 | 说明 |",
        "|------|------|",
        "| **现行** | `ft_runs/20260829_155626_*`（syj）、`20260829_155712_*`（fnz）→ 已写入 `models/current` |",
        "| **历史保留（勿删）** | 更早 Leave-Next（如 `20260829_001910` / `002039`）；旧 current → `models/current_archived/*_pre_f5_leave_next/`；各 run 内 `RUN_STATUS.json` |",
        "| **旁路不覆盖** | 实验 27 / 29 / 30 登记表保持原样；被试 FT **以本实验为准** |",
        "",
        "指针：`experiment_game/data/subjects/{id}/models/WEIGHTS_STATUS.md` · "
        "`experiment_game/data/subjects/_promotion_f5_leave_next_20260829.json`",
        "",
    ]

    for sid, sp in SUMMARIES.items():
        d = json.loads(sp.read_text(encoding="utf-8"))
        rel = sp.relative_to(REPO).as_posix()
        out.append(f"## {sid}")
        out.append("")
        out.append(f"- JSON：`{rel}`")
        if sid == "xjh0828":
            out.append("- 末档门控 **FAIL**，已按「全部替换」**强制晋升** current")
        else:
            out.append("- 门控各档 **PASS**")
        out.append("")
        out.append(
            "| R | hold | 窗acc | FT F5 MI | E1f零样本 MI | 总分 FT | Left | Right | Rest | PASS |"
        )
        out.append(
            "|---|------|-------|----------|--------------|---------|------|-------|------|------|"
        )
        for r in d["rows"]:
            ft = r["f5_ft"]
            e1 = r.get("f5_base_e1f") or {}
            bl = ft.get("by_label") or {}
            hold = r["heldout"].split("_")[1]
            e_mi = float(e1.get("mi_acc") or 0) * 100
            out.append(
                f"| {r['r_stage']} | {hold} | {r['heldout_acc']:.3f} | "
                f"{ft['mi_acc']*100:.1f}% | {e_mi:.1f}% | "
                f"{ft['score']:.1f}/{float(ft.get('score_max') or 45):.0f} | "
                f"{fmt_lab(bl, 'Left')} | {fmt_lab(bl, 'Right')} | {fmt_lab(bl, 'Rest')} | "
                f"{r['release_pass']} |"
            )
        out.append("")

        an_dir = REPO / f"experiment_game/data/subjects/{sid}/analysis"
        an_dir.mkdir(parents=True, exist_ok=True)
        an_lines = [
            f"# {sid} · Leave-Next + F5（现行 2026-08-29）",
            "",
            "**状态：现行。** 旧 Leave-Next / 旧 current 已标记 historical 并归档，**未删除**。",
            "",
            f"- 汇总：`{rel}`",
            f"- current：`experiment_game/data/subjects/{sid}/models/current`",
            f"- 权重说明：`experiment_game/data/subjects/{sid}/models/WEIGHTS_STATUS.md`",
            f"- 旧 current：`experiment_game/data/subjects/{sid}/models/current_archived/`",
            "",
            "详见 `资料/模型训练/31_旁路_被试LeaveNext_F5读出_syj_fnz_openbmi_accpaper/总结/结果登记表.md`。",
            "",
        ]
        (an_dir / "leave_next_f5_20260829.md").write_text(
            "\n".join(an_lines), encoding="utf-8"
        )

    reg = OUT_DIR / "总结" / "结果登记表.md"
    reg.write_text("\n".join(out) + "\n", encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# 实验 31 · 被试 Leave-Next + F5 读出（syj0828 / xjh0828）\n\n"
        "- **现行**被试 FT 权重与登记：见 `总结/结果登记表.md`\n"
        "- 旧 Leave-Next 实验保留在各 `ft_runs/`，标记 `historical`，不删除\n"
        "- 相关冻结：`experiment_game/docs/框架冻结确认_20260829.md`（F5/F7）\n",
        encoding="utf-8",
    )
    print("wrote", reg)

if __name__ == "__main__":
    main()
