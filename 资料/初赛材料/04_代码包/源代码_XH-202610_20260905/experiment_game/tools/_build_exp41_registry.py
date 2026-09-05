"""Build Exp41 registry markdown from cohort_index + live ramp counts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(r"D:\MI")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiment_game" / "tools"))

from run_leave_next_e1f_task_ramp import (  # noqa: E402
    INCLUDE_FT_INELIGIBLE,
    SUBJECTS_ROOT,
    _list_v3_sessions,
    _ramp_for_subject,
)

EXP41 = (
    _REPO
    / "资料"
    / "模型训练"
    / "41_旁路_真人被试LeaveNext_F5全队列统一_openbmi_accpaper"
)
OUT = EXP41 / "总结" / "结果登记表.md"
INDEX = EXP41 / "总结" / "cohort_index.json"

# Human notes for structure quirks
NOTES = {
    "syj0828": "排除 v4 ws01；命名 ws*",
    "fnz0828": "排除 v4 ws01；R5=ws07",
    "cyy0830": "v3 w02/w03 现因电极规则放宽可纳入",
    "cjf0831": "v3 w04/w05 现因电极规则放宽可纳入",
    "npl0831": "v3 w01 现因电极规则放宽可纳入",
    "wzr0830": "v3 w05 现因电极规则放宽可纳入",
    "ycx0831": "跳半场 w06；末档 w07",
    "ytl0901": "w02+w03 合并",
    "zyj0902": "跳 excluded w03（无 eeg）；末档 w07",
    "lsm0903": "六档至 w07；w05 半场嫌疑（分数字段空）",
    "lsy0903": "含原电极告警 w05",
    "djh0902": "多数场曾标电极饱和，现均默认纳入",
    "zcy0902": "含原电极告警 w04/w05",
    "fnz0830": "标准 RAMP_W",
    "xj0830": "标准 RAMP_W",
    "lmh0904": "2026-09-04 新增；标准 RAMP_W（w01–w06）",
    "lmy0904": "2026-09-04 新增；跳无 eeg 的 w02/w03；爬坡 w01→w04…w08",
}


def pct(x) -> str:
    return f"{100 * float(x):.1f}%" if isinstance(x, (int, float)) else "-"


def f3(x) -> str:
    return f"{float(x):.3f}" if isinstance(x, (int, float)) else "-"


def main() -> None:
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    # enrich with live ramp
    for item in idx:
        sid = item["subject_id"]
        by = _list_v3_sessions(sid)
        try:
            ramp = _ramp_for_subject(sid, by)
            item["live_n_ramp"] = len(ramp)
            item["live_holds"] = [h for _, h, _ in ramp]
        except Exception:
            item["live_n_ramp"] = 0
            item["live_holds"] = []
        item["note"] = NOTES.get(sid, "")

    ranked = sorted(
        [x for x in idx if x.get("last")],
        key=lambda x: -float(x["last"]["mi"]),
    )
    mis = [float(x["last"]["mi"]) for x in ranked]
    n_pass = sum(1 for x in ranked if x["last"].get("pass"))
    mean_mi = sum(mis) / len(mis) if mis else float("nan")

    n = len(ranked)
    lines: list[str] = []
    lines += [
        "# 实验 41 · 结果登记（真人被试 Leave-Next + F5 全队列）",
        "",
        "> 日期：2026-09-05 · **结构统一完成（含 lmh0904 / lmy0904）**  ",
        "> 主读口径：各被试 **最新 all4** `*leave_next*all4*f5_summary.json` · F5 试次 MI  ",
        "> 机读：[`cohort_index.json`](./cohort_index.json) · 全明细：[`_generated_cohort.md`](./_generated_cohort.md)  ",
        "> 方案：[`../方案.md`](../方案.md)",
        "",
        "## 0. 一句话",
        "",
        f"**{n} 人真人队列已统一结构与末档读数；末档 FT F5 MI 均值 {mean_mi:.1%}（{n_pass}/{n} 末档 PASS）。** "
        "官方 Exp34–40 交卷线不变；本表为自采监测总账。",
        "",
        "## 1. 结构总表（session / 爬坡）",
        "",
        "| 被试 | Leave-Next 可用键 | 落盘 nR | 当前可重跑档 | 结构备注 |",
        "|------|-------------------|---------|--------------|----------|",
    ]
    for x in sorted(idx, key=lambda z: z["subject_id"]):
        keys = ",".join(x.get("leave_next_keys") or [])
        lines.append(
            f"| {x['subject_id']} | `{keys}` | {x.get('n_rounds', 0)} | "
            f"{x.get('live_n_ramp', 0)} | {x.get('note') or '—'} |"
        )

    lines += [
        "",
        "> **落盘 nR**：已写入 summary 的档数（主读依据）。  ",
        "> **当前可重跑档**：按此刻 `index.json` + 纳入规则（**电极饱和不排除**）。",
        "",
        "## 2. 末档排名（按 FT F5 MI ↓）",
        "",
        "| 名次 | 被试 | 末档 hold | 三分类窗smooth | 三分类窗raw | FT F5 MI | E1f零样本 MI | 总分 | PASS | summary |",
        "|------|------|-----------|----------------|-------------|----------|--------------|------|------|----------|",
    ]
    for i, x in enumerate(ranked, 1):
        last = x["last"]
        win_s = last.get("win_smooth", last.get("win"))
        win_r = last.get("win_raw")
        lines.append(
            f"| {i} | **{x['subject_id']}** | {last['hold']} | {f3(win_s)} | {f3(win_r)} | "
            f"**{pct(last['mi'])}** | {pct(last.get('e1f_mi'))} | "
            f"{last.get('score')}/{last.get('score_max')} | {last.get('pass')} | "
            f"`{x.get('summary')}` |"
        )

    lines += [
        "",
        "> **三分类窗** = heldout 窗级 Rest+Left+Right；**FT F5 MI** = 试次级仅 L+R。",
        "",
        f"- 队列均值 MI = **{mean_mi:.1%}**；中位 ≈ **{sorted(mis)[len(mis)//2]:.1%}**；"
        f"末档 PASS = **{n_pass}/{n}**",
        "- 最强：**syj0828**（91.7%）；最弱：**fnz0830**（8.6%）",
        "- 相对零样本：多数人 FT≥E1f；**fnz0828** 零样本极低（2.8%）但 FT 仍仅 33.3%",
        "",
        "## 3. 结构特例速查",
        "",
        "| 规则 | 适用 |",
        "|------|------|",
        "| 排除全部 v4 | 全员 |",
        "| **电极 CZ/CPZ 饱和不排除** | 全员（2026-09-04） |",
        "| 仍排除 `record_excluded` / 无 eeg | 如 zyj0902 w03；lmy0904 w02/w03 |",
        "| 命名 `ws*` + 跳过首场 v4 | syj0828 · fnz0828 |",
        "| 合并场 `w02+w03` | ytl0901 |",
        "| 跳半场 / excluded | ycx0831 跳 w06；zyj0902 跳 w03；lmy0904 跳 w02/w03 |",
        "| 六档至 w07 | lsm0903 |",
        "| 2026-09-04/05 新增 | lmh0904 · lmy0904 |",
        "",
        "## 4. 与历史实验对齐",
        "",
        "| 来源 | 本表用法 |",
        "|------|----------|",
        "| Exp31 promote（syj/fnz e1f_task） | **不覆盖**本表 all4 主读；current 权重仍以 31 为准 |",
        "| Exp31 §A lsy/lsm | 已并入本表主读（同 JSON） |",
        "| Exp33 so vs all4 | 二人证据仍有效；全队列监测改读本表 |",
        "| Exp34–40 | 官方交卷隔离 |",
        "",
        "## 5. 结论",
        "",
        f"1. **结构已统一**：{n} 人 session 键、爬坡特例、强制纳入名单、落盘 JSON 均有单一入口。  ",
        "2. **表现分化大**：末档 MI 从 8.6%–91.7%；约六成末档过门控，但不等于都可 promote。  ",
        "3. **重跑缺口**：个别被试当前可重跑档短于历史落盘——若需刷新，须先修 index / 纳入策略。  ",
        "4. **下一步（非本实验强制）**：按材料需要挑选末档 PASS 且 Right 不崩的被试 promote；"
        "继续自采补齐弱被试，而不是改官方 S0。",
        "",
        "## 6. 再生",
        "",
        "```text",
        "python experiment_game/tools/_build_exp41_cohort.py",
        "python experiment_game/tools/_build_exp41_registry.py",
        "```",
        "",
        "分档全表与逐场 index 见 [`_generated_cohort.md`](./_generated_cohort.md)。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
