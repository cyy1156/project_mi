"""补 Task 完成后，更新方案 24 登记表 §5.1 Task 列。"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUT_BASE = ROOT / "code" / "train_lab" / "out" / "5090_alg_incr_3s_hop100_accpaper"
REGISTRY = (
    ROOT
    / "资料"
    / "模型训练"
    / "24_旁路_算法增量_自适应窗_置信投票_t0加权_集成_openbmi_accpaper"
    / "总结"
    / "结果登记表.md"
)

MEMBERS = [
    ("shallow（=S3）", "shallow_openbmi_3s_hop100_balbatch_accpaper", "run_20260823_095327"),
    ("T-shallow（α=0.6）", "shallow_openbmi_3s_hop100_balbatch_accpaper", "run_20260823_123900"),
    ("eegnet", "eegnet_openbmi_3s_hop100_balbatch_accpaper", "run_20260823_131435"),
    ("conformer", "conformer_openbmi_3s_hop100_balbatch_accpaper", "run_20260823_135213"),
]


def _fmt_acc(summary_path: Path) -> str:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    m = float(s["test_acc_paper_mean"])
    sd = float(s["test_acc_paper_std"])
    return f"{m:.4f}±{sd:.3f}"


def _fmt_task(summary_path: Path) -> str:
    return f"**{_fmt_acc(summary_path)}**"


def main() -> None:
    rows: list[str] = []
    results: list[tuple[str, str, str, str]] = []
    for label, out_name, run_id in MEMBERS:
        run_dir = OUT_BASE / out_name / "openbmi_3s_hop100" / run_id
        task_summary = run_dir / "task" / "summary.json"
        three_summary = run_dir / "three" / "summary.json"
        if not task_summary.is_file():
            raise SystemExit(f"missing task summary: {task_summary}")
        if not three_summary.is_file():
            raise SystemExit(f"missing three summary: {three_summary}")
        three_s = _fmt_acc(three_summary)
        if label.startswith("shallow（") or label == "conformer":
            three_s = f"**{three_s}**"
        task_s = _fmt_task(task_summary)
        results.append((label, run_id, three_s, task_s, str(run_dir)))
        rows.append(f"| {label} | — | {three_s} | {task_s} | `{run_id}` |")

    text = REGISTRY.read_text(encoding="utf-8")
    block = "\n".join(rows)
    new_section = (
        "### 5.1 成员\n\n"
        "| 成员 | 参数量 | Three | Task | run |\n"
        "|------|--------|-------|------|-----|\n"
        f"{block}\n"
    )
    text, n = re.subn(
        r"### 5\.1 成员\n\n\| 成员 \| 参数量 \| Three \| Task \| run \|\n\|[-| ]+\|\n(?:\|[^\n]+\|\n)+",
        new_section,
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit("failed to patch §5.1 table")

    log_line = (
        f"| {date.today().isoformat()} | E1f 四成员补 Task 头（5090 · skip-three） | "
        + " · ".join(f"{lab} Task={tsk}" for lab, _, _, tsk, _ in results)
        + " |"
    )
    if log_line not in text:
        text = text.replace(
            "| 2026-08-25 | W 回放 bugfix 重跑 |",
            f"{log_line}\n| 2026-08-25 | W 回放 bugfix 重跑 |",
        )

    REGISTRY.write_text(text, encoding="utf-8")
    print(f"updated {REGISTRY}")
    for label, run_id, three_s, task_s, run_dir in results:
        print(f"  {label} {run_id}: Three {three_s} · Task {task_s}  ({run_dir})")


if __name__ == "__main__":
    main()
