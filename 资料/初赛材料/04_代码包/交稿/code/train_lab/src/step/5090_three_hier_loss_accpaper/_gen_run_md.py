"""One-off: generate runs/5090_three_hier_loss_accpaper MD from chain summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))

from md_fold_detail import task_fold_md_lines, three_fold_md_lines  # noqa: E402

OUT = REPO / "code/train_lab/out/5090_three_hier_loss_accpaper"
RUNS = REPO / "资料/模型训练/runs/5090_three_hier_loss_accpaper"
RUNS.mkdir(parents=True, exist_ok=True)
data_dir = REPO / "code/preprocess_lab/out/openbmi_2s_hop100"

CHAIN = [
    ("S0_three", "shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper", "run_20260816_121646", "three", "S0 plain CE · shallow"),
    ("H1_three", "shallow_hier_h1_openbmi_2s_hop100_balbatch_accpaper", "run_20260816_133223", "three", "H1 CE+MI+LR · shallow"),
    ("H2_three", "shallow_hier_h2_openbmi_2s_hop100_balbatch_accpaper", "run_20260816_145815", "three", "H2 H1+margin+idle_suppress · shallow"),
    ("H3_three", "shallow_hier_h3_openbmi_2s_hop100_balbatch_accpaper", "run_20260816_165447", "three", "H3 H2 (+trial_cons later) · shallow"),
    ("T0_task", "shallow_hier_s0_openbmi_2s_hop100_balbatch_accpaper", "run_20260816_184618", "task", "S0 plain CE · shallow (Task T0)"),
]

for step, out_name, run_id, stage, note in CHAIN:
    out_root = OUT / out_name / "openbmi_2s_hop100" / run_id
    meta = json.loads((out_root / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads((out_root / stage / "summary.json").read_text(encoding="utf-8"))
    stamp = run_id.replace("run_", "")
    md_dir = RUNS / f"{stamp}_{out_name}"
    md_path = md_dir / f"{out_name}五折实验记录.md"
    md_dir.mkdir(parents=True, exist_ok=True)
    hp = summary["hparams"]
    lines = [
        f"# 被试独立五折实验记录（{stamp} / {out_name}）",
        "",
        f"- 开始：`{meta.get('stamp', stamp)}` · chain step **{step}**",
        "- device：`cuda` · **train_mode=`fast`**（5090 全量五折）",
        "- 训练设备：**NVIDIA RTX 5090**（32GB · conda cyy · PyTorch 2.11+cu128）",
        f"- data：`{data_dir}`（OpenBMI hop100 · blocks=EEG_MI_train）",
        f"- protocol：`{hp['protocol']}` | early_stop=**Acc_paper** | **balbatch** | no_rap",
        f"- model：`{out_name}` | {note}",
        f"- 权重：`{out_root}`",
        f"- shared hp：`{hp}`",
        "",
        "---",
        "",
        "## 最终结论（主报 Acc_paper）",
        "",
    ]
    if stage == "task":
        lines += [
            "### Task",
            f"- Val Acc_paper：`{summary['val_acc_paper_mean']:.4f} ± {summary['val_acc_paper_std']:.4f}`",
            f"- Test Acc_paper：`{summary['test_acc_paper_mean']:.4f} ± {summary['test_acc_paper_std']:.4f}`",
            f"- Test BalAcc_maj：`{summary['test_balacc_maj_mean']:.4f} ± {summary['test_balacc_maj_std']:.4f}`",
            f"- Test 窗级 BalAcc（附报）：`{summary['test_window_balacc_mean']:.4f} ± {summary['test_window_balacc_std']:.4f}`",
            "",
            *task_fold_md_lines(summary["folds"]),
        ]
    else:
        lines += [
            "### Three",
            f"- Val Acc_paper：`{summary['val_acc_paper_mean']:.4f} ± {summary['val_acc_paper_std']:.4f}`",
            f"- Test Acc_paper：`{summary['test_acc_paper_mean']:.4f} ± {summary['test_acc_paper_std']:.4f}`",
            f"- Test BalAcc_maj：`{summary['test_balacc_maj_mean']:.4f} ± {summary['test_balacc_maj_std']:.4f}`",
            f"- Test 窗级 BalAcc（附报）：`{summary['test_window_balacc_mean']:.4f} ± {summary['test_window_balacc_std']:.4f}`",
            "",
            *three_fold_md_lines(summary["folds"]),
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", md_path.relative_to(REPO))
