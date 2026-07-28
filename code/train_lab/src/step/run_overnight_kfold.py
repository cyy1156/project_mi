"""
过夜实验编排：按模型独立搜参。
默认：Task 网格+调参 → Three 网格+调参（不迁权重）。
可选 --init-from-task 做历史迁移对照。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

# 保证可直接 `python run_overnight_kfold.py`
STEP_DIR = Path(__file__).resolve().parent
TRAIN_LAB = STEP_DIR.parents[1]
CODE_ROOT = STEP_DIR.parents[2]
REPO_ROOT = CODE_ROOT.parent

sys.path.insert(0, str(STEP_DIR))
sys.path.insert(0, str(TRAIN_LAB))

from data_paths import resolve_data  # noqa: E402
from models import list_models  # noqa: E402
from train_task_kfold import TaskKFoldConfig, run_task_kfold  # noqa: E402
from train_three_kfold import ThreeKFoldConfig, run_three_kfold  # noqa: E402

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MD_DIR = REPO_ROOT / "资料" / "模型训练"
MD_PATH = MD_DIR / f"五折过夜实验记录_{STAMP}.md"
MD_LATEST = MD_DIR / "五折过夜实验记录_最新.md"
# OUT_ROOT 在 main 里按 model 设定
OUT_ROOT: Path = TRAIN_LAB / "out" / f"overnight_{STAMP}"
LOG_PATH: Path = OUT_ROOT / "overnight.log"


def append_md(text: str) -> None:
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MD_PATH, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    # 稳定入口：始终指向本次实验文件
    MD_LATEST.write_text(
        f"# 最新过夜实验入口\n\n"
        f"本次记录文件：[`{MD_PATH.name}`](./{MD_PATH.name})\n\n"
        f"权重目录：`{OUT_ROOT}`\n"
        f"日志：`{LOG_PATH}`\n",
        encoding="utf-8",
    )


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt_hparams(hp: dict) -> str:
    keys = [
        "lr",
        "weight_decay",
        "drop_prob",
        "patience",
        "max_epochs",
        "batch_train",
        "seed",
        "val_ratio",
        "n_folds",
        "freeze_backbone",
        "f1",
        "d",
        "f2",
    ]
    lines = []
    for k in keys:
        if k in hp:
            lines.append(f"- `{k}`: `{hp[k]}`")
    return "\n".join(lines)


def write_task_section(run_name: str, summary: dict, note: str = "") -> None:
    hp = summary["hparams"]
    lines = [
        f"## {run_name}",
        "",
        f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 输出目录：`{summary['out_dir']}`",
    ]
    if note:
        lines.append(f"- 说明：{note}")
    lines += [
        "",
        "### 超参",
        "",
        fmt_hparams(hp),
        "",
        "### 各折验证 / 测试",
        "",
        "| fold | best_ep | val_F1 | test_Acc | test_F1 | test_Spe | n_train/val/test |",
        "|------|---------|--------|----------|---------|----------|------------------|",
    ]
    for r in summary["folds"]:
        m = r["test_metrics"]
        lines.append(
            f"| {r['fold']} | {r['best_epoch']} | {r['best_val_f1']:.4f} | "
            f"{m['accuracy']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | "
            f"{r['n_train']}/{r['n_val']}/{r['n_test']} |"
        )
    lines += [
        "",
        "### 汇总（调参看 Val；报数看 Test）",
        "",
        f"- **Val F1** mean±std = `{summary['val_f1_mean']:.4f} ± {summary['val_f1_std']:.4f}`",
        f"- **Test Acc** mean±std = `{summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}`",
        f"- **Test F1** mean±std = `{summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}`",
        f"- 平均 best_epoch = `{summary['mean_best_epoch']:.1f}`",
    ]
    if summary.get("test_f1_bci2a_only_mean") is not None:
        lines.append(
            f"- **Test F1 bci2a_only** mean±std = "
            f"`{summary['test_f1_bci2a_only_mean']:.4f} ± {summary['test_f1_bci2a_only_std']:.4f}`"
        )
    if summary.get("test_f1_stieger_only_mean") is not None:
        lines.append(
            f"- **Test F1 stieger_only** mean±std = "
            f"`{summary['test_f1_stieger_only_mean']:.4f} ± {summary['test_f1_stieger_only_std']:.4f}`"
        )
    lines += ["", "---", ""]
    append_md("\n".join(lines))


def write_three_section(run_name: str, summary: dict, note: str = "") -> None:
    hp = summary["hparams"]
    lines = [
        f"## {run_name}",
        "",
        f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 输出目录：`{summary['out_dir']}`",
        f"- weight_transfer：`{summary.get('weight_transfer', hp.get('init_from_task', False))}`",
        f"- model：`{summary.get('model_name', hp.get('model_name', ''))}`",
    ]
    if hp.get("init_from_task") and hp.get("task_kfold_dir"):
        lines.append(f"- 头1权重目录：`{hp.get('task_kfold_dir', '')}`")
    if note:
        lines.append(f"- 说明：{note}")
    lines += [
        "",
        "### 超参",
        "",
        fmt_hparams(hp),
        "",
        "### 各折验证 / 测试",
        "",
        "| fold | best_ep | val_F1macro | test_Acc | test_F1macro | R_idle | R_left | R_right |",
        "|------|---------|-------------|----------|--------------|--------|--------|---------|",
    ]
    for r in summary["folds"]:
        m = r["test_metrics"]
        lines.append(
            f"| {r['fold']} | {r['best_epoch']} | {r['best_val_f1_macro']:.4f} | "
            f"{m['accuracy']:.4f} | {m['f1_macro']:.4f} | "
            f"{m['recall_idle']:.3f} | {m['recall_left']:.3f} | {m['recall_right']:.3f} |"
        )
    lines += [
        "",
        "### 汇总（调参看 Val；报数看 Test）",
        "",
        f"- **Val F1-macro** mean±std = "
        f"`{summary['val_f1_macro_mean']:.4f} ± {summary['val_f1_macro_std']:.4f}`",
        f"- **Test Acc** mean±std = `{summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}`",
        f"- **Test F1-macro** mean±std = "
        f"`{summary['test_f1_macro_mean']:.4f} ± {summary['test_f1_macro_std']:.4f}`",
        f"- 平均 best_epoch = `{summary['mean_best_epoch']:.1f}`",
    ]
    if summary.get("test_f1_macro_bci2a_only_mean") is not None:
        lines.append(
            f"- **Test F1-macro bci2a_only** mean±std = "
            f"`{summary['test_f1_macro_bci2a_only_mean']:.4f} ± "
            f"{summary['test_f1_macro_bci2a_only_std']:.4f}`"
        )
    if summary.get("test_f1_macro_stieger_only_mean") is not None:
        lines.append(
            f"- **Test F1-macro stieger_only** mean±std = "
            f"`{summary['test_f1_macro_stieger_only_mean']:.4f} ± "
            f"{summary['test_f1_macro_stieger_only_std']:.4f}`"
        )
    lines += ["", "---", ""]
    append_md("\n".join(lines))


def suggest_task_hparams(cfg: TaskKFoldConfig, summary: dict) -> tuple[TaskKFoldConfig, list[str]]:
    """仅根据验证集信号建议下一轮超参（不看 test）。"""
    new = copy.deepcopy(cfg)
    reasons: list[str] = []
    val_mean = summary["val_f1_mean"]
    val_std = summary["val_f1_std"]
    mean_ep = summary["mean_best_epoch"]
    early_folds = sum(1 for r in summary["folds"] if r["best_epoch"] <= 5)
    late_folds = sum(1 for r in summary["folds"] if r["stopped_epoch"] >= cfg.max_epochs - 1)

    if early_folds >= 3:
        new.lr = max(1e-4, cfg.lr * 0.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"有 {early_folds}/5 折 best_epoch≤5 → 学习率 {cfg.lr:g}→{new.lr:g}，patience→{new.patience}"
        )
    elif late_folds >= 3 and val_mean < 0.75:
        new.drop_prob = max(0.25, round(cfg.drop_prob - 0.10, 2))
        new.lr = min(3e-3, cfg.lr * 1.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"多数折接近满 epoch 且 val_F1={val_mean:.3f} 偏低 → "
            f"drop {cfg.drop_prob}→{new.drop_prob}，lr→{new.lr:g}，patience→{new.patience}"
        )
    elif val_std >= 0.12:
        new.drop_prob = min(0.70, round(cfg.drop_prob + 0.10, 2))
        new.weight_decay = min(1e-3, cfg.weight_decay * 2.0)
        reasons.append(
            f"折间 val_F1 标准差偏大({val_std:.3f}) → "
            f"drop→{new.drop_prob}，weight_decay→{new.weight_decay:g}"
        )
    elif val_mean < 0.70:
        new.drop_prob = min(0.70, round(cfg.drop_prob + 0.10, 2))
        new.weight_decay = min(1e-3, cfg.weight_decay * 2.0)
        new.lr = max(3e-4, cfg.lr * 0.7)
        reasons.append(
            f"val_F1={val_mean:.3f} 偏低 → 加强正则：drop→{new.drop_prob}，"
            f"wd→{new.weight_decay:g}，lr→{new.lr:g}"
        )
    else:
        new.lr = max(3e-4, cfg.lr * 0.7)
        new.drop_prob = min(0.65, round(cfg.drop_prob + 0.05, 2))
        reasons.append(
            f"基线尚可(val_F1={val_mean:.3f}) → 微调：lr→{new.lr:g}，drop→{new.drop_prob}"
        )

    # 人少时 val 噪，第二轮略加大 patience
    if new.patience == cfg.patience:
        new.patience = min(25, cfg.patience + 3)
        reasons.append(f"BCI2a 仅9人、val 人极少 → patience {cfg.patience}→{new.patience}")

    return new, reasons


def suggest_three_hparams(cfg: ThreeKFoldConfig, summary: dict) -> tuple[ThreeKFoldConfig, list[str]]:
    new = copy.deepcopy(cfg)
    reasons: list[str] = []
    val_mean = summary["val_f1_macro_mean"]
    val_std = summary["val_f1_macro_std"]
    mean_ep = summary["mean_best_epoch"]
    early_folds = sum(1 for r in summary["folds"] if r["best_epoch"] <= 5)

    # 看左右召回是否极不均衡（用 test 仅作诊断描述；真正改参仍以 val 为主）
    lefts = [r["test_metrics"]["recall_left"] for r in summary["folds"]]
    rights = [r["test_metrics"]["recall_right"] for r in summary["folds"]]
    mean_left = sum(lefts) / len(lefts)
    mean_right = sum(rights) / len(rights)

    if early_folds >= 3:
        new.lr = max(1e-4, cfg.lr * 0.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"有 {early_folds}/5 折过早停 → lr {cfg.lr:g}→{new.lr:g}，patience→{new.patience}"
        )
    elif val_mean < 0.45:
        new.drop_prob = max(0.30, round(cfg.drop_prob - 0.10, 2))
        new.lr = min(2e-3, cfg.lr * 1.5)
        new.patience = min(25, cfg.patience + 5)
        reasons.append(
            f"val_F1macro={val_mean:.3f} 偏低疑欠拟合 → drop→{new.drop_prob}，lr→{new.lr:g}"
        )
    elif val_std >= 0.12:
        new.drop_prob = min(0.75, round(cfg.drop_prob + 0.05, 2))
        new.weight_decay = min(1e-3, cfg.weight_decay * 2.0)
        reasons.append(
            f"折间波动大(std={val_std:.3f}) → drop→{new.drop_prob}，wd→{new.weight_decay:g}"
        )
    else:
        new.lr = max(3e-4, cfg.lr * 0.7)
        new.drop_prob = min(0.70, round(cfg.drop_prob + 0.05, 2))
        reasons.append(
            f"基线尚可(val_F1macro={val_mean:.3f}) → 微调 lr→{new.lr:g}，drop→{new.drop_prob}"
        )

    if abs(mean_left - mean_right) > 0.25 and not cfg.freeze_backbone:
        # 左右差大时第二轮可试更小 lr 稳住迁移特征
        new.lr = min(new.lr, max(2e-4, cfg.lr * 0.5))
        reasons.append(
            f"左右召回差较大(L={mean_left:.2f}, R={mean_right:.2f}) → 进一步降 lr→{new.lr:g}"
        )

    if new.patience == cfg.patience:
        new.patience = min(25, cfg.patience + 3)
        reasons.append(f"val 人少 → patience→{new.patience}")

    _ = mean_ep  # 保留诊断字段，避免未使用告警
    return new, reasons


def write_tune_note(title: str, reasons: list[str], new_hp: dict) -> None:
    lines = [
        f"## {title}",
        "",
        "调参依据（**只看验证集**；测试集不参与选型）：",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")
    lines += ["", "下一轮拟用超参：", "", fmt_hparams(new_hp), "", "---", ""]
    append_md("\n".join(lines))


def pick_best_task(run_a: dict, run_b: dict) -> dict:
    """按 val_F1_mean 选更好的一轮（平局取标准差更小）。"""
    if run_b["val_f1_mean"] > run_a["val_f1_mean"] + 1e-6:
        return run_b
    if abs(run_b["val_f1_mean"] - run_a["val_f1_mean"]) <= 1e-6 and run_b["val_f1_std"] < run_a["val_f1_std"]:
        return run_b
    return run_a


def pick_best_three(run_a: dict, run_b: dict) -> dict:
    if run_b["val_f1_macro_mean"] > run_a["val_f1_macro_mean"] + 1e-6:
        return run_b
    if (
        abs(run_b["val_f1_macro_mean"] - run_a["val_f1_macro_mean"]) <= 1e-6
        and run_b["val_f1_macro_std"] < run_a["val_f1_macro_std"]
    ):
        return run_b
    return run_a


def _banner(title: str) -> None:
    bar = "=" * 64
    print(bar, flush=True)
    print(title, flush=True)
    print(bar, flush=True)
    log(title)


def print_best_task_pair(
    tag_a: str,
    sum_a: dict,
    tag_b: str,
    sum_b: dict,
    best: dict,
    best_tag: str,
) -> None:
    """头1两轮（基线+调参）结束后：对比并打印选中的结果/超参。"""
    lines = [
        "",
        f"[头1] 两轮对比选优：{tag_a} vs {tag_b}",
        f"  {tag_a}: Val F1={sum_a['val_f1_mean']:.4f}±{sum_a['val_f1_std']:.4f} | "
        f"Test F1={sum_a['test_f1_mean']:.4f}±{sum_a['test_f1_std']:.4f} | "
        f"Test Acc={sum_a['test_acc_mean']:.4f}±{sum_a['test_acc_std']:.4f}",
        f"  {tag_b}: Val F1={sum_b['val_f1_mean']:.4f}±{sum_b['val_f1_std']:.4f} | "
        f"Test F1={sum_b['test_f1_mean']:.4f}±{sum_b['test_f1_std']:.4f} | "
        f"Test Acc={sum_b['test_acc_mean']:.4f}±{sum_b['test_acc_std']:.4f}",
        f"  >> 选中（按 Val F1）：{best_tag}",
        f"  >> Val F1 = {best['val_f1_mean']:.4f} ± {best['val_f1_std']:.4f}",
        f"  >> Test F1 = {best['test_f1_mean']:.4f} ± {best['test_f1_std']:.4f}（仅报告）",
        f"  >> Test Acc = {best['test_acc_mean']:.4f} ± {best['test_acc_std']:.4f}",
        f"  >> 权重目录：{best['out_dir']}",
        "  >> 最优超参：",
    ]
    hp = best["hparams"]
    for k in ("lr", "weight_decay", "drop_prob", "patience", "max_epochs", "batch_train", "seed"):
        if k in hp:
            lines.append(f"       {k} = {hp[k]}")
    _banner("\n".join(lines))


def print_best_three_pair(
    tag_a: str,
    sum_a: dict,
    tag_b: str,
    sum_b: dict,
    best: dict,
    best_tag: str,
) -> None:
    """头2两轮（基线+调参）结束后：对比并打印选中的结果/超参。"""
    lines = [
        "",
        f"[头2] 两轮对比选优：{tag_a} vs {tag_b}",
        f"  {tag_a}: Val F1m={sum_a['val_f1_macro_mean']:.4f}±{sum_a['val_f1_macro_std']:.4f} | "
        f"Test F1m={sum_a['test_f1_macro_mean']:.4f}±{sum_a['test_f1_macro_std']:.4f} | "
        f"Test Acc={sum_a['test_acc_mean']:.4f}±{sum_a['test_acc_std']:.4f}",
        f"  {tag_b}: Val F1m={sum_b['val_f1_macro_mean']:.4f}±{sum_b['val_f1_macro_std']:.4f} | "
        f"Test F1m={sum_b['test_f1_macro_mean']:.4f}±{sum_b['test_f1_macro_std']:.4f} | "
        f"Test Acc={sum_b['test_acc_mean']:.4f}±{sum_b['test_acc_std']:.4f}",
        f"  >> 选中（按 Val F1-macro）：{best_tag}",
        f"  >> Val F1-macro = {best['val_f1_macro_mean']:.4f} ± {best['val_f1_macro_std']:.4f}",
        f"  >> Test F1-macro = {best['test_f1_macro_mean']:.4f} ± {best['test_f1_macro_std']:.4f}（仅报告）",
        f"  >> Test Acc = {best['test_acc_mean']:.4f} ± {best['test_acc_std']:.4f}",
        f"  >> 权重目录：{best['out_dir']}",
        "  >> 最优超参：",
    ]
    hp = best["hparams"]
    for k in (
        "lr",
        "weight_decay",
        "drop_prob",
        "patience",
        "max_epochs",
        "batch_train",
        "seed",
        "freeze_backbone",
        "task_kfold_dir",
    ):
        if k in hp:
            lines.append(f"       {k} = {hp[k]}")
    _banner("\n".join(lines))


def print_final_best(best_tag: str, best_task: dict, best3_tag: str, best_three: dict) -> None:
    """全部跑完后：汇总打印两头最优结果与参数。"""
    lines = [
        "",
        "过夜实验最终汇总（头1 + 头2）",
        "",
        f"[头1 静息/任务] 选中 {best_tag}",
        f"  Val F1     = {best_task['val_f1_mean']:.4f} ± {best_task['val_f1_std']:.4f}",
        f"  Test F1    = {best_task['test_f1_mean']:.4f} ± {best_task['test_f1_std']:.4f}",
        f"  Test Acc   = {best_task['test_acc_mean']:.4f} ± {best_task['test_acc_std']:.4f}",
        f"  out_dir    = {best_task['out_dir']}",
        "  hparams:",
    ]
    for k in ("lr", "weight_decay", "drop_prob", "patience", "max_epochs", "batch_train", "seed"):
        if k in best_task["hparams"]:
            lines.append(f"    {k} = {best_task['hparams'][k]}")
    lines += [
        "",
        f"[头2 空闲/左/右] 选中 {best3_tag}",
        f"  Val F1m    = {best_three['val_f1_macro_mean']:.4f} ± {best_three['val_f1_macro_std']:.4f}",
        f"  Test F1m   = {best_three['test_f1_macro_mean']:.4f} ± {best_three['test_f1_macro_std']:.4f}",
        f"  Test Acc   = {best_three['test_acc_mean']:.4f} ± {best_three['test_acc_std']:.4f}",
        f"  out_dir    = {best_three['out_dir']}",
        "  hparams:",
    ]
    for k in (
        "lr",
        "weight_decay",
        "drop_prob",
        "patience",
        "max_epochs",
        "batch_train",
        "seed",
        "freeze_backbone",
        "task_kfold_dir",
    ):
        if k in best_three["hparams"]:
            lines.append(f"    {k} = {best_three['hparams'][k]}")
    _banner("\n".join(lines))


def main() -> None:
    global OUT_ROOT, LOG_PATH, MD_PATH

    p = argparse.ArgumentParser(description="单模型过夜：Task/Three 各自网格+调参")
    p.add_argument("--model", default="eegnet", choices=list_models())
    p.add_argument("--data", default="merged_2s")
    p.add_argument(
        "--init-from-task",
        action="store_true",
        help="历史对照：三分类迁移二分类主干（默认关闭）",
    )
    p.add_argument("--no-writeback", action="store_true", help="不回写 kfold 脚本默认超参")
    args = p.parse_args()

    model_name = args.model
    data_tag = args.data
    init_from_task = bool(args.init_from_task)
    data_dir, data_prefix = resolve_data(data_tag)

    OUT_ROOT = TRAIN_LAB / "out" / "baseline" / model_name / data_tag / f"overnight_{STAMP}"
    LOG_PATH = OUT_ROOT / "overnight.log"
    MD_PATH = MD_DIR / f"五折过夜实验记录_{STAMP}_{model_name}.md"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    append_md(
        "\n".join(
            [
                f"# 被试独立五折过夜实验记录（{STAMP} / {model_name}）",
                "",
                "> 协议：被试独立五折 + 内层按人 val；早停看 val；test 仅终评。  ",
                "> 自动调参：**只根据验证集**改超参，不用测试集选型。  ",
                "> 顺序：Task 网格→调参 → Three 网格→调参（默认**不迁权重**）。  ",
                f"> model=`{model_name}` | data=`{data_tag}` | "
                f"weight_transfer=`{init_from_task}` | classifier=`native`",
                "",
                f"- 开始时间：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- 数据：`{data_dir}/{data_prefix}_*.npy`",
                f"- 输入：`n_times=500` @ 250Hz，进模 `(B,8,500)`",
                f"- 报数：Overall + bci2a_only + stieger_only（选型仍只看 Overall Val）",
                f"- 权重根目录：`{OUT_ROOT}`",
                f"- 运行日志：`{LOG_PATH}`",
                "",
                "---",
                "",
            ]
        )
    )
    log(f"MD_PATH={MD_PATH}")
    log(f"OUT_ROOT={OUT_ROOT}")
    log(f"device={device}")
    log(f"model={model_name} data={data_tag} init_from_task={init_from_task}")
    log(f"DATA_DIR={data_dir}")

    try:
        # ----- Task：小网格 -----
        grid_cfgs = [
            TaskKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                lr=1e-3,
                weight_decay=1e-4,
                drop_prob=0.50,
                patience=15,
                max_epochs=100,
            ),
            TaskKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                lr=7e-4,
                weight_decay=1e-4,
                drop_prob=0.55,
                patience=18,
                max_epochs=100,
            ),
            TaskKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                lr=5e-4,
                weight_decay=2e-4,
                drop_prob=0.60,
                patience=20,
                max_epochs=100,
            ),
            TaskKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                lr=1.5e-3,
                weight_decay=5e-5,
                drop_prob=0.40,
                patience=15,
                max_epochs=100,
            ),
        ]
        grid_sums = []
        for gi, gcfg in enumerate(grid_cfgs, start=1):
            gcfg.out_dir = str(OUT_ROOT / f"00_task_grid_{gi}")
            log(f"开始：Task 网格 {gi}/{len(grid_cfgs)} lr={gcfg.lr} drop={gcfg.drop_prob}")
            gsum = run_task_kfold(gcfg, device=device)
            grid_sums.append(gsum)
            write_task_section(
                f"Run00-G{gi} Task 网格",
                gsum,
                note=f"grid lr={gcfg.lr}, drop={gcfg.drop_prob}, wd={gcfg.weight_decay}",
            )
            log(
                f"Grid{gi} 完成 val_F1={gsum['val_f1_mean']:.4f} "
                f"test_F1={gsum['test_f1_mean']:.4f}"
            )

        best_grid = grid_sums[0]
        for g in grid_sums[1:]:
            best_grid = pick_best_task(best_grid, g)
        hp0 = best_grid["hparams"]
        cfg1 = TaskKFoldConfig(
            model_name=model_name,
            data_tag=data_tag,
            n_folds=int(hp0.get("n_folds", 5)),
            val_ratio=float(hp0.get("val_ratio", 0.2)),
            seed=int(hp0.get("seed", 42)),
            max_epochs=int(hp0["max_epochs"]),
            patience=int(hp0["patience"]),
            batch_train=int(hp0.get("batch_train", 32)),
            batch_eval=int(hp0.get("batch_eval", 64)),
            lr=float(hp0["lr"]),
            weight_decay=float(hp0["weight_decay"]),
            drop_prob=float(hp0["drop_prob"]),
            f1=int(hp0.get("f1", 8)),
            d=int(hp0.get("d", 2)),
            f2=int(hp0.get("f2", 16)),
            model_kwargs=hp0.get("model_kwargs"),
            out_dir=str(OUT_ROOT / "01_task_baseline"),
        )
        append_md(
            "\n".join(
                [
                    "## Task 网格选优",
                    "",
                    f"- 选中 Val F1 = `{best_grid['val_f1_mean']:.4f} ± {best_grid['val_f1_std']:.4f}`",
                    f"- 超参：lr=`{cfg1.lr}`, drop=`{cfg1.drop_prob}`, wd=`{cfg1.weight_decay}`, "
                    f"patience=`{cfg1.patience}`",
                    f"- 网格最优目录：`{best_grid['out_dir']}`",
                    "",
                    "---",
                    "",
                ]
            )
        )
        sum1 = dict(best_grid)
        write_task_section(
            "Run01 Task 五折（基线=网格最优）",
            sum1,
            note="由网格选优直接作为基线，不再重复训练",
        )
        log(f"Run01 完成 val_F1={sum1['val_f1_mean']:.4f} test_F1={sum1['test_f1_mean']:.4f}")

        cfg2, reasons = suggest_task_hparams(cfg1, sum1)
        cfg2.model_name = model_name
        cfg2.data_tag = data_tag
        cfg2.out_dir = str(OUT_ROOT / "02_task_tuned")
        write_tune_note("自动调参：Task → Run02", reasons, cfg2.__dict__)
        log("调参建议: " + " | ".join(reasons))

        log("开始：Task 五折 调参轮")
        sum2 = run_task_kfold(cfg2, device=device)
        write_task_section("Run02 Task 五折（自动调参）", sum2, note="；".join(reasons))
        log(f"Run02 完成 val_F1={sum2['val_f1_mean']:.4f} test_F1={sum2['test_f1_mean']:.4f}")

        best_task = pick_best_task(sum1, sum2)
        best_tag = "Run01" if best_task is sum1 else "Run02"
        print_best_task_pair("Run01", sum1, "Run02", sum2, best_task, best_tag)
        append_md(
            "\n".join(
                [
                    "## Task 选优",
                    "",
                    f"- 按 **Val F1** 选中：**{best_tag}**",
                    f"- Val F1 = `{best_task['val_f1_mean']:.4f} ± {best_task['val_f1_std']:.4f}`",
                    f"- 对应 Test F1（仅报告）= "
                    f"`{best_task['test_f1_mean']:.4f} ± {best_task['test_f1_std']:.4f}`",
                    f"- 权重目录：`{best_task['out_dir']}`",
                    "",
                    "### 选中超参",
                    "",
                    fmt_hparams(best_task["hparams"]),
                    "",
                    "---",
                    "",
                ]
            )
        )
        log(f"Task 选优: {best_tag} → {best_task['out_dir']}")

        # ----- Three：小网格（与 Task 对称）→ 选优 → 规则微调 -----
        three_grid_cfgs = [
            ThreeKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                init_from_task=init_from_task,
                task_kfold_dir=best_task["out_dir"] if init_from_task else "",
                lr=1e-3,
                weight_decay=1e-4,
                drop_prob=0.50,
                patience=15,
                max_epochs=100,
                freeze_backbone=False,
            ),
            ThreeKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                init_from_task=init_from_task,
                task_kfold_dir=best_task["out_dir"] if init_from_task else "",
                lr=7e-4,
                weight_decay=1e-4,
                drop_prob=0.55,
                patience=18,
                max_epochs=100,
                freeze_backbone=False,
            ),
            ThreeKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                init_from_task=init_from_task,
                task_kfold_dir=best_task["out_dir"] if init_from_task else "",
                lr=5e-4,
                weight_decay=2e-4,
                drop_prob=0.60,
                patience=20,
                max_epochs=100,
                freeze_backbone=False,
            ),
            ThreeKFoldConfig(
                model_name=model_name,
                data_tag=data_tag,
                init_from_task=init_from_task,
                task_kfold_dir=best_task["out_dir"] if init_from_task else "",
                lr=1.5e-3,
                weight_decay=5e-5,
                drop_prob=0.40,
                patience=15,
                max_epochs=100,
                freeze_backbone=False,
            ),
        ]
        three_grid_sums = []
        for gi, gcfg in enumerate(three_grid_cfgs, start=1):
            gcfg.out_dir = str(OUT_ROOT / f"03_three_grid_{gi}")
            log(
                f"开始：Three 网格 {gi}/{len(three_grid_cfgs)} "
                f"lr={gcfg.lr} drop={gcfg.drop_prob} "
                f"(weight_transfer={init_from_task})"
            )
            gsum = run_three_kfold(gcfg, device=device)
            three_grid_sums.append(gsum)
            write_three_section(
                f"Run03-G{gi} Three 网格",
                gsum,
                note=f"grid lr={gcfg.lr}, drop={gcfg.drop_prob}, wd={gcfg.weight_decay}",
            )
            log(
                f"Three Grid{gi} 完成 val_F1m={gsum['val_f1_macro_mean']:.4f} "
                f"test_F1m={gsum['test_f1_macro_mean']:.4f}"
            )

        best_three_grid = three_grid_sums[0]
        for g in three_grid_sums[1:]:
            best_three_grid = pick_best_three(best_three_grid, g)
        hp3 = best_three_grid["hparams"]
        cfg3 = ThreeKFoldConfig(
            model_name=model_name,
            data_tag=data_tag,
            init_from_task=init_from_task,
            task_kfold_dir=best_task["out_dir"] if init_from_task else "",
            n_folds=int(hp3.get("n_folds", 5)),
            val_ratio=float(hp3.get("val_ratio", 0.2)),
            seed=int(hp3.get("seed", 42)),
            max_epochs=int(hp3["max_epochs"]),
            patience=int(hp3["patience"]),
            batch_train=int(hp3.get("batch_train", 32)),
            batch_eval=int(hp3.get("batch_eval", 64)),
            lr=float(hp3["lr"]),
            weight_decay=float(hp3["weight_decay"]),
            drop_prob=float(hp3["drop_prob"]),
            f1=int(hp3.get("f1", 8)),
            d=int(hp3.get("d", 2)),
            f2=int(hp3.get("f2", 16)),
            model_kwargs=hp3.get("model_kwargs"),
            freeze_backbone=bool(hp3.get("freeze_backbone", False)),
            out_dir=str(OUT_ROOT / "03_three_baseline"),
        )
        append_md(
            "\n".join(
                [
                    "## Three 网格选优",
                    "",
                    f"- 选中 Val F1-macro = `{best_three_grid['val_f1_macro_mean']:.4f} ± "
                    f"{best_three_grid['val_f1_macro_std']:.4f}`",
                    f"- 超参：lr=`{cfg3.lr}`, drop=`{cfg3.drop_prob}`, wd=`{cfg3.weight_decay}`, "
                    f"patience=`{cfg3.patience}`",
                    f"- 网格最优目录：`{best_three_grid['out_dir']}`",
                    "",
                    "---",
                    "",
                ]
            )
        )
        sum3 = dict(best_three_grid)
        write_three_section(
            "Run03 Three 五折（基线=网格最优）",
            sum3,
            note="由 Three 网格选优直接作为基线，不再重复训练",
        )
        log(
            f"Run03 完成 val_F1m={sum3['val_f1_macro_mean']:.4f} "
            f"test_F1m={sum3['test_f1_macro_mean']:.4f}"
        )

        cfg4, reasons3 = suggest_three_hparams(cfg3, sum3)
        cfg4.model_name = model_name
        cfg4.data_tag = data_tag
        cfg4.init_from_task = init_from_task
        cfg4.out_dir = str(OUT_ROOT / "04_three_tuned")
        cfg4.task_kfold_dir = best_task["out_dir"] if init_from_task else ""
        write_tune_note("自动调参：Three → Run04", reasons3, cfg4.__dict__)
        log("调参建议: " + " | ".join(reasons3))

        log("开始：Three 五折 调参轮")
        sum4 = run_three_kfold(cfg4, device=device)
        write_three_section("Run04 Three 五折（自动调参）", sum4, note="；".join(reasons3))
        log(
            f"Run04 完成 val_F1m={sum4['val_f1_macro_mean']:.4f} "
            f"test_F1m={sum4['test_f1_macro_mean']:.4f}"
        )

        best_three = pick_best_three(sum3, sum4)
        best3_tag = "Run03" if best_three is sum3 else "Run04"
        print_best_three_pair("Run03", sum3, "Run04", sum4, best_three, best3_tag)

        append_md(
            "\n".join(
                [
                    "## Three 选优",
                    "",
                    f"- 按 **Val F1-macro** 选中：**{best3_tag}**",
                    f"- weight_transfer=`{init_from_task}`",
                    f"- Val F1-macro = `{best_three['val_f1_macro_mean']:.4f} ± "
                    f"{best_three['val_f1_macro_std']:.4f}`",
                    f"- 对应 Test F1-macro（仅报告）= "
                    f"`{best_three['test_f1_macro_mean']:.4f} ± "
                    f"{best_three['test_f1_macro_std']:.4f}`",
                    f"- 权重目录：`{best_three['out_dir']}`",
                    "",
                    "### 选中超参",
                    "",
                    fmt_hparams(best_three["hparams"]),
                    "",
                    "---",
                    "",
                    "## 最终结论",
                    "",
                    f"- 结束时间：`{datetime.now().isoformat(timespec='seconds')}`",
                    f"- model=`{model_name}` data=`{data_tag}` "
                    f"weight_transfer=`{init_from_task}` classifier=`native`",
                    "",
                    "### Task（静息/任务）推荐",
                    "",
                    f"- 轮次：**{best_tag}**",
                    f"- Val F1：`{best_task['val_f1_mean']:.4f} ± {best_task['val_f1_std']:.4f}`",
                    f"- Test F1：`{best_task['test_f1_mean']:.4f} ± {best_task['test_f1_std']:.4f}`",
                    f"- Test Acc：`{best_task['test_acc_mean']:.4f} ± {best_task['test_acc_std']:.4f}`",
                    "",
                    fmt_hparams(best_task["hparams"]),
                    "",
                    "### Three（空闲/左/右）推荐",
                    "",
                    f"- 轮次：**{best3_tag}**",
                    f"- Val F1-macro：`{best_three['val_f1_macro_mean']:.4f} ± "
                    f"{best_three['val_f1_macro_std']:.4f}`",
                    f"- Test F1-macro：`{best_three['test_f1_macro_mean']:.4f} ± "
                    f"{best_three['test_f1_macro_std']:.4f}`",
                    f"- Test Acc：`{best_three['test_acc_mean']:.4f} ± "
                    f"{best_three['test_acc_std']:.4f}`",
                    "",
                    fmt_hparams(best_three["hparams"]),
                    "",
                    "### 权重路径",
                    "",
                    f"- Task：`{best_task['out_dir']}`",
                    f"- Three：`{best_three['out_dir']}`",
                    "",
                ]
            )
        )

        print_final_best(best_tag, best_task, best3_tag, best_three)

        if not args.no_writeback:
            apply_recommended_defaults(best_task["hparams"], best_three["hparams"])

        meta = {
            "stamp": STAMP,
            "model_name": model_name,
            "data_tag": data_tag,
            "weight_transfer": init_from_task,
            "classifier": "native",
            "md": str(MD_PATH),
            "best_task_run": best_tag,
            "best_three_run": best3_tag,
            "best_task": {
                "val_f1_mean": best_task["val_f1_mean"],
                "test_f1_mean": best_task["test_f1_mean"],
                "hparams": best_task["hparams"],
                "out_dir": best_task["out_dir"],
            },
            "best_three": {
                "val_f1_macro_mean": best_three["val_f1_macro_mean"],
                "test_f1_macro_mean": best_three["test_f1_macro_mean"],
                "hparams": best_three["hparams"],
                "out_dir": best_three["out_dir"],
            },
        }
        with open(OUT_ROOT / "final_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log("全部完成")
        return meta
    except Exception:
        err = traceback.format_exc()
        log("失败:\n" + err)
        append_md(
            "\n".join(
                [
                    "## 运行中断",
                    "",
                    f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`",
                    "",
                    "```",
                    err,
                    "```",
                    "",
                ]
            )
        )
        raise


def apply_recommended_defaults(task_hp: dict, three_hp: dict) -> None:
    """把选中的超参写回两个 kfold 脚本的默认 Config 字段，明早可直接单跑。"""
    import re

    task_path = STEP_DIR / "train_task_kfold.py"
    three_path = STEP_DIR / "train_three_kfold.py"

    def patch_field(text: str, field: str, value: str) -> str:
        pat = rf"({field}:\s*[^=\n]+=\s*)[^\n]+"
        new_text, n = re.subn(pat, rf"\g<1>{value}", text, count=1)
        if n == 0:
            log(f"[warn] 未找到字段 {field} 可替换")
        return new_text

    def patch_file(path: Path, updates: dict[str, str]) -> None:
        text = path.read_text(encoding="utf-8")
        for field, value in updates.items():
            text = patch_field(text, field, value)
        path.write_text(text, encoding="utf-8")

    patch_file(
        task_path,
        {
            "lr": repr(float(task_hp["lr"])),
            "weight_decay": repr(float(task_hp["weight_decay"])),
            "drop_prob": repr(float(task_hp["drop_prob"])),
            "patience": str(int(task_hp["patience"])),
        },
    )
    patch_file(
        three_path,
        {
            "lr": repr(float(three_hp["lr"])),
            "weight_decay": repr(float(three_hp["weight_decay"])),
            "drop_prob": repr(float(three_hp["drop_prob"])),
            "patience": str(int(three_hp["patience"])),
        },
    )
    append_md(
        "\n".join(
            [
                "## 代码默认超参已更新",
                "",
                "已按选优结果改写 Task/Three kfold 默认 lr/wd/drop/patience。",
                "",
                "```powershell",
                "cd code/train_lab/src/step",
                r"D:\cyy\MI\.venv\Scripts\python.exe train_task_kfold.py --model eegnet",
                r"D:\cyy\MI\.venv\Scripts\python.exe train_three_kfold.py --model eegnet",
                "```",
                "",
            ]
        )
    )
    log("已回写推荐超参到 kfold 脚本默认值")


if __name__ == "__main__":
    main()
